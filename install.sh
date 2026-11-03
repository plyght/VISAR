#!/bin/sh
# VISAR installer for macOS and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/plyght/VISAR/main/install.sh | sh
#
# Picks the build for this machine (the AVX2 one on x86-64 CPUs that have
# it), checks it against the release's SHA256SUMS and puts `visar` in
# ~/.local/bin. Settings, all optional:
#
#   VISAR_VERSION=v0.2.0      a release tag, or "nightly" (default: the latest
#                             release, else the nightly build of main)
#   VISAR_INSTALL_DIR=DIR     where to put visar (default: ~/.local/bin)
#   VISAR_BASELINE=1          the portable x86-64 build even when AVX2 is there
#   NO_COLOR=1                plain output
#
#   curl -fsSL .../install.sh | sh -s -- --uninstall
#
# Everything runs from main at the bottom, so a download cut short runs nothing.

set -eu

REPO="plyght/VISAR"

if [ -t 2 ] && [ -z "${NO_COLOR:-}" ]; then
	BOLD=$(printf '\033[1m') DIM=$(printf '\033[2m') RED=$(printf '\033[31m')
	GREEN=$(printf '\033[32m') YELLOW=$(printf '\033[33m') CYAN=$(printf '\033[36m')
	AMBER=$(printf '\033[38;5;214m') RESET=$(printf '\033[0m')
else
	BOLD='' DIM='' RED='' GREEN='' YELLOW='' CYAN='' AMBER='' RESET=''
fi

step() { printf '%s›%s %s\n' "$CYAN" "$RESET" "$*" >&2; }
info() { printf '  %s%s%s\n' "$DIM" "$*" "$RESET" >&2; }
warn() { printf '%swarning:%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die() {
	printf '%serror:%s %s\n' "$RED" "$RESET" "$*" >&2
	exit 1
}

banner() {
	printf '\n  %s┌╴%s %svisar%s %s0.99%s\n' "$CYAN" "$RESET" "$BOLD" "$RESET" "$AMBER" "$RESET" >&2
	printf '  %s└╴%s %scomputer vision in pure Zig%s\n\n' "$CYAN" "$RESET" "$DIM" "$RESET" >&2
}

have() { command -v "$1" >/dev/null 2>&1; }

fetch() {
	if have curl; then
		curl -fsSL --retry 3 --proto '=https' --tlsv1.2 -o "$2" "$1"
	elif have wget; then
		wget -q --https-only -O "$2" "$1"
	else
		die "curl or wget is needed to download visar"
	fi
}

# The latest release's tag from the /releases/latest redirect (no API token),
# or nothing when no stable release exists yet.
latest_tag() {
	url="https://github.com/$REPO/releases/latest"
	if have curl; then
		final=$(curl -fsSLI --retry 3 -o /dev/null -w '%{url_effective}' "$url" 2>/dev/null) || final=""
	else
		final=$(wget -q --max-redirect=5 -S --spider "$url" 2>&1 | sed -n 's/^ *[Ll]ocation: *//p' | tail -n 1 | tr -d '\r')
	fi
	tag=${final##*/}
	case "$tag" in v*) printf '%s\n' "$tag" ;; *) printf '\n' ;; esac
}

has_x86_64_v3() {
	case "$1" in
	linux)
		[ -r /proc/cpuinfo ] || return 1
		flags=$(grep -m 1 '^flags' /proc/cpuinfo) || return 1
		for f in avx2 fma bmi1 bmi2 f16c movbe abm; do
			case " $flags " in *" $f "*) ;; *) return 1 ;; esac
		done
		;;
	macos)
		feats="$(sysctl -n machdep.cpu.features machdep.cpu.leaf7_features machdep.cpu.extfeatures 2>/dev/null | tr '\n' ' ')"
		for f in AVX2 FMA BMI1 BMI2 F16C MOVBE LZCNT; do
			case " $feats " in *" $f "*) ;; *) return 1 ;; esac
		done
		;;
	*) return 1 ;;
	esac
}

detect_target() {
	case "$(uname -s)" in
	Linux) os=linux ;;
	Darwin) os=macos ;;
	*) die "no prebuilt visar for $(uname -s); build from source: https://github.com/$REPO#build" ;;
	esac
	case "$(uname -m)" in
	x86_64 | amd64) arch=x86_64 ;;
	aarch64 | arm64) arch=aarch64 ;;
	*) die "no prebuilt visar for $(uname -m); build from source: https://github.com/$REPO#build" ;;
	esac
	# An x86-64 shell under Rosetta on Apple Silicon: the native build is faster.
	if [ "$os" = macos ] && [ "$arch" = x86_64 ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
		arch=aarch64
	fi
	case "$os" in
	linux) TARGET="$arch-linux-musl" ;;
	macos) TARGET="$arch-macos" ;;
	esac
	VARIANT=""
	CPU_NOTE="NEON build"
	if [ "$arch" = x86_64 ]; then
		if [ -n "${VISAR_BASELINE:-}" ]; then
			CPU_NOTE="portable x86-64 build (VISAR_BASELINE set)"
		elif has_x86_64_v3 "$os"; then
			VARIANT="-v3"
			CPU_NOTE="AVX2 + FMA build (x86-64-v3)"
		else
			CPU_NOTE="portable x86-64 build (this CPU has no AVX2)"
		fi
	fi
}

sha256_of() {
	if have sha256sum; then
		sha256sum "$1" | cut -d ' ' -f 1
	elif have shasum; then
		shasum -a 256 "$1" | cut -d ' ' -f 1
	elif have openssl; then
		openssl dgst -sha256 -r "$1" | cut -d ' ' -f 1
	else
		return 1
	fi
}

uninstall() {
	dir="${VISAR_INSTALL_DIR:-$HOME/.local/bin}"
	step "Removing visar"
	if [ -e "$dir/visar" ]; then
		rm -f "$dir/visar"
		info "removed $dir/visar"
	else
		info "no $dir/visar"
	fi
	printf '%sdone%s\n' "$GREEN" "$RESET" >&2
}

main() {
	for arg in "$@"; do
		case "$arg" in
		--uninstall)
			uninstall
			return
			;;
		-h | --help)
			sed -n '2,18p' "$0" 2>/dev/null >&2 || true
			return
			;;
		*) die "unknown option: $arg" ;;
		esac
	done

	banner
	detect_target
	tag="${VISAR_VERSION:-}"
	if [ -z "$tag" ]; then
		step "Finding the latest release"
		tag=$(latest_tag)
		if [ -z "$tag" ]; then
			tag=nightly
			info "no stable release yet; using the nightly build of main"
		fi
	fi
	case "$tag" in v* | nightly) ;; *) tag="v$tag" ;; esac
	name="visar-$tag-$TARGET$VARIANT"
	base="https://github.com/$REPO/releases/download/$tag"
	info "$tag for $TARGET, $CPU_NOTE"

	tmp=$(mktemp -d 2>/dev/null || mktemp -d -t visar)
	trap 'rm -rf "$tmp"' EXIT INT TERM

	step "Downloading $name.tar.gz"
	fetch "$base/$name.tar.gz" "$tmp/$name.tar.gz" || die "download failed: $base/$name.tar.gz"

	step "Verifying the checksum"
	fetch "$base/SHA256SUMS" "$tmp/SHA256SUMS" || die "could not download SHA256SUMS for $tag"
	want=$(grep " $name.tar.gz\$" "$tmp/SHA256SUMS" | cut -d ' ' -f 1)
	[ -n "$want" ] || die "$name.tar.gz is not listed in SHA256SUMS"
	got=$(sha256_of "$tmp/$name.tar.gz") || die "no sha256sum, shasum or openssl to verify the download"
	[ "$want" = "$got" ] || die "checksum mismatch for $name.tar.gz (expected $want, got $got)"

	tar -xzf "$tmp/$name.tar.gz" -C "$tmp"
	dir="${VISAR_INSTALL_DIR:-$HOME/.local/bin}"
	mkdir -p "$dir"
	install_path="$dir/visar"
	mv "$tmp/$name/visar" "$install_path.new"
	chmod 755 "$install_path.new"
	mv -f "$install_path.new" "$install_path"
	# Downloaded with curl, so no quarantine attribute; clear it anyway in
	# case a browser fetched the archive.
	if [ "$os" = macos ] && have xattr; then xattr -d com.apple.quarantine "$install_path" 2>/dev/null || true; fi

	printf '%s✓%s installed %s%s%s\n' "$GREEN" "$RESET" "$BOLD" "$install_path" "$RESET" >&2
	case ":$PATH:" in
	*":$dir:"*) ;;
	*) warn "$dir is not on your PATH; add it, e.g.: echo 'export PATH=\"$dir:\$PATH\"' >> ~/.zshrc" ;;
	esac
	printf '\n  try: %svisar track%s   (live camera tracking)\n       %svisar --help%s\n\n' "$BOLD" "$RESET" "$BOLD" "$RESET" >&2
}

main "$@"
