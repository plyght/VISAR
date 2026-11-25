//! visar.photo: computational photography (OpenCV's photo module):
//! non-local means denoising, inpainting and the domain-transform filters.

const std = @import("std");

pub const nlmeans = @import("nlmeans.zig");
pub const NlMeansOptions = nlmeans.Options;
pub const fastNlMeansDenoising = nlmeans.fastNlMeansDenoising;
pub const fastNlMeansDenoisingColored = nlmeans.fastNlMeansDenoisingColored;

pub const inpaint_mod = @import("inpaint.zig");
pub const InpaintMethod = inpaint_mod.Method;
pub const inpaint = inpaint_mod.inpaint;

pub const npr = @import("npr.zig");
pub const EdgePreservingFlags = npr.Flags;
pub const edgePreservingFilter = npr.edgePreservingFilter;
pub const stylization = npr.stylization;
pub const pencilSketch = npr.pencilSketch;

test {
    std.testing.refAllDecls(@This());
}
