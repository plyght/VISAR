//! Execution: `Exec` decides how a kernel's rows are spread over threads.
//! `Exec.serial` runs everything on the calling thread (the only option on
//! freestanding and single-threaded targets); `Exec.on(&pool)` uses a
//! persistent fork-join pool of resident workers that spin briefly and then
//! sleep on a futex between jobs, so a dispatch costs a few microseconds.
//! Derived from ditch's `tensor.Pool`.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Allocator = std.mem.Allocator;

pub const threads_supported = !builtin.single_threaded and builtin.os.tag != .freestanding and builtin.os.tag != .wasi;

pub const Exec = struct {
    pool: ?*const Pool = null,

    pub const serial: Exec = .{};

    pub fn on(pool: *const Pool) Exec {
        return .{ .pool = pool };
    }

    pub fn threads(self: Exec) usize {
        return if (self.pool) |p| p.threads else 1;
    }

    /// Runs `func(ctx, start, end)` over `[0, n)`. Each chunk has at least
    /// `grain` items (so tiny jobs stay on one thread); chunks are claimed
    /// dynamically, so uneven rows balance out.
    pub fn parallelFor(self: Exec, n: usize, grain: usize, ctx: anytype, comptime func: fn (@TypeOf(ctx), usize, usize) void) void {
        if (n == 0) return;
        const pool = self.pool orelse return func(ctx, 0, n);
        const max_chunks = @max(1, n / @max(1, grain));
        const chunks = @min(max_chunks, pool.threads * 4);
        if (chunks <= 1 or pool.threads <= 1) return func(ctx, 0, n);
        pool.run(n, chunks, ctx, func);
    }
};

pub const Pool = struct {
    threads: usize,
    workers: ?*Workers = null,

    /// A pool using `threads` threads in total (the caller counts as one);
    /// null means one per logical CPU. Falls back to serial execution when
    /// threads cannot be spawned.
    pub fn init(gpa: Allocator, io: Io, threads: ?usize) Pool {
        if (!threads_supported) return .{ .threads = 1 };
        const n = @max(1, threads orelse (std.Thread.getCpuCount() catch 1));
        var pool: Pool = .{ .threads = n };
        if (n > 1) pool.workers = Workers.spawn(gpa, io, n - 1) catch null;
        if (pool.workers == null) pool.threads = 1;
        return pool;
    }

    pub fn deinit(self: *Pool) void {
        if (self.workers) |w| w.shutdown();
        self.* = undefined;
    }

    fn run(self: *const Pool, n: usize, chunks: usize, ctx: anytype, comptime func: fn (@TypeOf(ctx), usize, usize) void) void {
        const Ctx = @TypeOf(ctx);
        const w = self.workers orelse return func(ctx, 0, n);
        const Tramp = struct {
            fn call(p: *const anyopaque, s: usize, e: usize) void {
                const c: *const Ctx = @ptrCast(@alignCast(p));
                func(c.*, s, e);
            }
        };
        w.mutex.lockUncancelable(w.io);
        defer w.mutex.unlock(w.io);
        w.dispatch(.{ .func = Tramp.call, .ctx = @ptrCast(&ctx), .n = n, .chunks = chunks });
    }
};

const Job = struct {
    func: *const fn (*const anyopaque, usize, usize) void,
    ctx: *const anyopaque,
    n: usize,
    chunks: usize,
};

/// Resident worker threads. A job is published by bumping `generation`;
/// every thread (workers and the caller) claims chunks through `next` until
/// the job is exhausted, and the caller returns when `remaining` reaches zero.
const Workers = struct {
    gpa: Allocator,
    io: Io,
    threads: []std.Thread,
    /// Serialises jobs when several threads share one pool.
    mutex: Io.Mutex = .init,
    generation: std.atomic.Value(u32) = .init(0),
    remaining: std.atomic.Value(u32) = .init(0),
    next: std.atomic.Value(usize) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    job: Job = undefined,

    const spin_iterations = 2000;

    fn spawn(gpa: Allocator, io: Io, count: usize) !*Workers {
        const self = try gpa.create(Workers);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .io = io, .threads = &.{} };
        const threads = try gpa.alloc(std.Thread, count);
        errdefer gpa.free(threads);
        var spawned: usize = 0;
        errdefer {
            self.stop.store(true, .release);
            _ = self.generation.fetchAdd(1, .release);
            self.io.futexWake(u32, &self.generation.raw, std.math.maxInt(u32));
            for (threads[0..spawned]) |t| t.join();
        }
        // Read the baseline here, not in the new thread, so a worker still
        // starting when the first job is published does not skip it.
        const first_generation = self.generation.load(.acquire);
        for (threads) |*t| {
            t.* = try std.Thread.spawn(.{}, workerMain, .{ self, first_generation });
            spawned += 1;
        }
        self.threads = threads;
        return self;
    }

    fn shutdown(self: *Workers) void {
        self.stop.store(true, .release);
        _ = self.generation.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.generation.raw, std.math.maxInt(u32));
        for (self.threads) |t| t.join();
        self.gpa.free(self.threads);
        self.gpa.destroy(self);
    }

    fn dispatch(self: *Workers, job: Job) void {
        self.job = job;
        self.next.store(0, .monotonic);
        self.remaining.store(@intCast(self.threads.len), .monotonic);
        _ = self.generation.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.generation.raw, std.math.maxInt(u32));
        self.work();
        var spins: usize = 0;
        while (true) {
            const r = self.remaining.load(.acquire);
            if (r == 0) break;
            if (spins < spin_iterations) {
                spins += 1;
                std.atomic.spinLoopHint();
            } else self.io.futexWaitUncancelable(u32, &self.remaining.raw, r);
        }
    }

    fn work(self: *Workers) void {
        const job = self.job;
        const per = (job.n + job.chunks - 1) / job.chunks;
        while (true) {
            const c = self.next.fetchAdd(1, .monotonic);
            if (c >= job.chunks) break;
            const start = c * per;
            if (start >= job.n) break;
            job.func(job.ctx, start, @min(job.n, start + per));
        }
    }

    fn workerMain(self: *Workers, first_generation: u32) void {
        var seen = first_generation;
        while (true) {
            var spins: usize = 0;
            while (true) {
                const g = self.generation.load(.acquire);
                if (g != seen) {
                    seen = g;
                    break;
                }
                if (spins < spin_iterations) {
                    spins += 1;
                    std.atomic.spinLoopHint();
                } else self.io.futexWaitUncancelable(u32, &self.generation.raw, g);
            }
            if (self.stop.load(.acquire)) return;
            self.work();
            if (self.remaining.fetchSub(1, .acq_rel) == 1)
                self.io.futexWake(u32, &self.remaining.raw, 1);
        }
    }
};

test "parallelFor covers the range exactly once" {
    const gpa = std.testing.allocator;
    var pool = Pool.init(gpa, std.testing.io, 4);
    defer pool.deinit();
    const n = 10_000;
    const hits = try gpa.alloc(std.atomic.Value(u32), n);
    defer gpa.free(hits);
    for (hits) |*h| h.* = .init(0);
    const Ctx = struct { hits: []std.atomic.Value(u32) };
    for ([_]Exec{ .serial, .on(&pool) }) |exec| {
        for (0..3) |_| exec.parallelFor(n, 16, Ctx{ .hits = hits }, struct {
            fn f(c: Ctx, s: usize, e: usize) void {
                for (s..e) |i| _ = c.hits[i].fetchAdd(1, .monotonic);
            }
        }.f);
    }
    for (hits) |h| try std.testing.expectEqual(@as(u32, 6), h.load(.monotonic));
}
