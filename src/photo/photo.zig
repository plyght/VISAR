//! visar.photo: computational photography (OpenCV's photo module):
//! non-local means denoising and inpainting.

const std = @import("std");

pub const nlmeans = @import("nlmeans.zig");
pub const NlMeansOptions = nlmeans.Options;
pub const fastNlMeansDenoising = nlmeans.fastNlMeansDenoising;
pub const fastNlMeansDenoisingColored = nlmeans.fastNlMeansDenoisingColored;

pub const inpaint_mod = @import("inpaint.zig");
pub const InpaintMethod = inpaint_mod.Method;
pub const inpaint = inpaint_mod.inpaint;

test {
    std.testing.refAllDecls(@This());
}
