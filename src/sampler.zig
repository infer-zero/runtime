//! Sampler interface.

vtable: *const VTable,

pub const Options = struct {
    /// 0.0 = greedy, deterministic; higher is more "creative"
    temperature: f32,
    top_k: usize,
    /// Nucleus sampling: keep smallest set of tokens with cumulative prob >= top_p (1.0 = disabled)
    top_p: f32,
    /// Keep tokens with prob >= min_p * max_prob (0.0 = disabled)
    min_p: f32,
    /// 1.0 = no penalty, >1.0 = penalize repetition
    repetition_penalty: f32,
    repetition_penalty_last_n: u32,

    pub const default: @This() = .{
        .temperature = 0.8,
        .top_k = 40,
        .top_p = 0.95,
        .min_p = 0.05,
        .repetition_penalty = 1.1,
        .repetition_penalty_last_n = 64,
    };
};

pub const VTable = struct {
    /// Sample a token from the logits distribution.
    /// May modify `logits` in place (the Default does).
    sample: *const fn (*Sampler, logits: []f32, history: []const TokenID, options: Options) TokenID,
};

/// Sample a token from the logits distribution.
/// May modify `logits` in place (the Default does).
pub fn sample(
    self: *@This(),
    logits: []f32,
    history: []const TokenID,
    options: Options,
) TokenID {
    return self.vtable.sample(self, logits, history, options);
}

const Sampler = @This();
pub const TokenID = u32;
pub const Default = @import("sampler_default.zig");

pub const Profile = enum { base, instruct_non_thinking, instruct_thinking };
