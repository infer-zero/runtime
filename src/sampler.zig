//! Token sampling strategies for text generation.
//!
//! Supports temperature scaling, top-k, top-p (nucleus), min-p filtering,
//! and repetition penalty.

options: Options,
rng: std.Random.DefaultPrng,

pub const TokenID = u32;

pub const Options = struct {
    temperature: f32,
    top_k: usize,
    top_p: f32, // Nucleus sampling: keep smallest set of tokens with cumulative prob >= top_p (1.0 = disabled)
    min_p: f32, // Keep tokens with prob >= min_p * max_prob (0.0 = disabled)
    repetition_penalty: f32, // 1.0 = no penalty, >1.0 = penalize repetition
    /// Window over which the repetition penalty looks back. Matches
    /// llama.cpp's `repeat_last_n` default. Penalty is applied once per
    /// UNIQUE token in this window (not per occurrence) — without this,
    /// common tokens like " " accumulate penalty^N where N is their
    /// count in history, which crushes them on long contexts and produces
    /// the multilingual / missing-space drift we observed at rep=1.1.
    repetition_penalty_last_n: u32,
    seed: ?u64,

    pub const default: @This() = .{
        .temperature = 0.8,
        .top_k = 40,
        .top_p = 0.95,
        .min_p = 0.05,
        .repetition_penalty = 1.1,
        .repetition_penalty_last_n = 64,
        .seed = null,
    };
};

/// Sampling regime requested by a caller. Wider than loader-level
/// `ModelKind` because hybrid Qwen3 checkpoints support both thinking
/// and non-thinking modes from one file with different recipes per
/// mode.
pub const Profile = enum { base, instruct_non_thinking, instruct_thinking };

/// Per-family preset table. Each field is optional — a family only
/// fills the profiles its HF card publishes a recipe for.
pub const Presets = struct {
    base: ?Options = null,
    instruct_non_thinking: ?Options = null,
    instruct_thinking: ?Options = null,

    pub fn get(self: Presets, profile: Profile) ?Options {
        return switch (profile) {
            .base => self.base,
            .instruct_non_thinking => self.instruct_non_thinking,
            .instruct_thinking => self.instruct_thinking,
        };
    }
};

pub fn init(io: std.Io, options: Options) @This() {
    const seed = options.seed orelse blk: {
        var seed_bytes: [8]u8 = undefined;
        io.random(&seed_bytes);
        break :blk std.mem.readInt(u64, &seed_bytes, .little);
    };
    return .{
        .options = options,
        .rng = std.Random.DefaultPrng.init(seed),
    };
}

/// Sample a token from the logits distribution using `self.options`.
/// Takes logits buffer and history for repetition penalty.
/// Note: This function modifies the logits array in place.
pub fn sample(
    self: *@This(),
    logits: []f32,
    history: []const TokenID,
) TokenID {
    return self.sampleWith(logits, history, self.options);
}

/// Sample a token using an explicit `Options` instead of `self.options`.
/// One-shot — `self.options` is not mutated. RNG is shared with `sample`.
pub fn sampleWith(
    self: *@This(),
    logits: []f32,
    history: []const TokenID,
    options: Options,
) TokenID {
    // Apply repetition penalty ONCE per unique token in the last
    // `repetition_penalty_last_n` history positions. Matches llama.cpp's
    // semantics. The earlier implementation iterated all history and
    // applied per-occurrence, so common tokens (spaces, punctuation) got
    // logit / penalty^count and were crushed below alternatives — which
    // surfaced as multilingual / missing-space drift on long-context
    // generation, especially at higher penalty values like 1.1.
    if (options.repetition_penalty != 1.0 and history.len > 0) {
        const last_n = @as(usize, options.repetition_penalty_last_n);
        const start = if (history.len > last_n) history.len - last_n else 0;
        const window = history[start..];

        for (window, 0..) |token_id, i| {
            if (token_id >= logits.len) continue;
            // Skip if this token was already penalized earlier in the window.
            // O(N^2) over the window (default N=64 → 4096 ops/sample);
            // negligible vs. the model forward pass.
            var already_seen = false;
            for (window[0..i]) |earlier| {
                if (earlier == token_id) {
                    already_seen = true;
                    break;
                }
            }
            if (already_seen) continue;
            const logit = logits[token_id];
            if (logit > 0) {
                logits[token_id] = logit / options.repetition_penalty;
            } else {
                logits[token_id] = logit * options.repetition_penalty;
            }
        }
    }

    // Greedy decoding when temperature is 0
    if (options.temperature == 0.0) {
        return argmax(logits);
    }

    // Top-k filtering: keep only the top-k logits
    if (options.top_k > 0 and options.top_k < logits.len) {
        const threshold = findTopKThreshold(logits, options.top_k);
        for (logits) |*logit| {
            if (logit.* < threshold) {
                logit.* = -1e10; // Very negative but not -inf to avoid NaN in softmax
            }
        }
    }

    // Apply temperature scaling
    for (logits) |*logit| {
        logit.* /= options.temperature;
    }

    // Apply softmax to get probabilities
    softmax(logits);

    // Apply min-p filtering: keep tokens with prob >= min_p * max_prob
    if (options.min_p > 0.0) {
        var max_prob: f32 = 0.0;
        for (logits) |prob| {
            max_prob = @max(max_prob, prob);
        }
        const min_threshold = options.min_p * max_prob;
        for (logits) |*prob| {
            if (prob.* < min_threshold) {
                prob.* = 0.0;
            }
        }
    }

    // Apply top-p (nucleus) filtering: keep smallest set of tokens with cumulative prob >= top_p
    if (options.top_p < 1.0) {
        applyTopP(logits, options.top_p);
    }

    // Sample from the probability distribution (handles unnormalized probs)
    var sum: f32 = 0.0;
    for (logits) |prob| {
        sum += prob;
    }

    if (sum == 0.0) {
        return argmax(logits);
    }

    const rand_val = self.rng.random().float(f32) * sum;
    var cumulative: f32 = 0.0;
    for (logits, 0..) |prob, token_index| {
        cumulative += prob;
        if (rand_val < cumulative) {
            return @intCast(token_index);
        }
    }

    // Fallback to argmax (shouldn't normally happen)
    return argmax(logits);
}

fn argmax(values: []const f32) TokenID {
    var max_value = values[0];
    var max_index: usize = 0;
    for (values, 0..) |value, index| {
        if (value > max_value) {
            max_value = value;
            max_index = index;
        }
    }
    return @intCast(max_index);
}

/// Find the k-th largest value in the array (threshold for top-k)
fn findTopKThreshold(values: []const f32, k: usize) f32 {
    if (k == 0) return std.math.floatMax(f32);
    if (k >= values.len) return -std.math.floatMax(f32);

    // Simple O(n*k) algorithm: maintain top-k values
    const max_top_k = 64;
    var top_values: [max_top_k]f32 = undefined;
    const actual_k = @min(k, max_top_k);

    // Initialize with very negative values
    for (0..actual_k) |slot| {
        top_values[slot] = -std.math.floatMax(f32);
    }

    // Track the minimum of our top-k values
    var min_top: f32 = -std.math.floatMax(f32);
    var min_idx: usize = 0;

    for (values) |value| {
        if (value > min_top) {
            // Replace the minimum in top_values
            top_values[min_idx] = value;

            // Find new minimum
            min_top = top_values[0];
            min_idx = 0;
            for (1..actual_k) |slot| {
                if (top_values[slot] < min_top) {
                    min_top = top_values[slot];
                    min_idx = slot;
                }
            }
        }
    }

    return min_top;
}

fn softmax(values: []f32) void {
    if (values.len == 0) return;

    // Find max for numerical stability
    var max_value = values[0];
    for (values[1..]) |value| {
        if (!std.math.isNan(value)) {
            max_value = @max(max_value, value);
        }
    }

    // Compute exp and sum
    var sum_exp: f32 = 0.0;
    for (values) |*value| {
        if (std.math.isNan(value.*)) {
            value.* = 0.0;
            continue;
        }
        value.* = @exp(value.* - max_value);
        sum_exp += value.*;
    }

    // Prevent division by zero
    if (sum_exp == 0.0 or std.math.isNan(sum_exp)) {
        // Uniform distribution fallback
        const uniform = 1.0 / @as(f32, @floatFromInt(values.len));
        for (values) |*value| {
            value.* = uniform;
        }
        return;
    }

    // Normalize to probabilities
    for (values) |*value| {
        value.* /= sum_exp;
        if (std.math.isNan(value.*)) {
            value.* = 0.0;
        }
    }
}

/// Apply top-p (nucleus) sampling: keep smallest set of tokens with cumulative prob >= top_p
/// Zeros out probabilities of tokens outside the nucleus.
fn applyTopP(probs: []f32, top_p: f32) void {
    if (probs.len == 0) return;

    // Iteratively find max prob tokens until we hit the cumulative threshold
    var cumulative: f32 = 0.0;

    while (cumulative < top_p) {
        // Find the token with max probability that hasn't been "selected" yet
        var max_prob: f32 = 0.0;
        var max_idx: usize = 0;
        var found = false;

        for (probs, 0..) |prob, idx| {
            if (prob > max_prob) {
                max_prob = prob;
                max_idx = idx;
                found = true;
            }
        }

        if (!found or max_prob == 0.0) break;

        // Add to cumulative and mark as selected by making it negative temporarily
        cumulative += max_prob;
        probs[max_idx] = -max_prob;
    }

    // Restore selected probabilities (negative) and zero out unselected (positive)
    for (probs) |*prob| {
        if (prob.* < 0.0) {
            prob.* = -prob.*;
        } else {
            prob.* = 0.0;
        }
    }
}

test "argmax" {
    const values = [_]f32{ 1.0, 3.0, 2.0, 0.5 };
    try std.testing.expectEqual(@as(TokenID, 1), argmax(&values));
}

test "softmax produces valid distribution" {
    var values = [_]f32{ 1.0, 2.0, 3.0 };
    softmax(&values);

    var sum: f32 = 0.0;
    for (values) |value| sum += value;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
}

test "greedy sampling" {
    var opts = Options.default;
    opts.temperature = 0.0;
    var sampler = init(std.testing.io, opts);
    var logits = [_]f32{ 1.0, 5.0, 2.0, 3.0 };
    const token = sampler.sample(&logits, &.{});
    try std.testing.expectEqual(@as(TokenID, 1), token);
}

test "temperature scaling" {
    // Higher temperature = more uniform distribution
    var opts_low = Options.default;
    opts_low.temperature = 0.1;
    opts_low.seed = 42;
    var sampler_low = init(std.testing.io, opts_low);

    var opts_high = Options.default;
    opts_high.temperature = 2.0;
    opts_high.seed = 42;
    var sampler_high = init(std.testing.io, opts_high);

    var logits1 = [_]f32{ 1.0, 5.0, 2.0, 3.0 };
    var logits2 = [_]f32{ 1.0, 5.0, 2.0, 3.0 };

    // Low temp should almost always pick the highest
    const token_low = sampler_low.sample(&logits1, &.{});
    _ = sampler_high.sample(&logits2, &.{});

    // With very low temperature, should pick argmax
    try std.testing.expectEqual(@as(TokenID, 1), token_low);
}

test "top_k filtering" {
    var opts = Options.default;
    opts.top_k = 2;
    opts.temperature = 1.0;
    opts.seed = 42;
    var sampler = init(std.testing.io, opts);
    const logits = [_]f32{ 1.0, 5.0, 4.0, 0.5 };

    // Run multiple times - should only ever pick from top 2 (indices 1 and 2)
    var trial: usize = 0;
    while (trial < 10) : (trial += 1) {
        var logits_copy = logits;
        const token = sampler.sample(&logits_copy, &.{});
        try std.testing.expect(token == 1 or token == 2);
    }
}

test "repetition penalty" {
    var opts = Options.default;
    opts.temperature = 0.0;
    opts.repetition_penalty = 100.0;
    var sampler = init(std.testing.io, opts);
    const logits = [_]f32{ 1.0, 5.0, 4.9, 0.5 };

    // Without history, picks token 1 (highest)
    var logits1 = logits;
    const token1 = sampler.sample(&logits1, &.{});
    try std.testing.expectEqual(@as(TokenID, 1), token1);

    // With token 1 in history and high penalty, should pick token 2
    var logits2 = logits;
    const history = [_]TokenID{1};
    const token2 = sampler.sample(&logits2, &history);
    try std.testing.expectEqual(@as(TokenID, 2), token2);
}

// Regression: a token appearing N times in history must be penalized
// ONCE (per-unique semantics), not N times. The earlier per-occurrence
// implementation crushed common tokens (spaces, punctuation) with
// `logit / penalty^N` on long contexts and produced the multilingual
// drift on Qwen3-4B at rep=1.1. The penalty (2.0) is small enough that
// per-unique leaves token 1 above token 2 (5/2 = 2.5 > 2.0); the
// per-occurrence broken behavior would compound it across the 50
// repeats to (5/2^50 ≪ 2.0) and pick token 2.
test "repetition penalty: per-unique, not per-occurrence" {
    var opts = Options.default;
    opts.temperature = 0.0;
    opts.repetition_penalty = 2.0;
    opts.repetition_penalty_last_n = 64;
    var sampler = init(std.testing.io, opts);
    const logits = [_]f32{ 1.0, 5.0, 2.0, 0.5 };

    // History has token 1 repeated 50 times. Per-unique penalty: 5/2 = 2.5
    // (still highest). Per-occurrence (broken): 5/2^50 ≈ 0 (token 2 wins).
    var history: [50]TokenID = undefined;
    for (&history) |*h| h.* = 1;

    var logits_copy = logits;
    const token = sampler.sample(&logits_copy, &history);
    try std.testing.expectEqual(@as(TokenID, 1), token);
}

// Regression: tokens older than `repetition_penalty_last_n` must NOT
// be penalized. Without windowing, an old prompt occurrence of a token
// would still count against it many decode steps later, which compounds
// over long contexts.
test "repetition penalty: last_n window excludes old history" {
    var opts = Options.default;
    opts.temperature = 0.0;
    opts.repetition_penalty = 100.0;
    opts.repetition_penalty_last_n = 4;
    var sampler = init(std.testing.io, opts);
    const logits = [_]f32{ 1.0, 5.0, 4.9, 0.5 };

    // Token 1 appears at history[0], window is last 4 → only history[1..5]
    // is considered. Token 1 is OUT of window, so no penalty. Argmax = 1.
    const history = [_]TokenID{ 1, 2, 3, 0, 2 };
    var logits_copy = logits;
    const token = sampler.sample(&logits_copy, &history);
    try std.testing.expectEqual(@as(TokenID, 1), token);
}

// Regression: negative-logit branch (multiply instead of divide). The
// original sampler uses asymmetric penalty — positive logits divide,
// negative logits multiply (push further from zero). Per-unique must
// apply that branch correctly too, not silently fall back to per-occurrence.
test "repetition penalty: negative logits scaled per-unique" {
    var opts = Options.default;
    opts.temperature = 0.0;
    opts.repetition_penalty = 2.0;
    opts.repetition_penalty_last_n = 64;
    var sampler = init(std.testing.io, opts);

    // Token 0 starts negative; per-unique penalty: -1.0 * 2.0 = -2.0
    // (pushes further negative). Per-occurrence (broken) over 10 repeats:
    // -1.0 * 2.0^10 = -1024 (would be the same direction but extreme).
    // Token 1 is positive 1.5 — argmax = 1 either way.
    // To distinguish per-unique from per-occurrence cleanly, check the
    // penalized logit is exactly -2.0 (not -1024).
    const history = [_]TokenID{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    var logits = [_]f32{ -1.0, 1.5, 0.0 };
    _ = sampler.sample(&logits, &history);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), logits[0], 0.001);
}

// Out-of-range token IDs in history must be silently skipped (vocab
// expansion across model versions can leave stale token IDs above the
// current logits.len in serialized history).
test "repetition penalty: ignores out-of-range token ids" {
    var opts = Options.default;
    opts.temperature = 0.0;
    opts.repetition_penalty = 100.0;
    var sampler = init(std.testing.io, opts);
    const logits = [_]f32{ 1.0, 5.0, 4.9, 0.5 };

    // history[0] is out of range; without skip we'd read OOB / panic.
    const history = [_]TokenID{ 9999, 1 };
    var logits_copy = logits;
    const token = sampler.sample(&logits_copy, &history);
    // Token 1 still penalized (in range), so token 2 wins.
    try std.testing.expectEqual(@as(TokenID, 2), token);
}

const std = @import("std");
