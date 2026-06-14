//! Token-sampling interface. Like `Model` and `Context`, this is a
//! view type: implementations embed a `Sampler` field and install a
//! `VTable` whose method recovers the concrete type via
//! `@fieldParentPtr`. Most callers use the provided `Sampler.Default`
//! (temperature / top-k / top-p / min-p / repetition penalty);
//! variants can bring their own by passing a `*Sampler` through
//! `Context.Options.sampler`.
//!
//! `options` is interface DATA, not implementation detail: presets
//! (`Model.sampler_presets` → `ChatSession.init`) and per-turn
//! overrides (`Context.nextWith`) read/write the standard `Options`
//! regardless of which implementation is installed. Custom samplers
//! interpret the knobs as they see fit (or ignore them).

options: Options,
vtable: *const VTable,

const Sampler = @This();

pub const TokenID = u32;

/// The default implementation. Embeds `Sampler` as `.interface`.
pub const Default = @import("default_sampler.zig");

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
    /// the multilingual / missing-space drift at rep=1.1.
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
};

pub const VTable = struct {
    /// Sample a token from the logits distribution. May modify `logits`
    /// in place (the Default does). `options` is whatever the caller
    /// resolved — `self.options` for `sample`, an explicit override for
    /// `sampleWith` — so implementations never read `self.options`
    /// directly.
    sample: *const fn (*Sampler, logits: []f32, history: []const TokenID, options: Options) TokenID,
};

/// Sample a token from the logits distribution using `self.options`.
/// May modify `logits` in place.
pub fn sample(
    self: *Sampler,
    logits: []f32,
    history: []const TokenID,
) TokenID {
    return self.vtable.sample(self, logits, history, self.options);
}

/// Sample a token using an explicit `Options` instead of `self.options`.
/// One-shot — `self.options` is not mutated. Implementation state (e.g.
/// the Default's RNG) is shared with `sample`.
pub fn sampleWith(
    self: *Sampler,
    logits: []f32,
    history: []const TokenID,
    options: Options,
) TokenID {
    return self.vtable.sample(self, logits, history, options);
}
