//! One live conversation: a tokenizer (for encode/decode), a per-session
//! sampler (with its own RNG), history + logits buffers, plus a small
//! vtable of hot-path methods (`restart`, `prefill`, `next`).
//!
//! `Context` is a **view** type, embedded inside each variant's
//! `ConcreteContext` wrapper by `Engine.createContext`. The caller owns
//! that wrapper — recoverable via `@fieldParentPtr` when the variant type
//! is known at comptime — and is responsible for its cleanup. `deinit`
//! frees only the generic state this struct owns; see its doc.

allocator: std.mem.Allocator,
tokenizer: Tokenizer,
sampler: Sampler,
max_len: usize,
current_token: u32,
history: std.ArrayList(u32),
logits_buf: []f32,
/// `logits_buf` already holds logits for the next sample — set by
/// `prefill`, cleared by the first `next` that consumes them.
has_pending_logits: bool,
vtable: *const VTable,
/// Producing `Engine`. Set by every `createContext` implementation;
/// carries the family's tokenizer hooks and sampler presets.
engine: *Engine,

const Context = @This();

pub const Options = struct {
    sampler_options: Sampler.Options = .default,
    max_len: ?usize = null,
};

pub const VTable = struct {
    /// Wipe the per-variant KV state back to empty without destroying
    /// the wrapper. `Context.restart` calls this, then resets the
    /// generic history / current_token / has_pending_logits.
    restart: *const fn (*Context) anyerror!void,

    /// Drop all KV entries at positions ≥ `position`. Used by
    /// ChatSession to implement **ephemeral thinking**: reasoning
    /// tokens are committed to the cache during generation so the
    /// model can read its own scratchpad, then rolled back at turn
    /// boundary so they don't persist across turns.
    ///
    /// Most variants can implement this as a single counter reset
    /// ("logical length goes back to N; KV cells ≥ N are stale and
    /// overwritten on the next prefill"). After this call, the next
    /// `prefill` must write K/V starting at position `position`.
    truncateTo: *const fn (*Context, usize) anyerror!void,

    /// Variant's prefill: write K/V for tokens[0..N-1] (skip-last
    /// convention). `Context.prefill` then calls `next` on the last
    /// token to commit its slot and get the first sample's logits.
    prefill: *const fn (*Context, []const u32) anyerror!void,

    /// Variant's next: commit `token`'s KV slot, run the forward pass,
    /// return logits. The returned slice must stay valid until the next
    /// `next` / call; `Context.next` and friends `memcpy` it into
    /// their own `logits_buf` before doing anything else.
    next: *const fn (*Context, u32) anyerror![]const f32,

    /// Optional: free the `ConcreteContext` wrapper (and its embedded
    /// `Context`). Polymorphic callers that own the Context's lifetime
    /// call this on shutdown. Variants whose wrapper is freed out-of-band
    /// (e.g. via `@fieldParentPtr` by a comptime-aware caller) can leave
    /// it null. When null, callers must skip the call — whoever owns
    /// the wrapper is responsible for teardown.
    destroy: ?*const fn (*Context) void = null,
};

/// Free the generic state owned by this Context (logits buffer,
/// history). Does **not** touch any variant-specific state and does
/// **not** destroy the enclosing `ConcreteContext`. Variants call this
/// from their own `deinit` before freeing their KV state and destroying
/// the wrapper.
pub fn deinit(self: *Context) void {
    self.allocator.free(self.logits_buf);
    self.history.deinit(self.allocator);
}

/// Discard the current KV state and start fresh. History and logits
/// buffer are reset; sampler RNG state is preserved.
pub fn restart(self: *Context) !void {
    try self.vtable.restart(self);
    self.history.clearRetainingCapacity();
    self.current_token = 0;
    self.has_pending_logits = false;
}

/// Drop KV entries (and history) at positions ≥ `position`. Used by
/// `ChatSession` to strip ephemeral thinking at turn boundaries —
/// reasoning tokens are committed during generation so the model can
/// reason, then truncated away before the next turn begins.
///
/// After this call: `history.len == position`, `current_token` is the
/// token at `position - 1` (or 0 if `position == 0`), and the caller
/// can `prefill` new text which will write K/V starting at `position`.
pub fn truncateTo(self: *Context, position: usize) !void {
    std.debug.assert(position <= self.history.items.len);
    try self.vtable.truncateTo(self, position);
    self.history.shrinkRetainingCapacity(position);
    self.current_token = if (position == 0) 0 else self.history.items[position - 1];
    self.has_pending_logits = false;
}

/// Encode text to tokens. Routes through the engine's optional `encode`
/// hook when set (variants that delegate tokenization to an external
/// library), and falls back to the embedded `Tokenizer.encode` otherwise.
pub fn encode(self: *Context, allocator: std.mem.Allocator, text: []const u8) ![]const u32 {
    return try self.engine.encode(allocator, text, self.tokenizer);
}

/// Decode tokens to text. Mirrors `encode` — engine hook first, then the
/// embedded `Tokenizer.decode` fallback.
pub fn decode(self: *Context, allocator: std.mem.Allocator, tokens: []const u32) ![]const u8 {
    return try self.engine.decode(allocator, tokens, self.tokenizer);
}

/// Extend the context with `prompt`. After return, every token in
/// `prompt` is in the K/V cache, `current_token` is the last prompt
/// token, and `logits_buf` holds the logits for sampling the next token.
/// Safe to call repeatedly to extend across turns.
pub fn prefill(self: *Context, prompt: []const u8) !void {
    const tokens = try self.encode(self.allocator, prompt);
    defer self.allocator.free(tokens);
    if (tokens.len == 0) return;

    const prev_len = self.history.items.len;
    try self.history.appendSlice(self.allocator, tokens);
    errdefer self.history.shrinkRetainingCapacity(prev_len);

    try self.vtable.prefill(self, tokens);
    const logits = try self.vtable.next(self, tokens[tokens.len - 1]);
    // Variants are free to either compute into a private buffer (then we
    // copy out into our `logits_buf`) or write directly into our buffer
    // and hand the same slice back. Skip the no-op memcpy in the latter
    // case — `@memcpy` panics on aliasing in safety-checked builds.
    if (logits.ptr != self.logits_buf.ptr) @memcpy(self.logits_buf, logits);

    self.current_token = tokens[tokens.len - 1];
    self.has_pending_logits = true;
}

/// Sample the next token, append it to history, and return its decoded
/// text. Does **not** perform any end-of-turn detection — the caller
/// (typically `ChatSession`) is responsible for classifying
/// `current_token` after `next` returns and taking whatever action is
/// appropriate (absorbing EOT, injecting a per-turn suffix, finalizing
/// the turn).
///
/// Returns `error.ContextFull` if the context has reached `max_len`.
pub fn next(self: *Context) ![]const u8 {
    return self.nextInner(null);
}

/// Like `next` but samples with `sampler_override` instead of the
/// session's stored `sampler.options`. One-shot — `sampler.options` is
/// not mutated.
pub fn nextWith(self: *Context, sampler_override: Sampler.Options) ![]const u8 {
    return self.nextInner(sampler_override);
}

fn nextInner(self: *Context, sampler_override: ?Sampler.Options) ![]const u8 {
    if (self.history.items.len >= self.max_len) {
        return error.ContextFull;
    }

    if (self.has_pending_logits) {
        self.has_pending_logits = false;
    } else {
        const logits = try self.vtable.next(self, self.current_token);
        if (logits.ptr != self.logits_buf.ptr) @memcpy(self.logits_buf, logits);
    }

    const token = if (sampler_override) |opts|
        self.sampler.sampleWith(self.logits_buf, self.history.items, opts)
    else
        self.sampler.sample(self.logits_buf, self.history.items);

    self.current_token = token;
    try self.history.append(self.allocator, token);

    return try self.decode(self.allocator, &.{token});
}

/// Commit the KV slot for `current_token` without appending anywhere or
/// sampling. Used by callers (e.g. ChatSession) to finalize the
/// just-sampled token's cache state before absorbing follow-up tokens or
/// letting a turn end cleanly.
pub fn commitCurrent(self: *Context) !void {
    _ = try self.vtable.next(self, self.current_token);
    self.has_pending_logits = false;
}

/// Append `token` to history, commit its KV slot, and set
/// `current_token`. Used to splice follow-up tokens (e.g. an
/// end-of-turn suffix) into the cache without running them through
/// the sampler.
pub fn absorb(self: *Context, token: u32) !void {
    try self.history.append(self.allocator, token);
    self.current_token = token;
    _ = try self.vtable.next(self, token);
    self.has_pending_logits = false;
}


const std = @import("std");
const Tokenizer = @import("tokenizer.zig");
const Sampler = @import("sampler.zig");
const Engine = @import("engine.zig");
