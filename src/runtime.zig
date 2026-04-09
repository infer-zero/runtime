//! Inference driver: owns a per-conversation `Context`, a `Tokenizer` for
//! text↔token conversion, and a `Sampler` for logits→token selection. Takes
//! a borrowed `*Inference` plus a borrowed `*const Info`; knows nothing about
//! chat templates, special tokens, or tool calls. Those all belong to
//! `Model` / `ChatSession` sitting above.
//!
//! Callers that want raw completion construct a `Runtime` directly from a
//! `Model` (via `Runtime.init`) and use `Context.prefill` + `Context.next` in
//! a loop — stopping when they choose, typically by calling
//! `model.classifyToken` on `context.current_token` externally.

allocator: std.mem.Allocator,
inference: *Inference,
info: *const Info,
tokenizer: Tokenizer,
sampler: Sampler,
max_len: usize,

const Runtime = @This();

pub const Options = struct {
    sampler_options: Sampler.Options = .default,
    max_len: ?usize = null,
};

/// Initialize a Runtime bound to a `Model`. Captures pointers into the model's
/// `inference` and `info` — the caller must keep the `Model` alive for the
/// lifetime of the `Runtime`.
pub fn init(allocator: std.mem.Allocator, model: *Model, options: Options) @This() {
    const vocab = model.vocabulary();
    return .{
        .allocator = allocator,
        .inference = &model.inference,
        .info = &model.info,
        .tokenizer = Tokenizer.init(vocab),
        .sampler = Sampler.init(options.sampler_options),
        .max_len = options.max_len orelse model.info.max_len,
    };
}

pub fn deinit(self: *@This()) void {
    self.inference.deinit();
}

pub fn start(self: *@This()) !Context {
    return Context.init(self);
}

pub const Context = struct {
    runtime: *Runtime,
    context: *anyopaque,
    current_token: u32,
    history: std.ArrayListUnmanaged(u32),
    logits_buf: []f32,
    /// `logits_buf` already holds logits for the next sample — set by
    /// `prefill`, cleared by the first `next` that consumes them.
    has_pending_logits: bool,

    pub fn init(runtime: *Runtime) !@This() {
        const context = try runtime.inference.createContext();
        errdefer runtime.inference.destroyContext(context);

        const logits_buf = try runtime.allocator.alloc(f32, runtime.info.vocabulary_size);
        errdefer runtime.allocator.free(logits_buf);

        return .{
            .runtime = runtime,
            .context = context,
            .current_token = 0,
            .history = .empty,
            .logits_buf = logits_buf,
            .has_pending_logits = false,
        };
    }

    /// Extend the context with `prompt`. After return, every token in
    /// `prompt` is in the K/V cache, `ctx.position == history.len`, and
    /// `logits_buf` holds the logits for sampling the next token. Safe to
    /// call repeatedly to extend across turns.
    pub fn prefill(self: *@This(), prompt: []const u8) !void {
        const tokens = try self.runtime.tokenizer.encode(self.runtime.allocator, prompt);
        defer self.runtime.allocator.free(tokens);
        if (tokens.len == 0) return;

        const prev_len = self.history.items.len;
        try self.history.appendSlice(self.runtime.allocator, tokens);
        errdefer self.history.shrinkRetainingCapacity(prev_len);

        // inference.prefill writes K/V for tokens[0..N-1] (its skip-last
        // convention). inference.next then commits the last token's K/V slot
        // AND produces the logits we need for the first sample.
        try self.runtime.inference.prefill(self.context, tokens);
        const logits = try self.runtime.inference.next(self.context, tokens[tokens.len - 1]);
        @memcpy(self.logits_buf, logits);

        self.current_token = tokens[tokens.len - 1];
        self.has_pending_logits = true;
    }

    /// Sample the next token, append it to history, and return its decoded
    /// text. Does **not** perform any end-of-turn detection — the caller
    /// (typically `ChatSession`) is responsible for calling
    /// `model.classifyToken(ctx.current_token)` after `next` returns and
    /// taking whatever action is appropriate (absorbing EOT, injecting a
    /// per-turn suffix, finalizing the turn).
    ///
    /// Returns `error.ContextFull` if the context has reached `runtime.max_len`.
    pub fn next(self: *@This()) ![]const u8 {
        if (self.history.items.len >= self.runtime.max_len) {
            return error.ContextFull;
        }

        if (self.has_pending_logits) {
            self.has_pending_logits = false;
        } else {
            const logits = try self.runtime.inference.next(self.context, self.current_token);
            @memcpy(self.logits_buf, logits);
        }

        const token = self.runtime.sampler.sample(self.logits_buf, self.history.items);

        self.current_token = token;
        try self.history.append(self.runtime.allocator, token);

        return try self.runtime.tokenizer.decode(self.runtime.allocator, &.{token});
    }

    /// Commit the KV slot for `self.current_token` without appending
    /// anywhere or sampling. Used by callers (e.g. ChatSession) to finalize
    /// the just-sampled token's cache state before absorbing follow-up
    /// tokens or letting a turn end cleanly.
    pub fn commitCurrent(self: *@This()) !void {
        _ = try self.runtime.inference.next(self.context, self.current_token);
        self.has_pending_logits = false;
    }

    /// Append `token` to history, commit its KV slot, and set
    /// `current_token`. Used to splice follow-up tokens (e.g. an
    /// end-of-turn suffix) into the cache without running them through
    /// the sampler.
    pub fn absorb(self: *@This(), token: u32) !void {
        try self.history.append(self.runtime.allocator, token);
        self.current_token = token;
        _ = try self.runtime.inference.next(self.context, token);
        self.has_pending_logits = false;
    }

    pub fn deinit(self: *@This()) void {
        self.runtime.inference.destroyContext(self.context);
        self.history.deinit(self.runtime.allocator);
        self.runtime.allocator.free(self.logits_buf);
    }
};

const std = @import("std");
const Tokenizer = @import("tokenizer.zig");
const Sampler = @import("sampler.zig");
const Model = @import("model.zig");
const Inference = @import("inference.zig");
const Info = @import("info.zig");
