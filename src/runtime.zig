allocator: std.mem.Allocator,
model: Model,
tokenizer: Tokenizer,
sampler: Sampler,
max_len: usize,

const Runtime = @This();

pub const Options = struct {
    sampler_options: Sampler.Options = .default,
    max_len: ?usize = null,
};

pub fn init(allocator: std.mem.Allocator, model: Model, options: Options) @This() {
    var m = model;
    const vocab = m.vocabulary();
    return .{
        .allocator = allocator,
        .model = model,
        .tokenizer = Tokenizer.init(vocab),
        .sampler = Sampler.init(options.sampler_options),
        .max_len = options.max_len orelse model.max_len,
    };
}

pub fn deinit(self: *@This()) void {
    self.model.deinit();
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
        const context = try runtime.model.createContext();
        errdefer runtime.model.destroyContext(context);

        const logits_buf = try runtime.allocator.alloc(f32, runtime.model.vocabulary_size);
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

        // model.prefill writes K/V for tokens[0..N-1] (its skip-last
        // convention). model.next then commits the last token's K/V slot AND
        // produces the logits we need for the first sample.
        try self.runtime.model.prefill(self.context, tokens);
        const logits = try self.runtime.model.next(self.context, tokens[tokens.len - 1]);
        @memcpy(self.logits_buf, logits);

        self.current_token = tokens[tokens.len - 1];
        self.has_pending_logits = true;
    }

    pub fn next(self: *@This()) !?[]const u8 {
        if (self.runtime.model.classifyToken(self.current_token) == .end_of_turn) {
            return null;
        }
        if (self.history.items.len >= self.runtime.max_len) {
            return error.ContextFull;
        }

        if (self.has_pending_logits) {
            self.has_pending_logits = false;
        } else {
            const logits = try self.runtime.model.next(self.context, self.current_token);
            @memcpy(self.logits_buf, logits);
        }

        const token = self.runtime.sampler.sample(self.logits_buf, self.history.items);

        self.current_token = token;
        try self.history.append(self.runtime.allocator, token);

        if (self.runtime.model.classifyToken(token) == .end_of_turn) {
            // Absorb EOT + per-turn suffix into the cache so the next
            // prefill starts at the canonical inter-turn offset. Driven via
            // model.next one token at a time (1-2 tokens total). current_token
            // stays as EOT so a stray follow-up next() short-circuits at the
            // guard above instead of extending a closed turn.
            _ = try self.runtime.model.next(self.context, token);

            const suffix = self.runtime.model.end_of_turn_suffix;
            if (suffix.len > 0) {
                const suffix_tokens = try self.runtime.tokenizer.encode(self.runtime.allocator, suffix);
                defer self.runtime.allocator.free(suffix_tokens);
                for (suffix_tokens) |suffix_token| {
                    try self.history.append(self.runtime.allocator, suffix_token);
                    _ = try self.runtime.model.next(self.context, suffix_token);
                }
            }
            self.has_pending_logits = false;
            return null;
        }

        const text = try self.runtime.tokenizer.decode(self.runtime.allocator, &.{token});
        return text;
    }

    pub fn deinit(self: *@This()) void {
        self.runtime.model.destroyContext(self.context);
        self.history.deinit(self.runtime.allocator);
        self.runtime.allocator.free(self.logits_buf);
    }
};

const std = @import("std");
const Tokenizer = @import("tokenizer.zig");
const Sampler = @import("sampler.zig");
const Model = @import("model.zig");
