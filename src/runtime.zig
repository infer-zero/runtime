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
        };
    }

    pub fn prefill(self: *@This(), prompt: []const u8) !void {
        const tokens = try self.runtime.tokenizer.encode(self.runtime.allocator, prompt);
        defer self.runtime.allocator.free(tokens);

        const prev_len = self.history.items.len;
        try self.history.appendSlice(self.runtime.allocator, tokens);
        errdefer self.history.shrinkRetainingCapacity(prev_len);

        try self.runtime.model.prefill(self.context, tokens);
        self.current_token = tokens[tokens.len - 1];
    }

    pub fn next(self: *@This()) !?[]const u8 {
        const eos_token_id = self.runtime.model.eos_token_id;

        if (self.current_token == eos_token_id) {
            return null;
        }
        if (self.history.items.len >= self.runtime.max_len) {
            return error.ContextFull;
        }

        const logits = try self.runtime.model.next(self.context, self.current_token);

        @memcpy(self.logits_buf, logits);

        const token = self.runtime.sampler.sample(self.logits_buf, self.history.items);

        self.current_token = token;
        try self.history.append(self.runtime.allocator, token);

        if (token == eos_token_id) {
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
