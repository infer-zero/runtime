ptr: *anyopaque,
vtable: *const VTable,
eos_token_id: u32,
vocabulary_size: usize,
max_len: usize,
special_tokens: SpecialTokens = .{},
/// Text written to the KV cache immediately after the model emits its end-of-turn
/// token, to keep the cache aligned with the canonical chat template (which
/// typically has e.g. `<|im_end|>\n` between turns — the model only emits the
/// `<|im_end|>` part). For ChatML models (Qwen, Phi4, LFM2 etc.) set this to "\n".
/// Empty for models with no inter-turn separator (Llama 2, Granite).
end_of_turn_suffix: []const u8 = "",

const Model = @This();

pub const TokenClass = enum {
    content,
    thinking_start,
    thinking_end,
    tool_call_start,
    tool_call_end,
    end_of_turn,
};

/// Per-model semantic special-token IDs. Populated by each model at init time
/// from tokenizer lookups, then mounted on the type-erased Model via wrap().
/// All fields are `?u32`; null means "this model doesn't use that token class."
///
/// `end_of_turn` and `end_of_turn_alt` cover the realistic upper bound — no model
/// in the wild has more than two end-of-turn tokens (Qwen uses both `<|im_end|>`
/// and `<|end_of_text|>`; Llama 3 uses `<|eot_id|>` and `<|end_of_text|>`; most
/// other models have just one). If `end_of_turn` is null and the model has no
/// custom classifyToken vtable entry, Model.classifyToken falls back to
/// matching `eos_token_id`.
pub const SpecialTokens = struct {
    end_of_turn: ?u32 = null,
    end_of_turn_alt: ?u32 = null,
    thinking_start: ?u32 = null,
    thinking_end: ?u32 = null,
    tool_call_start: ?u32 = null,
    tool_call_end: ?u32 = null,

    /// True if any field has been populated. Used by Model.classifyToken to
    /// decide whether to use the data path or fall back to the vtable.
    pub fn isPopulated(self: SpecialTokens) bool {
        return self.end_of_turn != null or
            self.end_of_turn_alt != null or
            self.thinking_start != null or
            self.thinking_end != null or
            self.tool_call_start != null or
            self.tool_call_end != null;
    }
};

pub const ChatOptions = struct {
    system_prompt: ?[]const u8 = null,
    thinking: bool = false,
    tools: []const ToolSpec = &.{},
};

/// Per-message rendering hints passed to formatMessage.
pub const MessageFormat = struct {
    /// If true, append the model's "open assistant turn" marker after the
    /// message, priming the model to generate immediately on the next decode call.
    prime_assistant: bool,
    /// Whether thinking mode is enabled. When false, ChatML models that use
    /// the canonical empty-think-block convention (Qwen3) inject `<think></think>`
    /// after the assistant prime so the model skips reasoning and produces a
    /// direct answer. ChatSession passes this from session.options.thinking.
    thinking: bool = false,
    /// True when this message is being re-rendered as part of prior conversation
    /// history (not the most recent turn). For Qwen3-style models that strip
    /// `<think>` blocks from older assistant turns per the canonical chat
    /// template, formatMessage uses this flag to omit the reasoning content.
    /// Models without a prior-vs-current distinction can ignore this field.
    /// Set by ChatSession.rebuildContext (full_replay mode) and by
    /// ChatSession.replay for replayed assistant messages.
    is_prior_turn: bool = false,
};

pub const VTable = struct {
    deinit: *const fn (*Model) void,
    vocabulary: *const fn (*Model) Vocabulary,
    createContext: *const fn (*Model) anyerror!*anyopaque,
    destroyContext: *const fn (*Model, *anyopaque) void,
    prefill: *const fn (*Model, *anyopaque, []const u32) anyerror!void,
    next: *const fn (*Model, *anyopaque, u32) anyerror![]const f32,
    chatFormat: *const fn (*Model, Allocator, []const Message, ChatOptions) anyerror![]const u8,
    classifyToken: *const fn (*Model, u32) TokenClass,
    /// New API. Models that haven't been ported yet get a default stub from wrap()
    /// that returns error.NewApiNotImplemented; ChatSession requires this to work.
    formatSystem: *const fn (*Model, Allocator, ChatOptions) anyerror![]const u8,
    formatMessage: *const fn (*Model, Allocator, Message, MessageFormat) anyerror![]const u8,
};

/// Return a typed wrapper for the given concrete model type.
pub fn wrap(comptime T: type) type {
    return struct {
        ptr: *T,

        pub fn init(ptr: *T) @This() {
            return .{ .ptr = ptr };
        }

        /// Produce the type-erased Model interface.
        pub fn model(self: @This()) Model {
            return .{
                .ptr = self.ptr,
                .vtable = &vtable_instance,
                .eos_token_id = self.ptr.eos_token_id,
                .vocabulary_size = self.ptr.config.vocabulary_size,
                .max_len = self.ptr.config.max_len,
                .special_tokens = if (@hasField(T, "special_tokens")) self.ptr.special_tokens else .{},
                .end_of_turn_suffix = if (@hasDecl(T, "end_of_turn_suffix")) T.end_of_turn_suffix else "",
            };
        }

        const vtable_instance: VTable = .{
            .deinit = deinitFn,
            .vocabulary = vocabularyFn,
            .createContext = createContextFn,
            .destroyContext = destroyContextFn,
            .prefill = prefillFn,
            .next = nextFn,
            .chatFormat = chatFormatFn,
            .classifyToken = classifyTokenFn,
            .formatSystem = if (@hasDecl(T, "formatSystem")) formatSystemFn else defaultFormatSystemFn,
            .formatMessage = if (@hasDecl(T, "formatMessage")) formatMessageFn else defaultFormatMessageFn,
        };

        fn deinitFn(m: *Model) void {
            const concrete: *T = @ptrCast(@alignCast(m.ptr));
            concrete.deinit();
        }

        fn vocabularyFn(m: *Model) Vocabulary {
            const concrete: *T = @ptrCast(@alignCast(m.ptr));
            return concrete.vocabulary();
        }

        fn createContextFn(m: *Model) anyerror!*anyopaque {
            const concrete: *T = @ptrCast(@alignCast(m.ptr));
            const ctx = try concrete.createContext();
            return @ptrCast(ctx);
        }

        fn destroyContextFn(m: *Model, ctx: *anyopaque) void {
            const concrete: *T = @ptrCast(@alignCast(m.ptr));
            const typed_ctx: *T.Context = @ptrCast(@alignCast(ctx));
            concrete.destroyContext(typed_ctx);
        }

        fn prefillFn(m: *Model, ctx: *anyopaque, tokens: []const u32) anyerror!void {
            const concrete: *T = @ptrCast(@alignCast(m.ptr));
            const typed_ctx: *T.Context = @ptrCast(@alignCast(ctx));
            try concrete.prefill(typed_ctx, tokens);
        }

        fn nextFn(m: *Model, ctx: *anyopaque, token: u32) anyerror![]const f32 {
            const concrete: *T = @ptrCast(@alignCast(m.ptr));
            const typed_ctx: *T.Context = @ptrCast(@alignCast(ctx));
            return try concrete.next(typed_ctx, token);
        }

        fn chatFormatFn(_: *Model, allocator: Allocator, messages: []const Message, options: ChatOptions) anyerror![]const u8 {
            return try T.chatFormat(allocator, messages, options);
        }

        fn classifyTokenFn(m: *Model, token: u32) TokenClass {
            if (@hasDecl(T, "classifyToken")) {
                const concrete: *T = @ptrCast(@alignCast(m.ptr));
                return concrete.classifyToken(token);
            }
            return if (token == m.eos_token_id) .end_of_turn else .content;
        }

        fn formatSystemFn(_: *Model, allocator: Allocator, options: ChatOptions) anyerror![]const u8 {
            return try T.formatSystem(allocator, options);
        }

        fn formatMessageFn(_: *Model, allocator: Allocator, message: Message, fmt: MessageFormat) anyerror![]const u8 {
            return try T.formatMessage(allocator, message, fmt);
        }

        fn defaultFormatSystemFn(_: *Model, _: Allocator, _: ChatOptions) anyerror![]const u8 {
            return error.NewApiNotImplemented;
        }

        fn defaultFormatMessageFn(_: *Model, _: Allocator, _: Message, _: MessageFormat) anyerror![]const u8 {
            return error.NewApiNotImplemented;
        }
    };
}

/// Initialize a concrete model and wrap it in the Model interface.
pub fn init(comptime T: type, allocator: Allocator, path: []const u8) !@This() {
    const concrete = try T.init(allocator, path);
    return wrap(T).init(concrete).model();
}

pub fn deinit(self: *@This()) void {
    self.vtable.deinit(self);
}

pub fn vocabulary(self: *@This()) Vocabulary {
    return self.vtable.vocabulary(self);
}

pub fn createContext(self: *@This()) !*anyopaque {
    return try self.vtable.createContext(self);
}

pub fn destroyContext(self: *@This(), ctx: *anyopaque) void {
    self.vtable.destroyContext(self, ctx);
}

pub fn prefill(self: *@This(), ctx: *anyopaque, tokens: []const u32) !void {
    try self.vtable.prefill(self, ctx, tokens);
}

pub fn next(self: *@This(), ctx: *anyopaque, token: u32) ![]const f32 {
    return try self.vtable.next(self, ctx, token);
}

pub fn chatFormat(self: *@This(), allocator: Allocator, messages: []const Message, options: ChatOptions) ![]const u8 {
    return try self.vtable.chatFormat(self, allocator, messages, options);
}

/// Classify a token. Prefers the data-driven SpecialTokens path for ported
/// models; falls back to the vtable's classifyToken for unported models so
/// they continue working with their old custom logic. Once the fanout is
/// complete and every model populates special_tokens, the vtable entry can be
/// removed and this method becomes pure data lookup.
pub fn classifyToken(self: *@This(), token: u32) TokenClass {
    if (self.special_tokens.isPopulated()) {
        return self.classifyTokenFromData(token);
    }
    return self.vtable.classifyToken(self, token);
}

pub fn formatSystem(self: *@This(), allocator: Allocator, options: ChatOptions) ![]const u8 {
    return try self.vtable.formatSystem(self, allocator, options);
}

pub fn formatMessage(self: *@This(), allocator: Allocator, message: Message, fmt: MessageFormat) ![]const u8 {
    return try self.vtable.formatMessage(self, allocator, message, fmt);
}

/// Classify a token using the data-driven SpecialTokens table on this Model.
/// Used by classifyToken below as the preferred path; called directly when a
/// caller wants to bypass the vtable for ported models.
pub fn classifyTokenFromData(self: *const @This(), token: u32) TokenClass {
    const st = self.special_tokens;
    if (st.thinking_start) |id| if (token == id) return .thinking_start;
    if (st.thinking_end) |id| if (token == id) return .thinking_end;
    if (st.tool_call_start) |id| if (token == id) return .tool_call_start;
    if (st.tool_call_end) |id| if (token == id) return .tool_call_end;
    if (st.end_of_turn) |id| if (token == id) return .end_of_turn;
    if (st.end_of_turn_alt) |id| if (token == id) return .end_of_turn;
    return .content;
}

const std = @import("std");
const Allocator = std.mem.Allocator;
const Vocabulary = @import("vocabulary.zig");
const message_mod = @import("message.zig");
const Message = message_mod.Message;
const ToolSpec = message_mod.ToolSpec;

test "model wrap and runtime integration" {
    const testing = std.testing;
    const Runtime = @import("runtime.zig");

    const MockModel = struct {
        eos_token_id: u32 = 2,
        config: struct { vocabulary_size: usize = 4, max_len: usize = 16 } = .{},
        logits: [4]f32 = .{ 0.0, 0.0, 1.0, 0.0 }, // always predict token 2 (eos)

        pub const Context = struct { dummy: u8 = 0 };

        pub fn deinit(_: *@This()) void {}

        pub fn vocabulary(_: *@This()) Vocabulary {
            return .{
                .merge_index = .{},
                .encoding = .{},
                .decoding = .{},
            };
        }

        pub fn createContext(_: *@This()) !*Context {
            return try testing.allocator.create(Context);
        }

        pub fn destroyContext(_: *@This(), ctx: *Context) void {
            testing.allocator.destroy(ctx);
        }

        pub fn prefill(_: *@This(), _: *Context, _: []const u32) !void {}

        pub fn next(self: *@This(), _: *Context, _: u32) ![]const f32 {
            return &self.logits;
        }

        pub fn chatFormat(_: Allocator, _: []const Message, _: ChatOptions) ![]const u8 {
            return "mock";
        }
    };

    var mock = MockModel{};
    var m = wrap(MockModel).init(&mock).model();

    // Exercise the vtable interface
    var runtime = Runtime.init(testing.allocator, m, .{});
    var ctx = try runtime.start();
    defer ctx.deinit();

    // Prefill with dummy tokens
    try m.prefill(ctx.context, &.{ 0, 1 });

    // Next should return logits via vtable
    const logits = try m.next(ctx.context, 1);
    try testing.expectEqual(@as(usize, 4), logits.len);
    try testing.expectEqual(@as(f32, 1.0), logits[2]);

    // Default formatSystem/formatMessage stubs should return error.NewApiNotImplemented
    // for models that don't declare the new API.
    try testing.expectError(error.NewApiNotImplemented, m.formatSystem(testing.allocator, .{}));
    try testing.expectError(error.NewApiNotImplemented, m.formatMessage(
        testing.allocator,
        .{ .user = "hi" },
        .{ .prime_assistant = true },
    ));

    // With empty special_tokens, classifyTokenFromData returns content for everything;
    // the eos fallback lives in classifyToken, which delegates to the vtable for
    // unported models like this MockModel. The vtable default returns end_of_turn for eos.
    try testing.expectEqual(TokenClass.content, m.classifyTokenFromData(2));
    try testing.expectEqual(TokenClass.content, m.classifyTokenFromData(0));
    try testing.expectEqual(TokenClass.end_of_turn, m.classifyToken(2));
    try testing.expectEqual(TokenClass.content, m.classifyToken(0));
    try testing.expect(!m.special_tokens.isPopulated());

    // Cleanup
    m.deinit();
}

test "classifyTokenFromData with populated SpecialTokens" {
    const testing = std.testing;

    var m: Model = .{
        .ptr = undefined,
        .vtable = undefined,
        .eos_token_id = 99,
        .vocabulary_size = 0,
        .max_len = 0,
        .special_tokens = .{
            .end_of_turn = 1,
            .end_of_turn_alt = 2,
            .thinking_start = 10,
            .thinking_end = 11,
            .tool_call_start = 20,
            .tool_call_end = 21,
        },
    };

    try testing.expectEqual(TokenClass.end_of_turn, m.classifyTokenFromData(1));
    try testing.expectEqual(TokenClass.end_of_turn, m.classifyTokenFromData(2));
    try testing.expectEqual(TokenClass.thinking_start, m.classifyTokenFromData(10));
    try testing.expectEqual(TokenClass.thinking_end, m.classifyTokenFromData(11));
    try testing.expectEqual(TokenClass.tool_call_start, m.classifyTokenFromData(20));
    try testing.expectEqual(TokenClass.tool_call_end, m.classifyTokenFromData(21));
    try testing.expectEqual(TokenClass.content, m.classifyTokenFromData(50));
    // The data path doesn't fall back to eos_token_id; that's classifyToken's job.
    try testing.expectEqual(TokenClass.content, m.classifyTokenFromData(99));
    try testing.expect(m.special_tokens.isPopulated());
}
