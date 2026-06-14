//! Chat-template interface. Like `Model` and `Context`, this is a view
//! type: implementations either fill one in directly (stateless
//! templates — the vtable fns ignore `self`) or embed a `Chat` field in
//! a stateful wrapper and recover the concrete type via
//! `@fieldParentPtr` (e.g. infer-llama's template-carrying overlay).
//!
//! Data fields (primes, suffix, special-token IDs) are interface DATA —
//! `ChatSession` reads them directly on the hot path. Rendering and
//! tool-call parsing go through the vtable. The live, turn-by-turn
//! driver that consumes all of this is `ChatSession`
//! (`chat_session.zig`); the message vocabulary it renders lives in
//! `message.zig`.
//!
//! Variants attach one to `Model.chat`. Not required for
//! completion-only models — those leave `model.chat = null`, and
//! `ChatSession.init` then fails fast with
//! `error.ModelDoesNotSupportChat`.

/// Text written to the cache right before the model generates —
/// the "assistant turn opener." For ChatML this is
/// `"<|im_start|>assistant\n"`. For LLaMA 2 it's `""` (LLaMA 2
/// fuses assistant output directly after `[/INST]`, no opener).
/// ChatSession prefills this right after rendering the preceding
/// user/tool_result so `turn_start_pos` can be captured between
/// the two — enabling the ephemeral-thinking rollback.
assistant_prime: []const u8 = "",

/// Alternative prime used when `ChatOptions.thinking == false`.
/// For Qwen3 and similar reasoning models, this embeds a
/// canonical empty `<think></think>` block that tells the model
/// "I've done my reasoning, go straight to content" — e.g.
/// `"<|im_start|>assistant\n<think>\n\n</think>\n\n"`. `null` →
/// `assistant_prime` is used regardless of thinking mode.
assistant_prime_no_thinking: ?[]const u8 = null,

/// Inter-turn separator spliced after the EOT token on the
/// no-thinking path. For ChatML set to `"\n"`; empty for models
/// with no separator (Llama 2, Granite). Not consulted on the
/// thinking-rollback path — `formatMessage` produces the template's
/// trailing separator itself.
end_of_turn_suffix: []const u8 = "",

/// Per-model semantic special-token IDs.
special_tokens: SpecialTokens = .{},

/// The vocabulary's designated end-of-sequence token, copied here
/// once by the variant's `load()`. Used as a fallback by
/// `classifyToken` when a model's chat-specific end-of-turn markers
/// happen to coincide with (or aren't distinct from) the
/// tokenizer's EOS.
eos_token_id: u32,

vtable: *const VTable,

const Chat = @This();

pub const VTable = struct {
    /// Render the system block (system_prompt + tool specs) for a
    /// conversation. Called once per session init. Returned slice is
    /// owned by the caller and freed via the allocator.
    formatSystem: *const fn (*Chat, std.mem.Allocator, ChatOptions) anyerror![]const u8,

    /// Render one completed message (user, assistant, tool_result) as a
    /// standalone chunk. Assistants are rendered **without** their
    /// `thinking` field — the field stays on the Message struct for
    /// display/logging but is never written to the KV cache.
    formatMessage: *const fn (*Chat, std.mem.Allocator, Message) anyerror![]const u8,

    /// Optional tool-call body parser. A variant that defines
    /// `tool_call_start` / `tool_call_end` boundary tokens must also
    /// install this so `ChatSession` knows how to parse the buffered
    /// body. Today Qwen3 wires `runtime.hermes.parseToolCall`;
    /// non-Hermes dialects supply their own. When null, tool-call
    /// events degrade to content-only handling.
    ///
    /// Parses the **whole buffered span** into its constituent calls and
    /// returns them as a caller-owned slice: most dialects pack one call
    /// per span and return a single element (Hermes), but some emit
    /// several in one span (Olmo's `<function_calls>` block). The slice
    /// and each element's `name`/`arguments` are freed via `allocator`.
    parseToolCall: ?*const fn (*Chat, std.mem.Allocator, []const u8) anyerror![]ParsedToolCall = null,
};

pub const ParsedToolCall = struct {
    name: []const u8,
    arguments: []const u8,
};

pub const ChatOptions = struct {
    system_prompt: ?[]const u8 = null,
    thinking: bool = false,
    tools: []const ToolSpec = &.{},
};

pub const SpecialTokens = struct {
    end_of_turn: ?u32 = null,
    end_of_turn_alt: ?u32 = null,
    thinking_start: ?u32 = null,
    thinking_end: ?u32 = null,
    tool_call_start: ?u32 = null,
    tool_call_end: ?u32 = null,
};

/// Semantic classes used by `ChatSession` to drive its streaming
/// state machine.
pub const TokenClass = enum {
    content,
    thinking_start,
    thinking_end,
    tool_call_start,
    tool_call_end,
    end_of_turn,
};

/// Render the system block (system_prompt + tool specs). Returns a
/// freshly allocated, caller-owned slice.
pub fn formatSystem(self: *Chat, allocator: std.mem.Allocator, options: ChatOptions) ![]const u8 {
    return try self.vtable.formatSystem(self, allocator, options);
}

/// Render one completed message as a standalone chunk. Returns a
/// freshly allocated, caller-owned slice.
pub fn formatMessage(self: *Chat, allocator: std.mem.Allocator, msg: Message) ![]const u8 {
    return try self.vtable.formatMessage(self, allocator, msg);
}

/// Parse one buffered tool-call span into its constituent calls. Returns
/// `error.ToolsNotSupported` when the implementation didn't install a
/// parser. The returned slice and each element's `name`/`arguments` are
/// caller-owned and freed via `allocator`.
pub fn parseToolCall(self: *Chat, allocator: std.mem.Allocator, body: []const u8) ![]ParsedToolCall {
    const hook = self.vtable.parseToolCall orelse return error.ToolsNotSupported;
    return try hook(self, allocator, body);
}

/// Classify a token using the chat overlay's semantic special
/// tokens, falling back to `self.eos_token_id` (the vocabulary's
/// EOS, copied in at construction time so `Chat` stays
/// self-contained with no back-pointer to `Model`).
pub fn classifyToken(self: *const Chat, token: u32) TokenClass {
    const st = self.special_tokens;
    if (st.thinking_start) |id| if (token == id) return .thinking_start;
    if (st.thinking_end) |id| if (token == id) return .thinking_end;
    if (st.tool_call_start) |id| if (token == id) return .tool_call_start;
    if (st.tool_call_end) |id| if (token == id) return .tool_call_end;
    if (st.end_of_turn) |id| if (token == id) return .end_of_turn;
    if (st.end_of_turn_alt) |id| if (token == id) return .end_of_turn;
    if (token == self.eos_token_id) return .end_of_turn;
    return .content;
}

/// True iff `token` is a chat-layer end-of-turn marker. Does
/// **not** check `eos_token_id` — that's the `Model` layer's
/// responsibility. Called by `Model.isEndOfTurn` via delegation.
pub fn isEndOfTurn(self: *const Chat, token: u32) bool {
    if (self.special_tokens.end_of_turn) |id| if (token == id) return true;
    if (self.special_tokens.end_of_turn_alt) |id| if (token == id) return true;
    return false;
}

/// Select the correct prime string for the given thinking mode.
/// Returns `assistant_prime_no_thinking` when thinking is off and
/// one is set; otherwise falls back to `assistant_prime`.
pub fn primeFor(self: *const Chat, thinking: bool) []const u8 {
    if (!thinking) {
        if (self.assistant_prime_no_thinking) |p| return p;
    }
    return self.assistant_prime;
}

/// Render a full conversation as a single prompt string. Useful
/// for one-shot generation outside of a ChatSession. All assistant
/// messages are rendered without `thinking` (consistent with the
/// ephemeral-thinking invariant).
pub fn renderConversation(
    self: *Chat,
    allocator: std.mem.Allocator,
    messages: []const Message,
    options: ChatOptions,
) ![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    const system = try self.formatSystem(allocator, options);
    defer allocator.free(system);
    try buf.appendSlice(allocator, system);

    for (messages) |msg| {
        const to_render = switch (msg) {
            .assistant => |a| Message{ .assistant = .{
                .content = a.content,
                .thinking = null,
                .tool_calls = a.tool_calls,
            } },
            else => msg,
        };
        const chunk = try self.formatMessage(allocator, to_render);
        defer allocator.free(chunk);
        try buf.appendSlice(allocator, chunk);
    }

    // Prime the assistant if the last message isn't one already.
    if (messages.len > 0 and messages[messages.len - 1] != .assistant) {
        try buf.appendSlice(allocator, self.primeFor(options.thinking));
    }

    return try buf.toOwnedSlice(allocator);
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "Chat.classifyToken uses chat specials with eos fallback" {
    const chat: Chat = .{
        .vtable = undefined,
        .eos_token_id = 99,
        .special_tokens = .{
            .end_of_turn = 1,
            .end_of_turn_alt = 2,
            .thinking_start = 10,
            .thinking_end = 11,
            .tool_call_start = 20,
            .tool_call_end = 21,
        },
    };

    try testing.expectEqual(TokenClass.end_of_turn, chat.classifyToken(1));
    try testing.expectEqual(TokenClass.end_of_turn, chat.classifyToken(2));
    try testing.expectEqual(TokenClass.thinking_start, chat.classifyToken(10));
    try testing.expectEqual(TokenClass.thinking_end, chat.classifyToken(11));
    try testing.expectEqual(TokenClass.tool_call_start, chat.classifyToken(20));
    try testing.expectEqual(TokenClass.tool_call_end, chat.classifyToken(21));
    try testing.expectEqual(TokenClass.content, chat.classifyToken(50));
    try testing.expectEqual(TokenClass.end_of_turn, chat.classifyToken(99));
}

test "Chat.classifyToken with empty specials falls back to eos only" {
    const chat: Chat = .{
        .vtable = undefined,
        .eos_token_id = 2,
    };
    try testing.expectEqual(TokenClass.end_of_turn, chat.classifyToken(2));
    try testing.expectEqual(TokenClass.content, chat.classifyToken(0));
}

test "Chat.isEndOfTurn checks only chat specials" {
    const chat: Chat = .{
        .vtable = undefined,
        .eos_token_id = 99,
        .special_tokens = .{ .end_of_turn = 1, .end_of_turn_alt = 2 },
    };
    try testing.expect(chat.isEndOfTurn(1));
    try testing.expect(chat.isEndOfTurn(2));
    try testing.expect(!chat.isEndOfTurn(99));
    try testing.expect(!chat.isEndOfTurn(50));
}

const std = @import("std");
const Message = @import("message.zig").Message;
const ToolSpec = @import("message.zig").ToolSpec;
