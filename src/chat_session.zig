//! Multi-turn chat session: borrows exactly one `Context` (the live KV
//! cache) plus a `Chat` overlay (rendering + semantic token classification).
//! Drives the streaming state machine for thinking blocks, tool calls, and
//! end-of-turn suffix injection.
//!
//! ChatSession **borrows** the Context — it does not deinit on teardown.
//! The caller owns the ConcreteContext (from the variant's createContext)
//! and is responsible for its lifecycle.
//!
//! **Ephemeral thinking.** Reasoning tokens are committed to the KV cache
//! during generation so the model can read its own scratchpad, but at
//! each turn boundary ChatSession rolls the cache back and re-prefills
//! the assistant turn *without* thinking. The consequence: prior-turn
//! reasoning is never in the model's context on subsequent turns —
//! matching the canonical Qwen3 / DeepSeek-R1 / Phi-4-reasoning chat
//! templates that strip prior `<think>` blocks — and there's only one
//! mode of operation (no `incremental` vs `full_replay` split).
//!
//! Turn layout in the KV cache, by phase:
//!   after sendText:  [system][user_0][prime][thinking?][content][eot]
//!   after rollback:  [system][user_0][prime with empty-think-block?]
//!   after re-prefill:[system][user_0][assistant_0_turn with content only]

allocator: std.mem.Allocator,
arena: std.heap.ArenaAllocator,
chat: Chat,
context: *Context,
messages: std.ArrayListUnmanaged(Message),
options: Chat.ChatOptions,
phase: Phase = .idle,
/// KV-cache position recorded at the start of the current assistant
/// generation (after user+prime, before the model's first token).
/// Used by `absorbEndOfTurn` to truncate + re-prefill content without
/// thinking. Undefined unless `phase == .streaming`.
turn_start_pos: usize = 0,
content_buf: std.ArrayListUnmanaged(u8) = .empty,
thinking_buf: std.ArrayListUnmanaged(u8) = .empty,
tool_call_buf: std.ArrayListUnmanaged(u8) = .empty,
tool_calls: std.ArrayListUnmanaged(Message.ToolCall) = .empty,

const ChatSession = @This();

/// Streaming state machine. Transitions:
///   idle ── sendText/sendToolResult/replay ──▶ streaming(.content)
///   streaming(.content) ── thinking_start token ──▶ streaming(.thinking)
///   streaming(.thinking) ── thinking_end token ──▶ streaming(.content)
///   streaming(*) ── tool_call_start token ──▶ streaming(.tool_call)
///   streaming(.tool_call) ── tool_call_end token ──▶ streaming(.content)
///   streaming(*) ── end_of_turn token / EOS / ContextFull ──▶ idle (returns .end_of_turn)
///   idle ── next() ──▶ idle (returning null, no-op)
const Phase = union(enum) {
    streaming: StreamKind,
    idle,
};

const StreamKind = enum { content, thinking, tool_call };

pub const Event = union(enum) {
    content: []const u8,
    thinking: []const u8,
    tool_call: []const u8,
    thinking_start,
    thinking_end,
    tool_call_start,
    tool_call_end,
    end_of_turn,
};

/// Initialize a chat session that **borrows** `context`. The session
/// prefix (system_prompt + tools, rendered by `chat.formatSystem`) is
/// prefilled into the KV cache once here.
pub fn init(
    allocator: std.mem.Allocator,
    chat: Chat,
    context: *Context,
    options: Chat.ChatOptions,
) !ChatSession {
    const prefix = try chat.formatSystem(allocator, options);
    defer allocator.free(prefix);

    if (prefix.len > 0) {
        try context.prefill(prefix);
    }

    return .{
        .allocator = allocator,
        .arena = std.heap.ArenaAllocator.init(allocator),
        .chat = chat,
        .context = context,
        .messages = .empty,
        .options = options,
    };
}

/// Free the session's own state (message history, streaming buffers).
/// Does **not** deinit `self.context` — that is the caller's
/// responsibility since ChatSession only borrows it.
pub fn deinit(self: *ChatSession) void {
    self.arena.deinit();
}

/// Send a user message and prepare the model to generate. Prefills the
/// rendered user turn + assistant prime on top of the existing cache.
/// Records `turn_start_pos` between the two so `absorbEndOfTurn` can
/// roll back to the pre-prime position if thinking was emitted.
pub fn sendText(self: *ChatSession, text: []const u8) !void {
    const arena_alloc = self.arena.allocator();
    const owned = try arena_alloc.dupe(u8, text);
    const msg = Message{ .user = owned };
    try self.messages.append(arena_alloc, msg);

    try self.prefillUserAndPrime(msg);
    self.resetStreamingState();
}

/// Supply a tool result back to the model and prepare it to continue
/// generating. Mirrors `sendText`.
pub fn sendToolResult(self: *ChatSession, tool_call_id: []const u8, content: []const u8) !void {
    const arena_alloc = self.arena.allocator();
    const msg = Message{ .tool_result = .{
        .tool_call_id = try arena_alloc.dupe(u8, tool_call_id),
        .content = try arena_alloc.dupe(u8, content),
    } };
    try self.messages.append(arena_alloc, msg);

    try self.prefillUserAndPrime(msg);
    self.resetStreamingState();
}

/// Render + prefill the given (user or tool_result) message, then the
/// assistant prime, recording `turn_start_pos` between the two.
fn prefillUserAndPrime(self: *ChatSession, msg: Message) !void {
    const user_chunk = try self.chat.formatMessage(self.allocator, msg);
    defer self.allocator.free(user_chunk);
    if (user_chunk.len > 0) try self.context.prefill(user_chunk);

    self.turn_start_pos = self.context.history.items.len;

    const prime = self.chat.primeFor(self.options.thinking);
    if (prime.len > 0) try self.context.prefill(prime);
}

/// Stream the next event in the current generation. Returns null when
/// the turn is fully drained (after .end_of_turn has been emitted once)
/// or when no generation is in progress.
pub fn next(self: *ChatSession) !?Event {
    const stream_kind = switch (self.phase) {
        .idle => return null,
        .streaming => |kind| kind,
    };

    const text = self.context.next() catch |err| {
        if (err == error.ContextFull) return try self.endTurn();
        return err;
    };

    const token_class = self.chat.classifyToken(self.context.current_token);

    switch (token_class) {
        .thinking_start => {
            self.phase = .{ .streaming = .thinking };
            self.allocator.free(text);
            return .thinking_start;
        },
        .thinking_end => {
            self.phase = .{ .streaming = .content };
            self.allocator.free(text);
            return .thinking_end;
        },
        .tool_call_start => {
            self.phase = .{ .streaming = .tool_call };
            self.allocator.free(text);
            return .tool_call_start;
        },
        .tool_call_end => {
            self.phase = .{ .streaming = .content };
            self.allocator.free(text);
            try self.commitToolCall();
            return .tool_call_end;
        },
        .end_of_turn => {
            self.allocator.free(text);
            try self.absorbEndOfTurn();
            return try self.endTurn();
        },
        .content => {
            const arena_alloc = self.arena.allocator();
            switch (stream_kind) {
                .thinking => {
                    try self.thinking_buf.appendSlice(arena_alloc, text);
                    return .{ .thinking = text };
                },
                .content => {
                    try self.content_buf.appendSlice(arena_alloc, text);
                    return .{ .content = text };
                },
                .tool_call => {
                    try self.tool_call_buf.appendSlice(arena_alloc, text);
                    return .{ .tool_call = text };
                },
            }
        },
    }
}

/// Bring the cache to a canonical post-turn state. Two paths:
///
///   1. **Thinking was emitted** → roll the cache back to
///      `turn_start_pos` (just after user, just before the prime), then
///      re-prefill the full assistant turn via `formatMessage` with the
///      completed message's content + tool_calls but `thinking = null`.
///      The template includes the trailing `<|im_end|>\n` in its
///      rendering, so no separate suffix absorption is needed.
///
///   2. **No thinking** → the model already emitted everything the
///      canonical template would; commit the EOT token's slot and
///      absorb the per-turn suffix (e.g. `"\n"` for ChatML). Cheaper
///      because we skip re-prefilling the content.
fn absorbEndOfTurn(self: *ChatSession) !void {
    try self.context.commitCurrent();

    if (self.thinking_buf.items.len > 0) {
        // Ephemeral-thinking rollback path.
        try self.context.truncateTo(self.turn_start_pos);

        const asst_msg = Message{ .assistant = .{
            .content = self.content_buf.items,
            .thinking = null, // stripped from cache; preserved in `self.messages` by finalizeTurn
            .tool_calls = self.tool_calls.items,
        } };
        const chunk = try self.chat.formatMessage(self.allocator, asst_msg);
        defer self.allocator.free(chunk);
        if (chunk.len > 0) try self.context.prefill(chunk);
    } else {
        // No thinking emitted — cache already matches canonical layout,
        // just splice the inter-turn suffix.
        const suffix = self.chat.end_of_turn_suffix;
        if (suffix.len > 0) {
            const suffix_tokens = try self.context.tokenizer.encode(self.allocator, suffix);
            defer self.allocator.free(suffix_tokens);
            for (suffix_tokens) |token| {
                try self.context.absorb(token);
            }
        }
    }
}

/// Common end-of-turn transition: finalize the turn into messages and
/// emit the `.end_of_turn` event. Subsequent `next()` calls return null.
fn endTurn(self: *ChatSession) !Event {
    try self.finalizeTurn();
    self.phase = .idle;
    return .end_of_turn;
}

/// Drain the current turn to completion and return the assembled
/// assistant message by value.
pub fn receive(self: *ChatSession) !Message.Assistant {
    while (try self.next()) |event| switch (event) {
        .end_of_turn => break,
        else => {},
    };

    if (self.messages.items.len == 0) return error.NoTurnGenerated;
    const last = self.messages.items[self.messages.items.len - 1];
    if (last != .assistant) return error.NoTurnGenerated;
    return last.assistant;
}

/// Prefill a saved conversation history into this session. Must be
/// called before any sendText/sendToolResult calls. Each non-final
/// message is rendered via `formatMessage` and prefilled incrementally
/// on top of the existing system prefix. If the last message is a user
/// or tool_result, an assistant prime is appended so the session is
/// ready to generate.
pub fn replay(self: *ChatSession, messages: []const Message) !void {
    if (self.messages.items.len != 0) return error.AlreadyStarted;
    const arena_alloc = self.arena.allocator();

    for (messages) |msg| {
        // Completed assistant messages are rendered without their
        // `thinking` field (it stays in `self.messages` for display
        // but isn't committed to the KV cache — matches the
        // ephemeral-thinking invariant).
        const to_render = switch (msg) {
            .assistant => |a| Message{ .assistant = .{
                .content = a.content,
                .thinking = null,
                .tool_calls = a.tool_calls,
            } },
            else => msg,
        };
        const chunk = try self.chat.formatMessage(self.allocator, to_render);
        defer self.allocator.free(chunk);
        if (chunk.len > 0) try self.context.prefill(chunk);

        try self.messages.append(arena_alloc, try dupeMessage(arena_alloc, msg));
    }

    // If the last message is user/tool_result, prime for generation.
    if (messages.len > 0) {
        const last = messages[messages.len - 1];
        if (last != .assistant) {
            self.turn_start_pos = self.context.history.items.len;
            const prime = self.chat.primeFor(self.options.thinking);
            if (prime.len > 0) try self.context.prefill(prime);
            self.resetStreamingState();
        }
    }
}

fn resetStreamingState(self: *ChatSession) void {
    self.phase = .{ .streaming = .content };
    self.content_buf = .empty;
    self.thinking_buf = .empty;
    self.tool_call_buf = .empty;
    self.tool_calls = .empty;
}

/// Parse the JSON in `tool_call_buf` (one `<tool_call>...</tool_call>`
/// body in Hermes/Qwen3 convention) and append a `Message.ToolCall` to
/// `self.tool_calls`. Synthesizes the id since Qwen3 tool calls don't
/// carry one. Malformed JSON is silently dropped — the agent loop will
/// then see no tool_calls and treat the turn as content-only. Parse
/// failures are logged via `std.log.scoped(.chat_session)` at debug
/// level; callers enable them by raising the log level.
fn commitToolCall(self: *ChatSession) !void {
    const arena_alloc = self.arena.allocator();
    defer self.tool_call_buf.clearRetainingCapacity();

    const raw = std.mem.trim(u8, self.tool_call_buf.items, " \t\r\n");
    if (raw.len == 0) return;

    const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, raw, .{}) catch {
        logToolCallDrop("JSON parse error", raw);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        logToolCallDrop("top-level value is not an object", raw);
        return;
    }
    const obj = parsed.value.object;

    const name_v = obj.get("name") orelse {
        logToolCallDrop("missing \"name\" field", raw);
        return;
    };
    if (name_v != .string) {
        logToolCallDrop("\"name\" is not a string", raw);
        return;
    }

    const args_str: []u8 = if (obj.get("arguments")) |args_v|
        try std.json.Stringify.valueAlloc(self.allocator, args_v, .{})
    else
        try self.allocator.dupe(u8, "{}");
    defer self.allocator.free(args_str);

    var id_buf: [32]u8 = undefined;
    const id_str = std.fmt.bufPrint(&id_buf, "call_{d}", .{self.tool_calls.items.len}) catch unreachable;

    try self.tool_calls.append(arena_alloc, .{
        .id = try arena_alloc.dupe(u8, id_str),
        .name = try arena_alloc.dupe(u8, name_v.string),
        .arguments = try arena_alloc.dupe(u8, args_str),
    });
}

fn logToolCallDrop(reason: []const u8, raw: []const u8) void {
    const preview = if (raw.len <= 200) raw else raw[0..200];
    const ellipsis: []const u8 = if (raw.len > 200) "..." else "";
    log.debug("tool_call dropped: {s}\n  raw: \"{s}{s}\"", .{ reason, preview, ellipsis });
}

fn finalizeTurn(self: *ChatSession) !void {
    const arena_alloc = self.arena.allocator();

    // `thinking` keeps the null-vs-present distinction (null = no thinking
    // block emitted); `content` and `tool_calls` just dupe — empty inputs
    // dupe to empty slices which is exactly what the defaults want.
    const thinking: ?[]const u8 = if (self.thinking_buf.items.len > 0)
        try arena_alloc.dupe(u8, self.thinking_buf.items)
    else
        null;

    try self.messages.append(arena_alloc, .{ .assistant = .{
        .content = try arena_alloc.dupe(u8, self.content_buf.items),
        .thinking = thinking,
        .tool_calls = try arena_alloc.dupe(Message.ToolCall, self.tool_calls.items),
    } });

    self.content_buf = .empty;
    self.thinking_buf = .empty;
    self.tool_call_buf = .empty;
    self.tool_calls = .empty;
}

/// Deep-copy a Message into arena storage so the caller can free or
/// drop their own copy after replay() returns.
fn dupeMessage(arena: std.mem.Allocator, msg: Message) !Message {
    return switch (msg) {
        .system => |t| .{ .system = try arena.dupe(u8, t) },
        .user => |t| .{ .user = try arena.dupe(u8, t) },
        .tool_result => |r| .{ .tool_result = .{
            .tool_call_id = try arena.dupe(u8, r.tool_call_id),
            .content = try arena.dupe(u8, r.content),
        } },
        .assistant => |a| blk: {
            const tool_calls_dup = try arena.alloc(Message.ToolCall, a.tool_calls.len);
            for (a.tool_calls, 0..) |call, idx| {
                tool_calls_dup[idx] = .{
                    .id = try arena.dupe(u8, call.id),
                    .name = try arena.dupe(u8, call.name),
                    .arguments = try arena.dupe(u8, call.arguments),
                };
            }
            const thinking_dup: ?[]const u8 = if (a.thinking) |t| try arena.dupe(u8, t) else null;
            break :blk .{ .assistant = .{
                .content = try arena.dupe(u8, a.content),
                .thinking = thinking_dup,
                .tool_calls = tool_calls_dup,
            } };
        },
    };
}

/// The chat-template overlay that `ChatSession` drives. Variants build
/// one of these in their `load()` and attach it to `Model.chat`. Not
/// required for completion-only models — those leave `model.chat =
/// null`, and `ChatSession.init` then fails fast with
/// `error.ModelDoesNotSupportChat`.
pub const Chat = struct {
    /// Render the system block (system_prompt + tool specs) for a
    /// conversation. Called once per session init.
    formatSystem: *const fn (std.mem.Allocator, ChatOptions) anyerror![]const u8,

    /// Render one completed message (user, assistant, tool_result) as a
    /// standalone chunk. Assistants are rendered **without** their
    /// `thinking` field — the field stays on the Message struct for
    /// display/logging but is never written to the KV cache.
    formatMessage: *const fn (std.mem.Allocator, Message) anyerror![]const u8,

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

    /// Optional tool-calling support. Nested inside Chat because tool
    /// calls have no free-standing existence outside a chat template.
    tool: ?Tool = null,

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

    /// Classify a token using the chat overlay's semantic special
    /// tokens, falling back to `self.eos_token_id` (the vocabulary's
    /// EOS, copied in at construction time so `Chat` stays
    /// self-contained with no back-pointer to `Model`).
    pub fn classifyToken(self: Chat, token: u32) TokenClass {
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
    pub fn isEndOfTurn(self: Chat, token: u32) bool {
        if (self.special_tokens.end_of_turn) |id| if (token == id) return true;
        if (self.special_tokens.end_of_turn_alt) |id| if (token == id) return true;
        return false;
    }

    /// Select the correct prime string for the given thinking mode.
    /// Returns `assistant_prime_no_thinking` when thinking is off and
    /// one is set; otherwise falls back to `assistant_prime`.
    pub fn primeFor(self: Chat, thinking: bool) []const u8 {
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
        self: Chat,
        allocator: std.mem.Allocator,
        messages: []const Message,
        options: ChatOptions,
    ) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
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

    /// Optional tool-calling support nested inside `Chat`.
    pub const Tool = struct {
        renderToolSpec: *const fn (std.mem.Allocator, []const ToolSpec) anyerror![]const u8,
        parseToolCall: *const fn (std.mem.Allocator, []const u8) anyerror!ParsedToolCall,

        pub const ParsedToolCall = struct {
            name: []const u8,
            arguments: []const u8,
        };
    };
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "Chat.classifyToken uses chat specials with eos fallback" {
    const chat: Chat = .{
        .formatSystem = undefined,
        .formatMessage = undefined,
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

    try testing.expectEqual(Chat.TokenClass.end_of_turn, chat.classifyToken(1));
    try testing.expectEqual(Chat.TokenClass.end_of_turn, chat.classifyToken(2));
    try testing.expectEqual(Chat.TokenClass.thinking_start, chat.classifyToken(10));
    try testing.expectEqual(Chat.TokenClass.thinking_end, chat.classifyToken(11));
    try testing.expectEqual(Chat.TokenClass.tool_call_start, chat.classifyToken(20));
    try testing.expectEqual(Chat.TokenClass.tool_call_end, chat.classifyToken(21));
    try testing.expectEqual(Chat.TokenClass.content, chat.classifyToken(50));
    try testing.expectEqual(Chat.TokenClass.end_of_turn, chat.classifyToken(99));
}

test "Chat.classifyToken with empty specials falls back to eos only" {
    const chat: Chat = .{
        .formatSystem = undefined,
        .formatMessage = undefined,
        .eos_token_id = 2,
    };
    try testing.expectEqual(Chat.TokenClass.end_of_turn, chat.classifyToken(2));
    try testing.expectEqual(Chat.TokenClass.content, chat.classifyToken(0));
}

test "Chat.isEndOfTurn checks only chat specials" {
    const chat: Chat = .{
        .formatSystem = undefined,
        .formatMessage = undefined,
        .eos_token_id = 99,
        .special_tokens = .{ .end_of_turn = 1, .end_of_turn_alt = 2 },
    };
    try testing.expect(chat.isEndOfTurn(1));
    try testing.expect(chat.isEndOfTurn(2));
    try testing.expect(!chat.isEndOfTurn(99));
    try testing.expect(!chat.isEndOfTurn(50));
}

// =============================================================================
// Chat message types — nested here at file level because they are used by both
// ChatSession and Chat (and nowhere else in base). They're the "chat vocabulary"
// for turns, tool specs, and tool-call parameter schemas.
// =============================================================================

/// One turn in a conversation. `system` is represented via
/// `ChatOptions.system_prompt`, not this union — the `.system` variant
/// exists only to let callers store a system turn in `messages` for
/// echo/round-trip purposes; `formatMessage` ignores it.
pub const Message = union(enum) {
    system: []const u8,
    user: []const u8,
    assistant: Assistant,
    tool_result: ToolResult,

    pub const Assistant = struct {
        content: []const u8 = "",
        /// Preserved for display/logging but never written to the KV
        /// cache — see the ephemeral-thinking invariant in the
        /// ChatSession docstring.
        thinking: ?[]const u8 = null,
        tool_calls: []const ToolCall = &.{},
    };

    pub const ToolCall = struct {
        id: []const u8,
        name: []const u8,
        arguments: []const u8,
    };

    pub const ToolResult = struct {
        tool_call_id: []const u8,
        content: []const u8,
    };
};

/// Description of a callable tool exposed to the model. Each chat
/// template serializes this into its own native tool format inside
/// `formatSystem`.
pub const ToolSpec = struct {
    name: []const u8,
    description: []const u8,
    parameters: Parameters,
};

/// Parameters for a tool. Use `simple` for the common flat case
/// (covers nearly all agentic test scenarios) or `json_schema` as an
/// escape hatch for complex schemas (nested objects, enums, unions).
pub const Parameters = union(enum) {
    simple: []const Parameter,
    json_schema: []const u8,
};

pub const Parameter = struct {
    name: []const u8,
    type: ParamType,
    description: []const u8 = "",
    required: bool = true,
};

pub const ParamType = enum {
    string,
    integer,
    number,
    boolean,
};

const std = @import("std");
const log = std.log.scoped(.chat_session);
const Context = @import("context.zig");
