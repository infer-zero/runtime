allocator: std.mem.Allocator,
arena: std.heap.ArenaAllocator,
runtime: *Runtime,
context: Runtime.Context,
messages: std.ArrayListUnmanaged(Message),
options: Model.ChatOptions,
mode: Mode = .incremental,
phase: Phase = .idle,
content_buf: std.ArrayListUnmanaged(u8) = .empty,
thinking_buf: std.ArrayListUnmanaged(u8) = .empty,
tool_call_buf: std.ArrayListUnmanaged(u8) = .empty,
tool_calls: std.ArrayListUnmanaged(Message.ToolCall) = .empty,

/// Strategy for assembling the KV cache across turns.
///
/// `incremental` (default): each sendText/sendToolResult appends exactly the
/// new turn's tokens to the existing cache. O(N) total work over N turns. Stale
/// content (e.g. prior `<think>` blocks for Qwen3-style models) lingers in the
/// cache because we never re-render history.
///
/// `full_replay`: each sendText/sendToolResult discards the cache, rebuilds the
/// system prefix, then re-renders every prior message via formatMessage with
/// `is_prior_turn = true` for assistant turns — letting the model strip prior
/// thinking content and other "older turn" details exactly as the canonical
/// chat template would. O(N²) total work over N turns. Required for thinking
/// scenarios on Qwen3-style models that strip prior reasoning.
pub const Mode = enum {
    incremental,
    full_replay,
};

/// Streaming state machine. Replaces the old state/pending_end_of_turn/
/// turn_finalized triple. Transitions:
///   idle ── sendText/sendToolResult/replay ──▶ streaming(.content)
///   streaming(.content) ── thinking_start token ──▶ streaming(.thinking)
///   streaming(.thinking) ── thinking_end token ──▶ streaming(.content)
///   streaming(*) ── tool_call_start token ──▶ streaming(.tool_call)
///   streaming(.tool_call) ── tool_call_end token ──▶ streaming(.content)
///   streaming(*) ── end_of_turn token / EOS / ContextFull ──▶ eot_emitted
///   eot_emitted ── next() ──▶ idle (returning null)
///   idle ── next() ──▶ idle (returning null, no-op)
const Phase = union(enum) {
    streaming: StreamKind,
    eot_emitted,
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

/// Initialize a chat session bound to the given runtime. The session prefix
/// (system_prompt + tools, rendered by the model's formatSystem) is prefilled
/// into the KV cache once here. Subsequent sendText/sendToolResult calls then
/// either append to the existing cache (`mode = .incremental`) or destroy and
/// rebuild it on every turn (`mode = .full_replay`). See the `Mode` doc above.
///
/// The model must implement the formatSystem/formatMessage API. Models that
/// haven't been ported return error.NewApiNotImplemented from formatSystem.
pub fn init(
    allocator: std.mem.Allocator,
    runtime: *Runtime,
    options: Model.ChatOptions,
    mode: Mode,
) !@This() {
    var context = try runtime.start();
    errdefer context.deinit();

    const prefix = try runtime.model.formatSystem(allocator, options);
    defer allocator.free(prefix);

    if (prefix.len > 0) {
        try context.prefill(prefix);
    }

    return .{
        .allocator = allocator,
        .arena = std.heap.ArenaAllocator.init(allocator),
        .runtime = runtime,
        .context = context,
        .messages = .empty,
        .options = options,
        .mode = mode,
    };
}

pub fn deinit(self: *@This()) void {
    self.context.deinit();
    self.arena.deinit();
}

/// Send a user message and prepare the model to generate.
///
/// In incremental mode (default), prefills exactly the new turn (user block +
/// assistant opener) on top of the existing cache.
///
/// In full_replay mode, destroys the cache, re-renders the entire history with
/// `is_prior_turn = true` for prior assistant messages, and prefills the result.
pub fn sendText(self: *@This(), text: []const u8) !void {
    const arena_alloc = self.arena.allocator();
    const owned = try arena_alloc.dupe(u8, text);
    const msg = Message{ .user = owned };
    try self.messages.append(arena_alloc, msg);

    try self.prefillNewTurn(msg);
    self.resetStreamingState();
}

/// Supply a tool result back to the model and prepare it to continue generating.
/// Mirrors sendText: appends incrementally or rebuilds depending on mode.
pub fn sendToolResult(self: *@This(), tool_call_id: []const u8, content: []const u8) !void {
    const arena_alloc = self.arena.allocator();
    const msg = Message{ .tool_result = .{
        .tool_call_id = try arena_alloc.dupe(u8, tool_call_id),
        .content = try arena_alloc.dupe(u8, content),
    } };
    try self.messages.append(arena_alloc, msg);

    try self.prefillNewTurn(msg);
    self.resetStreamingState();
}

/// Prefill the just-appended new turn (which is `self.messages.items[last]`),
/// dispatching on `self.mode`. Shared by sendText and sendToolResult.
fn prefillNewTurn(self: *@This(), msg: Message) !void {
    switch (self.mode) {
        .incremental => {
            const turn = try self.runtime.model.formatMessage(
                self.allocator,
                msg,
                .{ .prime_assistant = true, .thinking = self.options.thinking },
            );
            defer self.allocator.free(turn);
            if (turn.len > 0) try self.context.prefill(turn);
        },
        .full_replay => try self.rebuildContext(),
    }
}

/// Discard the current KV cache and re-render the conversation from scratch.
/// Used by sendText/sendToolResult in full_replay mode. The just-appended
/// new user/tool_result message is at `self.messages.items[last]` and is
/// rendered with `prime_assistant = true`. All other messages are rendered as
/// prior turns; assistant messages get `is_prior_turn = true` so models that
/// strip prior reasoning content (Qwen3) can apply the canonical rule.
fn rebuildContext(self: *@This()) !void {
    self.context.deinit();
    self.context = try self.runtime.start();

    // 1. System block.
    const prefix = try self.runtime.model.formatSystem(self.allocator, self.options);
    defer self.allocator.free(prefix);
    if (prefix.len > 0) try self.context.prefill(prefix);

    // 2. All prior messages (everything except the just-appended new turn).
    const last_idx = self.messages.items.len - 1;
    for (self.messages.items[0..last_idx]) |prior_msg| {
        const chunk = try self.runtime.model.formatMessage(
            self.allocator,
            prior_msg,
            .{
                .prime_assistant = false,
                .thinking = self.options.thinking,
                .is_prior_turn = (prior_msg == .assistant),
            },
        );
        defer self.allocator.free(chunk);
        if (chunk.len > 0) try self.context.prefill(chunk);
    }

    // 3. The just-appended new message, with assistant prime.
    const last = self.messages.items[last_idx];
    const turn = try self.runtime.model.formatMessage(
        self.allocator,
        last,
        .{
            .prime_assistant = true,
            .thinking = self.options.thinking,
            .is_prior_turn = false,
        },
    );
    defer self.allocator.free(turn);
    if (turn.len > 0) try self.context.prefill(turn);
}

/// Stream the next event in the current generation. Returns null when the turn
/// is fully drained (after .end_of_turn has been emitted once) or when no
/// generation is in progress.
pub fn next(self: *@This()) !?Event {
    const stream_kind = switch (self.phase) {
        .idle => return null,
        .eot_emitted => {
            self.phase = .idle;
            return null;
        },
        .streaming => |kind| kind,
    };

    const text = self.context.next() catch |err| {
        if (err == error.ContextFull) return try self.endTurn();
        return err;
    } orelse return try self.endTurn();

    const token_class = self.runtime.model.classifyToken(self.context.current_token);

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

/// Common end-of-turn transition: finalize the turn into messages and emit
/// the .end_of_turn event. Subsequent next() calls return null then idle.
fn endTurn(self: *@This()) !Event {
    try self.finalizeTurn();
    self.phase = .eot_emitted;
    return .end_of_turn;
}

/// Drain the current turn to completion and return the assembled assistant
/// message by value. The slices inside (content, thinking, tool_calls) point
/// at arena-owned storage and remain valid for the lifetime of the session,
/// but the returned struct itself is a copy that survives subsequent
/// sendText/sendToolResult calls (which may reallocate `self.messages`).
///
/// Convenience for blocking-style callers (tests, scripts) that don't need
/// token-by-token streaming.
pub fn receive(self: *@This()) !Message.Assistant {
    while (try self.next()) |event| switch (event) {
        .end_of_turn => break,
        else => {},
    };

    if (self.messages.items.len == 0) return error.NoTurnGenerated;
    const last = self.messages.items[self.messages.items.len - 1];
    if (last != .assistant) return error.NoTurnGenerated;
    return last.assistant;
}

/// Prefill a saved conversation history into this session. Must be called before
/// any sendText/sendToolResult calls. Each message is rendered via formatMessage
/// and prefilled incrementally on top of the existing system prefix.
///
/// The last message determines whether the assistant is primed for generation:
/// if it's a user or tool_result message, the assistant opener is appended (ready
/// to call next/receive); if it's an assistant message, the conversation is
/// considered closed (call sendText next).
pub fn replay(self: *@This(), messages: []const Message) !void {
    if (self.messages.items.len != 0) return error.AlreadyStarted;
    const arena_alloc = self.arena.allocator();

    for (messages, 0..) |msg, i| {
        const is_last = (i == messages.len - 1);
        const prime = is_last and (msg != .assistant);

        // All replayed assistant messages are inherently prior turns: the next
        // sendText will append more dialogue after them. For Qwen3-style models
        // this means their <think> blocks (if any) get stripped.
        const chunk = try self.runtime.model.formatMessage(
            self.allocator,
            msg,
            .{
                .prime_assistant = prime,
                .thinking = self.options.thinking,
                .is_prior_turn = (msg == .assistant),
            },
        );
        defer self.allocator.free(chunk);
        if (chunk.len > 0) try self.context.prefill(chunk);

        try self.messages.append(arena_alloc, try dupeMessage(arena_alloc, msg));
    }

    self.resetStreamingState();
}

fn resetStreamingState(self: *@This()) void {
    self.phase = .{ .streaming = .content };
    self.content_buf = .empty;
    self.thinking_buf = .empty;
    self.tool_call_buf = .empty;
    self.tool_calls = .empty;
}

/// Parse the JSON in `tool_call_buf` (one `<tool_call>...</tool_call>` body
/// in Hermes/Qwen3 convention) and append a `Message.ToolCall` to
/// `self.tool_calls`. Synthesizes the id since Qwen3 tool calls don't carry
/// one. Malformed JSON is silently dropped — the agent loop will then see no
/// tool_calls and treat the turn as content-only.
fn commitToolCall(self: *@This()) !void {
    const arena_alloc = self.arena.allocator();
    defer self.tool_call_buf.clearRetainingCapacity();

    const raw = std.mem.trim(u8, self.tool_call_buf.items, " \t\r\n");
    if (raw.len == 0) return;

    const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, raw, .{}) catch return;
    defer parsed.deinit();
    if (parsed.value != .object) return;
    const obj = parsed.value.object;

    const name_v = obj.get("name") orelse return;
    if (name_v != .string) return;

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

fn finalizeTurn(self: *@This()) !void {
    const arena_alloc = self.arena.allocator();

    const thinking: ?[]const u8 = if (self.thinking_buf.items.len > 0)
        try arena_alloc.dupe(u8, self.thinking_buf.items)
    else
        null;

    const content = if (self.content_buf.items.len > 0)
        try arena_alloc.dupe(u8, self.content_buf.items)
    else
        "";

    const tool_calls = if (self.tool_calls.items.len > 0)
        try arena_alloc.dupe(Message.ToolCall, self.tool_calls.items)
    else
        &.{};

    try self.messages.append(arena_alloc, .{ .assistant = .{
        .content = content,
        .thinking = thinking,
        .tool_calls = tool_calls,
    } });

    self.content_buf = .empty;
    self.thinking_buf = .empty;
    self.tool_call_buf = .empty;
    self.tool_calls = .empty;
}

/// Deep-copy a Message into arena storage so the caller can free or drop their
/// own copy after replay() returns.
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

const std = @import("std");
const Runtime = @import("runtime.zig");
const Model = @import("model.zig");
const Message = @import("message.zig").Message;
