//! Hermes-style tool-call body parser. Wired into a chat overlay via
//! `Chat.VTable.parseToolCall`. Used by every family whose tool calls
//! are emitted as JSON of the shape `{"name": "...", "arguments": {...}}`
//! between `tool_call_start` / `tool_call_end` boundary tokens — Qwen3
//! and Granite Hybrid today.

const std = @import("std");
const log = std.log.scoped(.chat_session);
const Chat = @import("chat.zig");

/// Parse one Hermes-style tool-call body. Matches
/// `Chat.VTable.parseToolCall`; stateless, so the `chat` self pointer
/// is unused. Hermes emits exactly one call per span, so the returned
/// slice always has length 1. Returns an OOM error if allocation fails;
/// returns `error.MalformedToolCall` for any content-shape problem
/// (non-JSON, non-object, missing/non-string `name`). Diagnostic is
/// logged at debug level via `chat_session` scope. Caller owns the slice
/// and each element's `name`/`arguments` — free all via the passed
/// allocator.
pub fn parseToolCall(
    chat: *Chat,
    allocator: std.mem.Allocator,
    body: []const u8,
) anyerror![]Chat.ParsedToolCall {
    _ = chat;
    const raw = std.mem.trim(u8, body, " \t\r\n");
    if (raw.len == 0) {
        logDrop("empty body", raw);
        return error.MalformedToolCall;
    }

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch |err| {
        logDrop("JSON parse error", raw);
        return err;
    };
    defer parsed.deinit();

    if (parsed.value != .object) {
        logDrop("top-level value is not an object", raw);
        return error.MalformedToolCall;
    }
    const obj = parsed.value.object;

    const name_v = obj.get("name") orelse {
        logDrop("missing \"name\" field", raw);
        return error.MalformedToolCall;
    };
    if (name_v != .string) {
        logDrop("\"name\" is not a string", raw);
        return error.MalformedToolCall;
    }

    const args_str: []u8 = if (obj.get("arguments")) |args_v|
        try std.json.Stringify.valueAlloc(allocator, args_v, .{})
    else
        try allocator.dupe(u8, "{}");
    errdefer allocator.free(args_str);

    const name_dup = try allocator.dupe(u8, name_v.string);
    errdefer allocator.free(name_dup);

    const result = try allocator.alloc(Chat.ParsedToolCall, 1);
    result[0] = .{ .name = name_dup, .arguments = args_str };
    return result;
}

fn logDrop(reason: []const u8, raw: []const u8) void {
    const preview = if (raw.len <= 200) raw else raw[0..200];
    const ellipsis: []const u8 = if (raw.len > 200) "..." else "";
    log.debug("tool_call dropped: {s}\n  raw: \"{s}{s}\"", .{ reason, preview, ellipsis });
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

/// Stateless parser — the self pointer is never dereferenced, so the
/// tests pass `undefined`.
const no_chat: *Chat = undefined;

fn freeCalls(calls: []Chat.ParsedToolCall) void {
    for (calls) |c| {
        testing.allocator.free(c.name);
        testing.allocator.free(c.arguments);
    }
    testing.allocator.free(calls);
}

test "parseToolCall accepts well-formed Hermes JSON" {
    const body =
        \\{"name": "get_weather", "arguments": {"city": "Tokyo"}}
    ;
    const parsed = try parseToolCall(no_chat, testing.allocator, body);
    defer freeCalls(parsed);

    try testing.expectEqual(@as(usize, 1), parsed.len);
    try testing.expectEqualStrings("get_weather", parsed[0].name);
    try testing.expectEqualStrings("{\"city\":\"Tokyo\"}", parsed[0].arguments);
}

test "parseToolCall defaults missing arguments to empty object" {
    const body =
        \\{"name": "noop"}
    ;
    const parsed = try parseToolCall(no_chat, testing.allocator, body);
    defer freeCalls(parsed);

    try testing.expectEqual(@as(usize, 1), parsed.len);
    try testing.expectEqualStrings("noop", parsed[0].name);
    try testing.expectEqualStrings("{}", parsed[0].arguments);
}

test "parseToolCall rejects malformed JSON" {
    try testing.expectError(error.SyntaxError, parseToolCall(no_chat, testing.allocator, "not json"));
    try testing.expectError(error.MalformedToolCall, parseToolCall(no_chat, testing.allocator, "[]"));
    try testing.expectError(error.MalformedToolCall, parseToolCall(no_chat, testing.allocator, "{}"));
    try testing.expectError(error.MalformedToolCall, parseToolCall(no_chat, testing.allocator, "{\"name\": 5}"));
    try testing.expectError(error.MalformedToolCall, parseToolCall(no_chat, testing.allocator, ""));
}
