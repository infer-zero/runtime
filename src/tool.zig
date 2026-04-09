//! Optional tool-calling support. Scaffolding only at this stage — no model
//! variant opts in yet. ChatSession's existing generic JSON parser still runs
//! when `model.tool == null`. Variants that want native tool-call rendering
//! or parsing populate this struct in their `load()` function.

/// Render a tool spec block to be included in the system prompt. Called by
/// variants that want to override the default system-prompt tool rendering
/// their `formatSystem` does today (mostly a forward-looking hook).
renderToolSpec: *const fn (Allocator, []const ToolSpec) anyerror![]const u8,

/// Parse a single tool-call payload (the text between a tool_call_start and
/// tool_call_end token) into a structured `ParsedToolCall`. Called by
/// `ChatSession.commitToolCall` when `model.tool != null`; otherwise the
/// generic Hermes/Qwen3-style JSON parser runs.
parseToolCall: *const fn (Allocator, []const u8) anyerror!ParsedToolCall,

const Tool = @This();

pub const ParsedToolCall = struct {
    name: []const u8,
    arguments: []const u8,
};

const std = @import("std");
const Allocator = std.mem.Allocator;
const ToolSpec = @import("message.zig").ToolSpec;
