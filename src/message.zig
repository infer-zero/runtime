//! Chat message types

pub const Message = union(enum) {
    system: System,
    user: []const u8,
    assistant: Assistant,
    tool_result: ToolResult,

    pub const System = struct {
        thinking: bool = false,
        content: ?[]const u8 = null,
        tools: []const ToolSpec = &.{},
    };

    pub const Assistant = struct {
        content: []const u8 = "",
        thinking: ?[]const u8 = null,
        tool_calls: []const ToolCall = &.{},
    };

    pub const ToolSpec = struct {
        name: []const u8,
        description: []const u8,
        parameters: []const Parameter,
    };

    pub const ToolCall = struct {
        name: []const u8,
        arguments: []const Argument,
    };

    pub const ToolResult = struct {
        content: []const u8,
    };

    pub const Argument = struct {
        name: []const u8,
        value: Value,
    };

    pub const Value = union(enum) {
        string: []const u8,
        integer: i64,
        float: f64,
        boolean: bool,
    };

    pub const Parameter = struct {
        name: []const u8,
        description: []const u8,
        type: ParameterType,
    };

    pub const ParameterType = enum {
        string,
        integer,
        float,
        boolean,
    };

    pub const Marker = enum {
        content,
        turn_start,
        turn_end,
        think_start,
        think_end,
        tool_call_start,
        tool_call_end,
    };
};

const std = @import("std");
