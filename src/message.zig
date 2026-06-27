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

    pub const ToolCall = struct {
        id: []const u8,
        name: []const u8,
        arguments: []const u8,
    };

    pub const ToolResult = struct {
        id: []const u8,
        content: []const u8,
    };
};

pub const ToolSpec = struct {
    name: []const u8,
    description: []const u8,
    parameters: Parameters,
};

pub const Parameters = union(enum) {
    simple: []const SimpleParameter,
    structured: []const StructuredParameter,
    json_schema: []const u8,
};

pub const SimpleParameter = struct {
    name: []const u8,
    description: []const u8,
};

pub const StructuredParameter = struct {
    name: []const u8,
    description: []const u8,
    type: StructuredParameterType,
    required: bool = true,
};

pub const StructuredParameterType = enum {
    string,
    integer,
    number,
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
