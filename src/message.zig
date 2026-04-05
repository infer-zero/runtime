pub const Message = union(enum) {
    system: []const u8,
    user: []const u8,
    assistant: Assistant,
    tool_result: ToolResult,

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
        tool_call_id: []const u8,
        content: []const u8,
    };
};
