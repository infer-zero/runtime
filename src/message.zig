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

/// Description of a callable tool exposed to the model.
/// Each model serializes this into its own native tool format inside chatFormat/formatSystem.
pub const ToolSpec = struct {
    name: []const u8,
    description: []const u8,
    parameters: Parameters,
};

/// Parameters for a tool. Use `simple` for the common flat case (covers nearly all
/// agentic test scenarios) or `json_schema` as an escape hatch for complex schemas
/// (nested objects, enums, unions) where the structured form is insufficient.
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
