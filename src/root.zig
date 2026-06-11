// Tier 1 (pure data)
pub const Tensor = @import("tensor.zig");

// Tier 2 (services)
pub const Sampler = @import("sampler.zig");
pub const Tokenizer = @import("tokenizer.zig");
pub const Vocabulary = Tokenizer.Vocabulary;

// Chat overlay + message vocabulary (stateless, depends only on std;
// rendered by `Chat`, driven by `ChatSession`, held by `Model`).
pub const Chat = @import("chat.zig").Chat;
pub const ChatOptions = Chat.ChatOptions;
pub const SpecialTokens = Chat.SpecialTokens;
pub const TokenClass = Chat.TokenClass;
pub const Message = @import("chat.zig").Message;
pub const ToolSpec = @import("chat.zig").ToolSpec;
pub const Parameters = @import("chat.zig").Parameters;
pub const Parameter = @import("chat.zig").Parameter;
pub const ParamType = @import("chat.zig").ParamType;

// Tier 3 (session + factory)
pub const Context = @import("context.zig");
pub const Engine = @import("engine.zig");

// Tier 4 (aggregate)
pub const Model = @import("model.zig");

// Tier 5 (chat driver)
pub const ChatSession = @import("chat_session.zig");

// Tool-call body parsers shared across families.
pub const hermes = @import("hermes.zig");

test {
    _ = Tensor;
    _ = Sampler;
    _ = Tokenizer;
    _ = Context;
    _ = Model;
    _ = ChatSession;
    _ = @import("chat.zig");
    _ = hermes;
}
