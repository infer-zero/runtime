// Tier 1 (pure data)
pub const Tensor = @import("tensor.zig");

// Tier 2 (services)
pub const Sampler = @import("sampler.zig");
pub const Tokenizer = @import("tokenizer.zig");
pub const Vocabulary = Tokenizer.Vocabulary;

// Tier 3 (session + factory)
pub const Context = @import("context.zig");
pub const Engine = @import("engine.zig");

// Tier 4 (aggregate)
pub const Model = @import("model.zig");

// Tier 5 (chat driver + its overlay + chat vocabulary)
pub const ChatSession = @import("chat_session.zig");
pub const Chat = ChatSession.Chat;
pub const Tool = Chat.Tool;
pub const ChatOptions = Chat.ChatOptions;
pub const SpecialTokens = Chat.SpecialTokens;
pub const TokenClass = Chat.TokenClass;
pub const Message = ChatSession.Message;
pub const ToolSpec = ChatSession.ToolSpec;
pub const Parameters = ChatSession.Parameters;
pub const Parameter = ChatSession.Parameter;
pub const ParamType = ChatSession.ParamType;

test {
    _ = Tensor;
    _ = Sampler;
    _ = Tokenizer;
    _ = Context;
    _ = Model;
    _ = ChatSession;
    _ = Chat;
    _ = Tool;
}
