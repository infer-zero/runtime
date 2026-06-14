// Tier 1 (pure data)
pub const Tensor = @import("tensor.zig");

// Tier 2 (services)
pub const Sampler = @import("sampler.zig");
pub const Tokenizer = @import("tokenizer.zig");
pub const Vocabulary = Tokenizer.Vocabulary;

// Chat overlay (template interface; driven by `ChatSession`, held by
// `Model`) + the message vocabulary it renders (`message.zig`).
pub const Chat = @import("chat.zig");
pub const ChatOptions = Chat.ChatOptions;
pub const SpecialTokens = Chat.SpecialTokens;
pub const TokenClass = Chat.TokenClass;
pub const Message = @import("message.zig").Message;
pub const ToolSpec = @import("message.zig").ToolSpec;
pub const Parameters = @import("message.zig").Parameters;
pub const Parameter = @import("message.zig").Parameter;
pub const ParamType = @import("message.zig").ParamType;

// Tier 3 (session)
pub const Context = @import("context.zig");

// Tier 4 (aggregate handle: owns lifecycle, factory for contexts)
pub const Model = @import("model.zig");

// Tier 5 (chat driver)
pub const ChatSession = @import("chat_session.zig");

// Tool-call body parsers shared across families.
pub const hermes = @import("hermes.zig");

test {
    _ = Tensor;
    _ = Sampler;
    _ = Sampler.Default;
    _ = Tokenizer;
    _ = Tokenizer.Bpe;
    _ = Context;
    _ = Model;
    _ = ChatSession;
    _ = Chat;
    _ = @import("message.zig");
    _ = hermes;
}
