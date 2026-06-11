//! Aggregate view over a loaded variant: three borrowed pieces
//! (`tokenizer`, `engine`, and an optional `chat` overlay). A view, not an
//! owner — it has no `deinit` and no factory methods.
//!
//! The caller that constructed the concrete variant (e.g.
//! `var v = try MyVariant.load(...);`) owns the variant's lifecycle and
//! calls the variant's own deinit when done. Create contexts via
//! `model.engine.createContext(...)`; see `Engine.createContext` for the
//! wrapper-ownership contract.

/// Wraps the variant's `Vocabulary`: merges, encoding/decoding maps, and the
/// designated `eos_token_id`.
tokenizer: Tokenizer,
/// The variant's embedded `Engine`: metadata plus the `createContext` factory.
engine: *Engine,
/// Optional chat-template overlay (with optional nested `Tool`). Holder only;
/// `ChatSession` is the actual consumer. Null for completion-only models.
chat: ?Chat = null,

const Model = @This();

/// True iff `token` ends a turn in this model. Raw-completion callers
/// stop on the vocabulary's designated EOS; chat-capable models may also
/// emit distinct end-of-turn markers (Qwen's `<|im_end|>`, Llama 3's
/// `<|eot_id|>`) which `Chat.isEndOfTurn` knows about.
pub fn isEndOfTurn(self: *const Model, token: u32) bool {
    if (token == self.tokenizer.vocabulary.eos_token_id) return true;
    if (self.chat) |c| return c.isEndOfTurn(token);
    return false;
}

const Tokenizer = @import("tokenizer.zig");
const Engine = @import("engine.zig");
const Chat = @import("chat.zig").Chat;
