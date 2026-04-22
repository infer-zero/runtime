//! Aggregate view over a loaded variant. Three pieces, all borrowed:
//!   - `tokenizer`: wraps the variant's `Vocabulary` (merges, encoding/
//!                  decoding maps, designated `eos_token_id`)
//!   - `engine`:    pointer to the variant's embedded `Engine` (metadata +
//!                  `createContext` factory)
//!   - `chat`:      optional `ChatSession.Chat` overlay (with optional
//!                  nested `Tool`) — holder only; `ChatSession` is the
//!                  actual consumer.
//!
//! Model is a **view**, not an owner. It has no `deinit` and no factory
//! methods. The caller that constructed the concrete variant (e.g.
//! `var v = try MyVariant.load(...);`) owns the variant's lifecycle and
//! calls the variant's own deinit when done. Contexts are created by
//! calling `model.engine.createContext(...)`, which returns a `*Context`
//! pointing into a variant-specific `ConcreteContext` wrapper the
//! factory heap-allocated. Comptime-aware callers own the wrapper via
//! `@fieldParentPtr`; polymorphic borrowers (ChatSession) just use the
//! `*Context` and do not deinit.

tokenizer: Tokenizer,
engine: *Engine,
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
const Chat = @import("chat_session.zig").Chat;
