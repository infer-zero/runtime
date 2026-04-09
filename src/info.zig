//! Pure data describing a loaded model. No function pointers, no behavior —
//! just the facts `Runtime`, `ChatSession`, and token classification need at
//! runtime. Each model variant populates this in its `load()` function and
//! hands it to the `Model` aggregate.

eos_token_id: u32,
vocabulary_size: usize,
max_len: usize,
special_tokens: SpecialTokens = .{},

const Info = @This();

/// Per-model semantic special-token IDs. Populated by each model at init time
/// from tokenizer lookups. All fields are `?u32`; null means "this model
/// doesn't use that token class."
///
/// `end_of_turn` and `end_of_turn_alt` cover the realistic upper bound — no
/// model in the wild has more than two end-of-turn tokens (Qwen uses both
/// `<|im_end|>` and `<|end_of_text|>`; Llama 3 uses `<|eot_id|>` and
/// `<|end_of_text|>`; most other models have just one).
pub const SpecialTokens = struct {
    end_of_turn: ?u32 = null,
    end_of_turn_alt: ?u32 = null,
    thinking_start: ?u32 = null,
    thinking_end: ?u32 = null,
    tool_call_start: ?u32 = null,
    tool_call_end: ?u32 = null,
};

/// Semantic classes used by ChatSession to drive its streaming state machine.
pub const TokenClass = enum {
    content,
    thinking_start,
    thinking_end,
    tool_call_start,
    tool_call_end,
    end_of_turn,
};
