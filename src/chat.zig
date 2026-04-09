//! Optional chat support. A model variant opts in by constructing a `Chat`
//! struct in its `load()` function, pointing at its own `formatSystem` and
//! `formatMessage` free functions. Models that don't support chat simply
//! leave `model.chat = null`; `ChatSession.init` then fails fast with
//! `error.ModelDoesNotSupportChat` instead of surfacing a runtime error
//! mid-generation.
//!
//! Chat-only concerns that used to leak into the core `Model` struct live
//! here: `ChatOptions`, `MessageFormat`, and `end_of_turn_suffix`.

/// Direct free-function pointers — no type erasure needed since these take
/// `(Allocator, ...)` not `(*T)`. Each variant's `load()` fills these with
/// plain references to its existing `pub fn formatSystem` / `formatMessage`.
formatSystem: *const fn (Allocator, ChatOptions) anyerror![]const u8,
formatMessage: *const fn (Allocator, Message, MessageFormat) anyerror![]const u8,

/// Text written to the KV cache immediately after the model emits its
/// end-of-turn token, to keep the cache aligned with the canonical chat
/// template (which typically has e.g. `<|im_end|>\n` between turns — the
/// model only emits the `<|im_end|>` part). For ChatML models (Qwen, Phi4,
/// LFM2 etc.) set this to "\n". Empty for models with no inter-turn
/// separator (Llama 2, Granite).
///
/// `ChatSession` owns the injection. `Runtime` is agnostic.
end_of_turn_suffix: []const u8 = "",

const Chat = @This();

pub const ChatOptions = struct {
    system_prompt: ?[]const u8 = null,
    thinking: bool = false,
    tools: []const ToolSpec = &.{},
};

/// Per-message rendering hints passed to `formatMessage`.
pub const MessageFormat = struct {
    /// If true, append the model's "open assistant turn" marker after the
    /// message, priming the model to generate immediately on the next decode
    /// call.
    prime_assistant: bool,
    /// Whether thinking mode is enabled. When false, ChatML models that use
    /// the canonical empty-think-block convention (Qwen3) inject
    /// `<think></think>` after the assistant prime so the model skips
    /// reasoning and produces a direct answer. ChatSession passes this from
    /// session.options.thinking.
    thinking: bool = false,
    /// True when this message is being re-rendered as part of prior
    /// conversation history (not the most recent turn). For Qwen3-style
    /// models that strip `<think>` blocks from older assistant turns per the
    /// canonical chat template, `formatMessage` uses this flag to omit the
    /// reasoning content. Models without a prior-vs-current distinction can
    /// ignore this field. Set by ChatSession.rebuildContext (full_replay
    /// mode) and by ChatSession.replay for replayed assistant messages.
    is_prior_turn: bool = false,
};

/// Render a full conversation as a single prompt string, using a model's
/// optional chat support. Replaces the old `model.chatFormat` vtable entry.
/// Returns `error.ModelDoesNotSupportChat` if `model.chat == null`.
pub fn renderConversation(
    model: *Model,
    allocator: Allocator,
    messages: []const Message,
    options: ChatOptions,
) ![]const u8 {
    const chat = model.chat orelse return error.ModelDoesNotSupportChat;

    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    const system = try chat.formatSystem(allocator, options);
    defer allocator.free(system);
    try buf.appendSlice(allocator, system);

    for (messages, 0..) |msg, i| {
        const is_last = (i == messages.len - 1);
        const chunk = try chat.formatMessage(allocator, msg, .{
            .prime_assistant = is_last and (msg != .assistant),
            .thinking = options.thinking,
            .is_prior_turn = false,
        });
        defer allocator.free(chunk);
        try buf.appendSlice(allocator, chunk);
    }

    return try buf.toOwnedSlice(allocator);
}

const std = @import("std");
const Allocator = std.mem.Allocator;
const Model = @import("model.zig");
const message_mod = @import("message.zig");
const Message = message_mod.Message;
const ToolSpec = message_mod.ToolSpec;
