//! Aggregate handle that composes a model's pure data (`Info`), its tight
//! inference core (`Inference`), and optional capabilities (`Chat`, `Tool`).
//! Each model variant's `load()` function constructs one of these by wiring
//! its concrete implementation into an `Inference` and attaching whichever
//! optional features it supports.
//!
//! Cross-cutting features (chat, tool-calling, eventually speculative
//! decoding / grammar / LoRA / …) become new optional fields on this struct
//! plus variant opt-in — no edits to unrelated variants, no base VTable
//! changes.

info: Info,
inference: Inference,
chat: ?Chat = null,
tool: ?Tool = null,

const Model = @This();

pub fn deinit(self: *Model) void {
    self.inference.deinit();
}

pub fn vocabulary(self: *Model) Vocabulary {
    return self.inference.vocabulary();
}

pub fn createContext(self: *Model) !*anyopaque {
    return try self.inference.createContext();
}

pub fn destroyContext(self: *Model, ctx: *anyopaque) void {
    self.inference.destroyContext(ctx);
}

pub fn prefill(self: *Model, ctx: *anyopaque, tokens: []const u32) !void {
    try self.inference.prefill(ctx, tokens);
}

pub fn next(self: *Model, ctx: *anyopaque, token: u32) ![]const f32 {
    return try self.inference.next(ctx, token);
}

/// Classify a token via pure data lookup on `self.info.special_tokens`. Falls
/// back to matching `info.eos_token_id` so models that only populate
/// `eos_token_id` (and leave `special_tokens` defaulted) still see EOS as
/// `.end_of_turn`.
pub fn classifyToken(self: *const Model, token: u32) TokenClass {
    const st = self.info.special_tokens;
    if (st.thinking_start) |id| if (token == id) return .thinking_start;
    if (st.thinking_end) |id| if (token == id) return .thinking_end;
    if (st.tool_call_start) |id| if (token == id) return .tool_call_start;
    if (st.tool_call_end) |id| if (token == id) return .tool_call_end;
    if (st.end_of_turn) |id| if (token == id) return .end_of_turn;
    if (st.end_of_turn_alt) |id| if (token == id) return .end_of_turn;
    if (token == self.info.eos_token_id) return .end_of_turn;
    return .content;
}

/// Generic factory: construct the aggregate `Model` for a concrete variant
/// type `T` by calling `T.init(allocator, path)`, type-erasing it into an
/// `Inference`, and attaching the `Info` / `Chat` pieces reflected from
/// `T`'s public surface.
///
/// Required on `T`:
///   - `pub fn init(Allocator, []const u8) !*T`
///   - the six inference methods (`deinit`, `vocabulary`, `createContext`,
///     `destroyContext`, `prefill`, `next`) on `*T` with a `T.Context` type.
///   - `eos_token_id: u32` field (for `Info.eos_token_id`).
///   - `config: *` field with `vocabulary_size: usize` and `max_len: usize`
///     sub-fields (matches every existing variant).
///
/// Optional on `T`:
///   - `special_tokens: SpecialTokens` field (defaulted if absent).
///   - `pub fn formatSystem(Allocator, ChatOptions) ![]const u8`
///   - `pub fn formatMessage(Allocator, Message, MessageFormat) ![]const u8`
///   - `pub const end_of_turn_suffix: []const u8`
///
/// When both `formatSystem` and `formatMessage` are present, a `Chat` is
/// attached; otherwise `model.chat = null`. `tool` is always null at the
/// moment — variants that want native tool-call handling will grow an
/// analogous opt-in.
pub fn init(comptime T: type, allocator: std.mem.Allocator, path: []const u8) !Model {
    const concrete = try T.init(allocator, path);

    const special_tokens: Info.SpecialTokens = if (@hasField(T, "special_tokens"))
        concrete.special_tokens
    else
        .{};

    const info: Info = .{
        .eos_token_id = concrete.eos_token_id,
        .vocabulary_size = concrete.config.vocabulary_size,
        .max_len = concrete.config.max_len,
        .special_tokens = special_tokens,
    };

    const chat: ?Chat = if (@hasDecl(T, "formatSystem") and @hasDecl(T, "formatMessage"))
        .{
            .formatSystem = T.formatSystem,
            .formatMessage = T.formatMessage,
            .end_of_turn_suffix = if (@hasDecl(T, "end_of_turn_suffix")) T.end_of_turn_suffix else "",
        }
    else
        null;

    return .{
        .info = info,
        .inference = Inference.wrap(T).init(concrete).inference(),
        .chat = chat,
        .tool = null,
    };
}

/// Re-exports of types that used to live on `Model` directly. Variants still
/// import them as `@import("base").Model.ChatOptions` etc.; keeping these
/// aliases avoids churning every variant's import list.
pub const ChatOptions = Chat.ChatOptions;
pub const MessageFormat = Chat.MessageFormat;
pub const SpecialTokens = Info.SpecialTokens;
pub const TokenClass = Info.TokenClass;

const Info = @import("info.zig");
const Inference = @import("inference.zig");
const Chat = @import("chat.zig");
const Tool = @import("tool.zig");
const Vocabulary = @import("vocabulary.zig");

const std = @import("std");
const testing = std.testing;

test "classifyToken uses data path with eos fallback" {
    var m: Model = .{
        .info = .{
            .eos_token_id = 99,
            .vocabulary_size = 0,
            .max_len = 0,
            .special_tokens = .{
                .end_of_turn = 1,
                .end_of_turn_alt = 2,
                .thinking_start = 10,
                .thinking_end = 11,
                .tool_call_start = 20,
                .tool_call_end = 21,
            },
        },
        .inference = undefined,
    };

    try testing.expectEqual(TokenClass.end_of_turn, m.classifyToken(1));
    try testing.expectEqual(TokenClass.end_of_turn, m.classifyToken(2));
    try testing.expectEqual(TokenClass.thinking_start, m.classifyToken(10));
    try testing.expectEqual(TokenClass.thinking_end, m.classifyToken(11));
    try testing.expectEqual(TokenClass.tool_call_start, m.classifyToken(20));
    try testing.expectEqual(TokenClass.tool_call_end, m.classifyToken(21));
    try testing.expectEqual(TokenClass.content, m.classifyToken(50));
    // eos fallback when special_tokens doesn't enumerate it.
    try testing.expectEqual(TokenClass.end_of_turn, m.classifyToken(99));
}

test "classifyToken with empty special_tokens falls back to eos only" {
    var m: Model = .{
        .info = .{
            .eos_token_id = 2,
            .vocabulary_size = 0,
            .max_len = 0,
        },
        .inference = undefined,
    };

    try testing.expectEqual(TokenClass.end_of_turn, m.classifyToken(2));
    try testing.expectEqual(TokenClass.content, m.classifyToken(0));
}
