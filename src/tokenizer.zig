//! Tokenizer interface. Like `Model` and `Context`, this is a view
//! type: implementations embed a `Tokenizer` field and install a
//! `VTable` whose methods recover the concrete type via
//! `@fieldParentPtr`. Most variants use the provided `Tokenizer.Bpe`
//! (built from a loader-produced `Vocabulary`); wrappers around
//! external libraries (e.g. llama.cpp) bring their own.
//!
//! `eos_token_id` is interface DATA: `Model.isEndOfTurn` needs it
//! cheaply and unconditionally, so it lives here instead of behind a
//! vtable call. Implementations snapshot it at init.

/// The designated end-of-sequence token. Raw completion stops on this;
/// `Chat` uses it as a fallback when its chat-specific end-of-turn
/// markers aren't set.
eos_token_id: TokenID,
vtable: *const VTable,

const Tokenizer = @This();

pub const TokenID = u32;

/// The default BPE implementation (and its `Vocabulary` data type).
/// Embeds `Tokenizer` as `.interface`.
pub const Bpe = @import("bpe_tokenizer.zig");
pub const Vocabulary = Bpe.Vocabulary;

pub const VTable = struct {
    /// Encode text to tokens. Returned slice is owned by the caller and
    /// freed via `allocator`.
    encode: *const fn (*Tokenizer, std.mem.Allocator, []const u8) anyerror![]const TokenID,

    /// Decode tokens to text. Returned slice is owned by the caller and
    /// freed via `allocator`.
    decode: *const fn (*Tokenizer, std.mem.Allocator, []const TokenID) anyerror![]const u8,
};

/// Encode text to tokens. Returns a freshly allocated, caller-owned slice.
pub fn encode(self: *Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]const TokenID {
    return try self.vtable.encode(self, allocator, text);
}

/// Decode tokens to text. Returns a freshly allocated, caller-owned slice.
pub fn decode(self: *Tokenizer, allocator: std.mem.Allocator, tokens: []const TokenID) ![]const u8 {
    return try self.vtable.decode(self, allocator, tokens);
}

const std = @import("std");
