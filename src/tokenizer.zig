//! Tokenizer interface.

vtable: *const VTable,

/// The default BPE implementation (and its `Vocabulary` data type).
/// Embeds `Tokenizer` as `.interface`.
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

pub const Bpe = @import("tokenizer_bpe.zig");
pub const Vocabulary = Bpe.Vocabulary;

const Tokenizer = @This();
pub const TokenID = u32;

const std = @import("std");
