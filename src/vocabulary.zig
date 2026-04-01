merge_index: MergePairIndex,
encoding: EncodingVocabulary,
decoding: DecodingVocabulary,

unknown_token: ?[]const u8 = null,
normalizer: ?Normalizer = null,
post_processor: ?PostProcessor = null,
use_byte_level: bool = false,

special_tokens: SpecialTokens = .empty,
special_tokens_sorted: []const SpecialTokenEntry = &.{},

pub const TokenID = u32;

pub const Subword = []const u8;
pub const MergePairIndex = std.StringHashMapUnmanaged(usize);
pub const EncodingVocabulary = std.StringHashMapUnmanaged(TokenID);
pub const DecodingVocabulary = std.AutoHashMapUnmanaged(TokenID, Subword);
pub const SpecialTokens = std.StringHashMapUnmanaged(TokenID);

pub const SpecialTokenEntry = struct {
    text: []const u8,
    id: TokenID,
};

pub const Normalizer = union(enum) {
    sequence: []const Normalizer,
    prepend: []const u8,
    replace: struct {
        pattern: []const u8,
        content: []const u8,
    },
};

pub const PostProcessor = union(enum) {
    sequence: []const PostProcessor,
    template: []const TemplateProcessing,

    pub const TemplateProcessing = union(enum) {
        sequence: void,
        special_token: TokenID,
    };
};

const std = @import("std");
