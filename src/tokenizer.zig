//! BPE Tokenizer for text encoding/decoding, plus the `Vocabulary` data it
//! operates on (nested below). Every `Model` owns one `Tokenizer`, built from
//! the `Vocabulary` loaded by the variant.

const log = std.log.scoped(.infer);

const std = @import("std");

const MetaSpace: []const u8 = "▁";

vocabulary: Vocabulary,

const Tokenizer = @This();

pub const TokenID = u32;

/// Initialize tokenizer with vocabulary data.
/// The vocabulary's backing storage must remain valid for the tokenizer's lifetime.
pub fn init(vocabulary: Vocabulary) Tokenizer {
    return .{ .vocabulary = vocabulary };
}

/// Decode tokens to text
pub fn decode(self: Tokenizer, allocator: std.mem.Allocator, tokens: []const TokenID) ![]const u8 {
    var text: std.ArrayListUnmanaged(u8) = .empty;

    for (tokens) |token| {
        if (self.vocabulary.decoding.get(token)) |word| {
            const utf8_view = std.unicode.Utf8View.init(word) catch {
                try text.appendSlice(allocator, word);
                continue;
            };
            var utf8_iter = utf8_view.iterator();
            while (utf8_iter.nextCodepointSlice()) |codepoint_slice| {
                if (std.mem.eql(u8, codepoint_slice, MetaSpace)) {
                    try text.append(allocator, ' ');
                } else {
                    const codepoint = std.unicode.utf8Decode(codepoint_slice) catch {
                        try text.appendSlice(allocator, codepoint_slice);
                        continue;
                    };
                    if (codepoint < ByteLevelTable.unicode_to_byte.len) {
                        const byte_val = ByteLevelTable.unicode_to_byte[codepoint];
                        if (byte_val != 0 or codepoint == 0) {
                            try text.append(allocator, byte_val);
                        } else {
                            try text.appendSlice(allocator, codepoint_slice);
                        }
                    } else {
                        try text.appendSlice(allocator, codepoint_slice);
                    }
                }
            }
        }
    }

    return try text.toOwnedSlice(allocator);
}

/// Decode a single token (returns raw form)
pub fn decodeToken(self: Tokenizer, token: TokenID) []const u8 {
    return self.vocabulary.decoding.get(token) orelse "";
}

/// Encode text to tokens
pub fn encode(self: Tokenizer, allocator: std.mem.Allocator, input: []const u8) ![]const TokenID {
    const special_sorted = self.vocabulary.special_tokens_sorted;

    var result: std.ArrayListUnmanaged(TokenID) = .empty;
    defer result.deinit(allocator);

    if (special_sorted.len > 0) {
        // Split input on special tokens, then encode each regular segment
        var segments: std.ArrayListUnmanaged(Segment) = .empty;
        defer segments.deinit(allocator);
        try splitOnSpecialTokens(allocator, input, special_sorted, &segments);

        for (segments.items) |segment| {
            switch (segment) {
                .special_token_id => |id| try result.append(allocator, id),
                .text => |text| try self.encodeRegularText(allocator, text, &result),
            }
        }
    } else {
        try self.encodeRegularText(allocator, input, &result);
    }

    const tokens = try result.toOwnedSlice(allocator);
    defer allocator.free(tokens);

    const post = if (self.vocabulary.post_processor) |post_processor|
        try postProcess(allocator, post_processor, tokens)
    else
        try allocator.dupe(TokenID, tokens);

    return post;
}

const Segment = union(enum) {
    text: []const u8,
    special_token_id: TokenID,
};

fn splitOnSpecialTokens(
    allocator: std.mem.Allocator,
    input: []const u8,
    special_sorted: []const Vocabulary.SpecialTokenEntry,
    segments: *std.ArrayListUnmanaged(Segment),
) !void {
    var pos: usize = 0;
    var last_regular_start: usize = 0;

    while (pos < input.len) {
        var matched = false;
        for (special_sorted) |entry| {
            if (pos + entry.text.len <= input.len and
                std.mem.eql(u8, input[pos..][0..entry.text.len], entry.text))
            {
                // Flush any preceding regular text
                if (pos > last_regular_start) {
                    try segments.append(allocator, .{ .text = input[last_regular_start..pos] });
                }
                try segments.append(allocator, .{ .special_token_id = entry.id });
                pos += entry.text.len;
                last_regular_start = pos;
                matched = true;
                break;
            }
        }
        if (!matched) {
            pos += 1;
        }
    }

    // Flush trailing regular text
    if (last_regular_start < input.len) {
        try segments.append(allocator, .{ .text = input[last_regular_start..] });
    }
}

fn encodeRegularText(self: Tokenizer, allocator: std.mem.Allocator, input: []const u8, result: *std.ArrayListUnmanaged(TokenID)) !void {
    if (input.len == 0) return;

    const normalized = if (self.vocabulary.normalizer) |normalizer|
        try normalize(allocator, normalizer, input)
    else
        try allocator.dupe(u8, input);
    defer allocator.free(normalized);

    const pre_processed = if (self.vocabulary.use_byte_level)
        try ByteLevelTable.encode(allocator, normalized)
    else
        try sentencePiecePreprocess(allocator, normalized);
    defer allocator.free(pre_processed);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const subwords = try self.splitAndMerge(arena.allocator(), pre_processed);

    for (subwords) |subword| {
        if (self.vocabulary.encoding.get(subword)) |tokenID| {
            try result.append(allocator, tokenID);
        } else if (self.vocabulary.unknown_token) |unk| {
            const unk_token = self.vocabulary.encoding.get(unk) orelse unreachable;
            try result.append(allocator, unk_token);
        } else {
            log.err("tokenizer: token not found in vocabulary and no unknown token defined", .{});
            return error.TokenNotFound;
        }
    }
}

fn splitAndMerge(self: Tokenizer, arena: std.mem.Allocator, input: []const u8) ![]const []const u8 {
    var subwords: std.ArrayListUnmanaged([]const u8) = .empty;

    const utf8_view = std.unicode.Utf8View.init(input) catch unreachable;
    var utf8_iter = utf8_view.iterator();
    while (utf8_iter.nextCodepointSlice()) |char| {
        try subwords.append(arena, char);
    }

    if (subwords.items.len == 0) {
        return &.{};
    }

    var merged_buffer: std.ArrayListUnmanaged(u8) = .empty;
    var key_buf: std.ArrayListUnmanaged(u8) = .empty;

    while (true) {
        var best_merge_idx: ?usize = null;
        var best_position: usize = 0;

        for (0..subwords.items.len - 1) |pos| {
            const left = subwords.items[pos];
            const right = subwords.items[pos + 1];

            key_buf.clearRetainingCapacity();
            try key_buf.appendSlice(arena, left);
            try key_buf.append(arena, 0);
            try key_buf.appendSlice(arena, right);

            if (self.vocabulary.merge_index.get(key_buf.items)) |merge_idx| {
                if (best_merge_idx == null or merge_idx < best_merge_idx.?) {
                    best_merge_idx = merge_idx;
                    best_position = pos;
                }
            }
        }

        if (best_merge_idx == null) break;

        const left = subwords.items[best_position];
        const right = subwords.items[best_position + 1];

        merged_buffer.clearRetainingCapacity();
        try merged_buffer.appendSlice(arena, left);
        try merged_buffer.appendSlice(arena, right);

        subwords.items[best_position] = try arena.dupe(u8, merged_buffer.items);
        _ = subwords.orderedRemove(best_position + 1);
    }

    return subwords.items;
}

fn normalize(allocator: std.mem.Allocator, normalizer: Vocabulary.Normalizer, text: []const u8) ![]const u8 {
    switch (normalizer) {
        .sequence => |seq| {
            var result = text;
            var prev: ?[]const u8 = null;
            for (seq) |norm| {
                result = try normalize(allocator, norm, result);
                if (prev) |previous| allocator.free(previous);
                prev = result;
            }
            return result;
        },
        .prepend => |prefix| {
            const result = try allocator.alloc(u8, prefix.len + text.len);
            @memcpy(result[0..prefix.len], prefix);
            @memcpy(result[prefix.len..], text);
            return result;
        },
        .replace => |replace| {
            return try std.mem.replaceOwned(u8, allocator, text, replace.pattern, replace.content);
        },
    }
}

fn postProcess(allocator: std.mem.Allocator, post_processor: Vocabulary.PostProcessor, tokens: []const TokenID) ![]const TokenID {
    switch (post_processor) {
        .sequence => |seq| {
            var result = tokens;
            var prev: ?[]const TokenID = null;
            for (seq) |post| {
                result = try postProcess(allocator, post, result);
                if (prev) |previous| allocator.free(previous);
                prev = result;
            }
            return result;
        },
        .template => |template| {
            var result: std.ArrayListUnmanaged(TokenID) = .empty;
            defer result.deinit(allocator);

            for (template) |tmpl| {
                switch (tmpl) {
                    .special_token => |tokenID| try result.append(allocator, tokenID),
                    .sequence => try result.appendSlice(allocator, tokens),
                }
            }

            return try result.toOwnedSlice(allocator);
        },
    }
}

/// GPT-2 style ByteLevel encoding table.
const ByteLevelTable = struct {
    const byte_to_unicode: [256]u21 = blk: {
        var table: [256]u21 = undefined;
        var n: u21 = 0;

        for (0..256) |b| {
            const byte: u8 = @intCast(b);
            if ((byte >= 33 and byte <= 126) or
                (byte >= 161 and byte <= 172) or
                (byte >= 174 and byte <= 255))
            {
                table[b] = byte;
            } else {
                table[b] = 0xFFFF;
            }
        }

        for (0..256) |b| {
            if (table[b] == 0xFFFF) {
                table[b] = 256 + n;
                n += 1;
            }
        }

        break :blk table;
    };

    const unicode_to_byte: [324]u8 = blk: {
        var table: [324]u8 = undefined;
        @memset(&table, 0);

        for (0..256) |b| {
            const codepoint = byte_to_unicode[b];
            if (codepoint < 324) {
                table[codepoint] = @intCast(b);
            }
        }

        break :blk table;
    };

    fn encodeByteToUtf8(byte: u8, out: *[4]u8) u3 {
        const codepoint = byte_to_unicode[byte];
        if (codepoint < 0x80) {
            out[0] = @intCast(codepoint);
            return 1;
        } else if (codepoint < 0x800) {
            out[0] = @intCast(0xC0 | (codepoint >> 6));
            out[1] = @intCast(0x80 | (codepoint & 0x3F));
            return 2;
        } else {
            out[0] = @intCast(0xE0 | (codepoint >> 12));
            out[1] = @intCast(0x80 | ((codepoint >> 6) & 0x3F));
            out[2] = @intCast(0x80 | (codepoint & 0x3F));
            return 3;
        }
    }

    fn encode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        var result: std.ArrayListUnmanaged(u8) = .empty;
        errdefer result.deinit(allocator);
        try result.ensureTotalCapacity(allocator, input.len * 3);

        for (input) |byte| {
            var utf8_buf: [4]u8 = undefined;
            const len = encodeByteToUtf8(byte, &utf8_buf);
            try result.appendSlice(allocator, utf8_buf[0..len]);
        }

        return result.toOwnedSlice(allocator);
    }
};

fn sentencePiecePreprocess(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var result: std.ArrayListUnmanaged(u8) = .empty;
    errdefer result.deinit(allocator);

    const already_has_prefix = input.len >= MetaSpace.len and
        std.mem.eql(u8, input[0..MetaSpace.len], MetaSpace);

    var is_first = !already_has_prefix;
    var i: usize = 0;
    while (i < input.len) {
        if (input[i] == ' ') {
            try result.appendSlice(allocator, MetaSpace);
            i += 1;
        } else {
            const byte = input[i];
            const seq_len: usize = if (byte < 0x80) 1 else if (byte < 0xE0) 2 else if (byte < 0xF0) 3 else 4;
            const end = @min(i + seq_len, input.len);

            if (is_first) {
                try result.appendSlice(allocator, MetaSpace);
                is_first = false;
            }

            try result.appendSlice(allocator, input[i..end]);
            i = end;
        }
    }

    return result.toOwnedSlice(allocator);
}

/// Raw tokenizer data: merge tables, encoding/decoding maps, normalizer/
/// post-processor, special-token registry, and the designated EOS token.
/// Produced by the variant's loader (HuggingFace or GGUF) and passed to
/// `Tokenizer.init`. The *semantic* interpretation of special tokens
/// (end-of-turn, thinking markers, tool-call framing) lives on `Chat`, not
/// here — this struct is just the tokenizer's data.
pub const Vocabulary = struct {
    merge_index: MergePairIndex,
    encoding: EncodingVocabulary,
    decoding: DecodingVocabulary,

    unknown_token: ?[]const u8 = null,
    normalizer: ?Normalizer = null,
    post_processor: ?PostProcessor = null,
    use_byte_level: bool = false,

    special_tokens: SpecialTokens = .empty,
    special_tokens_sorted: []const SpecialTokenEntry = &.{},

    /// The vocabulary's designated end-of-sequence token. Raw completion
    /// stops on this; `Chat` uses it as a fallback when its chat-specific
    /// end-of-turn markers aren't set. Defaults to 0 — variants/adapters
    /// must populate it from the parsed tokenizer config at load time.
    eos_token_id: TokenID = 0,

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
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

fn createTestVocabulary(allocator: std.mem.Allocator) !Vocabulary {
    var encoding: Vocabulary.EncodingVocabulary = .{};
    // Basic characters
    try encoding.put(allocator, "h", 0);
    try encoding.put(allocator, "e", 1);
    try encoding.put(allocator, "l", 2);
    try encoding.put(allocator, "o", 3);
    try encoding.put(allocator, "he", 4);
    try encoding.put(allocator, "ll", 5);
    try encoding.put(allocator, "lo", 6);
    try encoding.put(allocator, "hel", 7);
    try encoding.put(allocator, "hello", 8);
    try encoding.put(allocator, "<unk>", 9);
    // With MetaSpace prefix (used after sentencePiecePreprocess)
    try encoding.put(allocator, MetaSpace ++ "h", 10);
    try encoding.put(allocator, MetaSpace ++ "he", 11);
    try encoding.put(allocator, MetaSpace ++ "hel", 12);
    try encoding.put(allocator, MetaSpace ++ "hello", 13);

    var decoding: Vocabulary.DecodingVocabulary = .{};
    try decoding.put(allocator, 0, "h");
    try decoding.put(allocator, 1, "e");
    try decoding.put(allocator, 2, "l");
    try decoding.put(allocator, 3, "o");
    try decoding.put(allocator, 4, "he");
    try decoding.put(allocator, 5, "ll");
    try decoding.put(allocator, 6, "lo");
    try decoding.put(allocator, 7, "hel");
    try decoding.put(allocator, 8, "hello");
    try decoding.put(allocator, 9, "<unk>");
    try decoding.put(allocator, 10, MetaSpace ++ "h");
    try decoding.put(allocator, 11, MetaSpace ++ "he");
    try decoding.put(allocator, 12, MetaSpace ++ "hel");
    try decoding.put(allocator, 13, MetaSpace ++ "hello");

    // Merge pairs with priority (lower index = higher priority)
    var merge_index: Vocabulary.MergePairIndex = .{};
    try merge_index.put(allocator, "h\x00e", 0);
    try merge_index.put(allocator, "l\x00l", 1);
    try merge_index.put(allocator, "l\x00o", 2);
    try merge_index.put(allocator, "he\x00l", 3);
    try merge_index.put(allocator, "hel\x00lo", 4);
    try merge_index.put(allocator, MetaSpace ++ "\x00h", 5);
    try merge_index.put(allocator, MetaSpace ++ "h\x00e", 6);
    try merge_index.put(allocator, MetaSpace ++ "he\x00l", 7);
    try merge_index.put(allocator, MetaSpace ++ "hel\x00lo", 8);

    return Vocabulary{
        .merge_index = merge_index,
        .encoding = encoding,
        .decoding = decoding,
        .unknown_token = "<unk>",
        .use_byte_level = false,
        .eos_token_id = 0,
    };
}

fn cleanupTestVocabulary(allocator: std.mem.Allocator, vocab: *Vocabulary) void {
    vocab.encoding.deinit(allocator);
    vocab.decoding.deinit(allocator);
    vocab.merge_index.deinit(allocator);
}

test "decodeToken returns token string" {
    const allocator = testing.allocator;
    var vocab = try createTestVocabulary(allocator);
    defer cleanupTestVocabulary(allocator, &vocab);

    const tokenizer = Tokenizer.init(vocab);

    try testing.expectEqualStrings("hello", tokenizer.decodeToken(8));
    try testing.expectEqualStrings("he", tokenizer.decodeToken(4));
    try testing.expectEqualStrings("", tokenizer.decodeToken(999));
}

test "decode converts tokens to text" {
    const allocator = testing.allocator;
    var vocab = try createTestVocabulary(allocator);
    defer cleanupTestVocabulary(allocator, &vocab);

    const tokenizer = Tokenizer.init(vocab);

    const tokens = [_]TokenID{8}; // "hello"
    const text = try tokenizer.decode(allocator, &tokens);
    defer allocator.free(text);

    try testing.expectEqualStrings("hello", text);
}

test "decode handles multiple tokens" {
    const allocator = testing.allocator;
    var vocab = try createTestVocabulary(allocator);
    defer cleanupTestVocabulary(allocator, &vocab);

    const tokenizer = Tokenizer.init(vocab);

    const tokens = [_]TokenID{ 4, 5, 3 }; // "he" + "ll" + "o"
    const text = try tokenizer.decode(allocator, &tokens);
    defer allocator.free(text);

    try testing.expectEqualStrings("hello", text);
}

test "encode tokenizes text using BPE merges" {
    const allocator = testing.allocator;

    var encoding: Vocabulary.EncodingVocabulary = .{};
    try encoding.put(allocator, MetaSpace, 0);
    try encoding.put(allocator, "a", 1);
    try encoding.put(allocator, "b", 2);
    try encoding.put(allocator, "ab", 3);
    defer encoding.deinit(allocator);

    var decoding: Vocabulary.DecodingVocabulary = .{};
    try decoding.put(allocator, 0, MetaSpace);
    try decoding.put(allocator, 1, "a");
    try decoding.put(allocator, 2, "b");
    try decoding.put(allocator, 3, "ab");
    defer decoding.deinit(allocator);

    var merge_index: Vocabulary.MergePairIndex = .{};
    try merge_index.put(allocator, "a\x00b", 0);
    defer merge_index.deinit(allocator);

    const vocab = Vocabulary{
        .merge_index = merge_index,
        .encoding = encoding,
        .decoding = decoding,
        .use_byte_level = false,
        .eos_token_id = 0,
    };

    const tokenizer = Tokenizer.init(vocab);

    const tokens = try tokenizer.encode(allocator, "ab");
    defer allocator.free(tokens);

    try testing.expectEqual(@as(usize, 2), tokens.len);
    try testing.expectEqual(@as(TokenID, 0), tokens[0]); // ▁
    try testing.expectEqual(@as(TokenID, 3), tokens[1]); // ab
}

test "encode handles unknown tokens" {
    const allocator = testing.allocator;
    var vocab = try createTestVocabulary(allocator);
    defer cleanupTestVocabulary(allocator, &vocab);

    const tokenizer = Tokenizer.init(vocab);

    const tokens = try tokenizer.encode(allocator, "x");
    defer allocator.free(tokens);

    try testing.expectEqual(@as(usize, 2), tokens.len);
    try testing.expectEqual(@as(TokenID, 9), tokens[0]); // <unk> for ▁
    try testing.expectEqual(@as(TokenID, 9), tokens[1]); // <unk> for x
}

test "encode works without unknown token when all chars in vocab" {
    const allocator = testing.allocator;
    var vocab = try createTestVocabulary(allocator);
    defer cleanupTestVocabulary(allocator, &vocab);

    try vocab.encoding.put(allocator, MetaSpace, 14);
    try vocab.encoding.put(allocator, "x", 15);
    try vocab.decoding.put(allocator, 14, MetaSpace);
    try vocab.decoding.put(allocator, 15, "x");

    vocab.unknown_token = null;
    const tokenizer = Tokenizer.init(vocab);

    const tokens = try tokenizer.encode(allocator, "x");
    defer allocator.free(tokens);

    try testing.expectEqual(@as(usize, 2), tokens.len);
    try testing.expectEqual(@as(TokenID, 14), tokens[0]);
    try testing.expectEqual(@as(TokenID, 15), tokens[1]);
}

test "encode empty string returns empty" {
    const allocator = testing.allocator;
    var vocab = try createTestVocabulary(allocator);
    defer cleanupTestVocabulary(allocator, &vocab);

    const tokenizer = Tokenizer.init(vocab);

    const tokens = try tokenizer.encode(allocator, "");
    defer allocator.free(tokens);

    try testing.expectEqual(@as(usize, 0), tokens.len);
}

test "ByteLevelTable roundtrip" {
    const allocator = testing.allocator;

    const input = "Hello, World! 123";
    const encoded = try ByteLevelTable.encode(allocator, input);
    defer allocator.free(encoded);

    try testing.expect(encoded.len >= input.len);
}

test "sentencePiecePreprocess adds meta space" {
    const allocator = testing.allocator;

    const result = try sentencePiecePreprocess(allocator, "hello world");
    defer allocator.free(result);

    try testing.expect(std.mem.startsWith(u8, result, MetaSpace));
    try testing.expect(std.mem.indexOf(u8, result[MetaSpace.len..], MetaSpace) != null);
}

test "sentencePiecePreprocess handles already prefixed input" {
    const allocator = testing.allocator;

    const input = MetaSpace ++ "hello";
    const result = try sentencePiecePreprocess(allocator, input);
    defer allocator.free(result);

    try testing.expectEqualStrings(input, result);
}

test "decode handles MetaSpace as regular space" {
    const allocator = testing.allocator;

    var encoding: Vocabulary.EncodingVocabulary = .{};
    try encoding.put(allocator, MetaSpace ++ "hello", 0);
    defer encoding.deinit(allocator);

    var decoding: Vocabulary.DecodingVocabulary = .{};
    try decoding.put(allocator, 0, MetaSpace ++ "hello");
    defer decoding.deinit(allocator);

    const vocab = Vocabulary{
        .merge_index = .{},
        .encoding = encoding,
        .decoding = decoding,
        .use_byte_level = false,
        .eos_token_id = 0,
    };

    const tokenizer = Tokenizer.init(vocab);

    const tokens = [_]TokenID{0};
    const text = try tokenizer.decode(allocator, &tokens);
    defer allocator.free(text);

    try testing.expectEqualStrings(" hello", text);
}

test "normalize with prepend" {
    const allocator = testing.allocator;

    const normalizer = Vocabulary.Normalizer{ .prepend = "PREFIX:" };
    const result = try normalize(allocator, normalizer, "text");
    defer allocator.free(result);

    try testing.expectEqualStrings("PREFIX:text", result);
}

test "normalize with replace" {
    const allocator = testing.allocator;

    const normalizer = Vocabulary.Normalizer{ .replace = .{ .pattern = "old", .content = "new" } };
    const result = try normalize(allocator, normalizer, "old text old");
    defer allocator.free(result);

    try testing.expectEqualStrings("new text new", result);
}

test "postProcess with template adds special tokens" {
    const allocator = testing.allocator;

    const template = [_]Vocabulary.PostProcessor.TemplateProcessing{
        .{ .special_token = 100 },
        .{ .sequence = {} },
        .{ .special_token = 101 },
    };
    const post_processor = Vocabulary.PostProcessor{ .template = &template };

    const input = [_]TokenID{ 1, 2, 3 };
    const result = try postProcess(allocator, post_processor, &input);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 5), result.len);
    try testing.expectEqual(@as(TokenID, 100), result[0]);
    try testing.expectEqual(@as(TokenID, 1), result[1]);
    try testing.expectEqual(@as(TokenID, 2), result[2]);
    try testing.expectEqual(@as(TokenID, 3), result[3]);
    try testing.expectEqual(@as(TokenID, 101), result[4]);
}

test "vocabulary works without merge_pairs field" {
    const allocator = testing.allocator;

    var encoding: Vocabulary.EncodingVocabulary = .{};
    try encoding.put(allocator, "a", 0);
    try encoding.put(allocator, "b", 1);
    try encoding.put(allocator, "ab", 2);
    defer encoding.deinit(allocator);

    var decoding: Vocabulary.DecodingVocabulary = .{};
    try decoding.put(allocator, 0, "a");
    try decoding.put(allocator, 1, "b");
    try decoding.put(allocator, 2, "ab");
    defer decoding.deinit(allocator);

    var merge_index: Vocabulary.MergePairIndex = .{};
    try merge_index.put(allocator, "a\x00b", 0);
    defer merge_index.deinit(allocator);

    const vocab = Vocabulary{
        .merge_index = merge_index,
        .encoding = encoding,
        .decoding = decoding,
        .use_byte_level = true,
        .eos_token_id = 0,
    };

    const tokenizer = Tokenizer.init(vocab);

    const tokens = try tokenizer.encode(allocator, "ab");
    defer allocator.free(tokens);

    try testing.expectEqual(@as(usize, 1), tokens.len);
    try testing.expectEqual(@as(TokenID, 2), tokens[0]);
}
