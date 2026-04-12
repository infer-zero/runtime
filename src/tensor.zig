const Tensor = @This();

data: []const u8,
data_type: DataType,

pub fn toF32(self: @This(), allocator: std.mem.Allocator) ![]const f32 {
    return self.data_type.toF32(allocator, self.data);
}

pub fn toF16(self: @This(), allocator: std.mem.Allocator) ![]const f16 {
    return self.data_type.toF16(allocator, self.data);
}

pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
    allocator.free(self.data);
}

pub const DataType = enum(u8) {
    BF16 = 0,
    FP32 = 1,
    FP16 = 2,
    Q8_0 = 3,
    Q4_0 = 4,
    Q6_K = 5,
    Q4_1 = 6,
    Q5_0 = 7,
    Q4_K = 8,
    Q5_K = 9,
    _,

    pub fn fromString(dtype_str: []const u8) @This() {
        return std.meta.stringToEnum(@This(), dtype_str) orelse .BF16;
    }

    pub fn byteSize(self: @This(), num_elements: usize) error{Overflow}!usize {
        return switch (self) {
            .BF16, .FP16 => std.math.mul(usize, num_elements, 2),
            .FP32 => std.math.mul(usize, num_elements, 4),
            .Q8_0 => std.math.mul(usize, num_elements / Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES),
            .Q4_0 => std.math.mul(usize, num_elements / Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES),
            .Q4_1 => std.math.mul(usize, num_elements / Q4_1_BLOCK_SIZE, Q4_1_BLOCK_BYTES),
            .Q5_0 => std.math.mul(usize, num_elements / Q5_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES),
            .Q5_K => std.math.mul(usize, num_elements / Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES),
            .Q6_K => std.math.mul(usize, num_elements / Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES),
            .Q4_K => std.math.mul(usize, num_elements / Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES),
            else => unreachable,
        };
    }

    pub fn numElements(self: @This(), num_bytes: usize) usize {
        return switch (self) {
            .BF16, .FP16 => num_bytes / 2,
            .FP32 => num_bytes / 4,
            .Q8_0 => (num_bytes / Q8_0_BLOCK_BYTES) * Q8_0_BLOCK_SIZE,
            .Q4_0 => (num_bytes / Q4_0_BLOCK_BYTES) * Q4_0_BLOCK_SIZE,
            .Q4_1 => (num_bytes / Q4_1_BLOCK_BYTES) * Q4_1_BLOCK_SIZE,
            .Q5_0 => (num_bytes / Q5_0_BLOCK_BYTES) * Q5_0_BLOCK_SIZE,
            .Q5_K => (num_bytes / Q5_K_BLOCK_BYTES) * Q5_K_BLOCK_SIZE,
            .Q6_K => (num_bytes / Q6_K_BLOCK_BYTES) * Q6_K_BLOCK_SIZE,
            .Q4_K => (num_bytes / Q4_K_BLOCK_BYTES) * Q4_K_BLOCK_SIZE,
            else => unreachable,
        };
    }

    pub fn toF32(self: @This(), allocator: std.mem.Allocator, data: []const u8) ![]const f32 {
        switch (self) {
            .BF16 => {
                if (data.len % 2 != 0) return error.InvalidData;
                const bf16_data: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, data));
                const result = try allocator.alloc(f32, bf16_data.len);
                for (bf16_data, 0..) |bf16, idx| {
                    const bits: u32 = @as(u32, bf16) << 16;
                    result[idx] = @bitCast(bits);
                }
                return result;
            },
            .FP32 => {
                if (data.len % 4 != 0) return error.InvalidData;
                const f32_data: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, data));
                return try allocator.dupe(f32, f32_data);
            },
            .FP16 => {
                if (data.len % 2 != 0) return error.InvalidData;
                const f16_data: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, data));
                const result = try allocator.alloc(f32, f16_data.len);
                for (f16_data, 0..) |f16_val, idx| {
                    result[idx] = @floatCast(f16_val);
                }
                return result;
            },
            .Q8_0 => {
                if (data.len % Q8_0_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q8_0_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q8_0_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q8_0_BLOCK_BYTES ..][0..Q8_0_BLOCK_BYTES];
                    const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                    for (0..Q8_0_BLOCK_SIZE) |elem| {
                        result[block_idx * Q8_0_BLOCK_SIZE + elem] = scale * @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + elem]))));
                    }
                }
                return result;
            },
            .Q4_0 => {
                if (data.len % Q4_0_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q4_0_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q4_0_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q4_0_BLOCK_BYTES ..][0..Q4_0_BLOCK_BYTES];
                    const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                    for (0..Q4_0_BLOCK_SIZE / 2) |j| {
                        const byte = block[2 + j];
                        const low: i8 = @as(i8, @intCast(byte & 0x0F)) - 8;
                        const high: i8 = @as(i8, @intCast(byte >> 4)) - 8;
                        result[block_idx * Q4_0_BLOCK_SIZE + j] = scale * @as(f32, @floatFromInt(low));
                        result[block_idx * Q4_0_BLOCK_SIZE + j + Q4_0_BLOCK_SIZE / 2] = scale * @as(f32, @floatFromInt(high));
                    }
                }
                return result;
            },
            .Q4_1 => {
                if (data.len % Q4_1_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q4_1_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q4_1_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q4_1_BLOCK_BYTES ..][0..Q4_1_BLOCK_BYTES];
                    const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                    const min: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[2..4], .little))));
                    for (0..Q4_1_BLOCK_SIZE / 2) |j| {
                        const byte = block[4 + j];
                        const low: f32 = @floatFromInt(@as(u8, byte & 0x0F));
                        const high: f32 = @floatFromInt(@as(u8, byte >> 4));
                        result[block_idx * Q4_1_BLOCK_SIZE + j] = low * scale + min;
                        result[block_idx * Q4_1_BLOCK_SIZE + j + Q4_1_BLOCK_SIZE / 2] = high * scale + min;
                    }
                }
                return result;
            },
            .Q5_0 => {
                // Q5_0: 32 elements per block, 22 bytes per block
                // Layout: d[2]:f16 + qh[4]:uint32 + qs[16]:packed_nibbles
                if (data.len % Q5_0_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q5_0_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q5_0_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q5_0_BLOCK_BYTES ..][0..Q5_0_BLOCK_BYTES];
                    const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                    const qh: u32 = std.mem.readInt(u32, block[2..6], .little);
                    for (0..Q5_0_BLOCK_SIZE / 2) |j| {
                        const byte = block[6 + j];
                        const low5: i8 = @as(i8, @intCast((byte & 0x0F) | (if ((qh >> @intCast(j)) & 1 != 0) @as(u8, 0x10) else 0))) - 16;
                        const high5: i8 = @as(i8, @intCast((byte >> 4) | (if ((qh >> @intCast(j + 16)) & 1 != 0) @as(u8, 0x10) else 0))) - 16;
                        result[block_idx * Q5_0_BLOCK_SIZE + j] = scale * @as(f32, @floatFromInt(low5));
                        result[block_idx * Q5_0_BLOCK_SIZE + j + Q5_0_BLOCK_SIZE / 2] = scale * @as(f32, @floatFromInt(high5));
                    }
                }
                return result;
            },
            .Q6_K => {
                // Q6_K: 256 elements per block, 210 bytes per block
                // Layout: ql[128] + qh[64] + scales[16] + d[2]
                // Matches GGML dequantize_row_q6_K reference implementation.
                if (data.len % Q6_K_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q6_K_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q6_K_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q6_K_BLOCK_BYTES ..][0..Q6_K_BLOCK_BYTES];
                    const ql_base = block[0..128];
                    const qh_base = block[128..192];
                    const sc_base: *const [16]i8 = @ptrCast(block[192..208]);
                    const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[208..210], .little))));
                    const out = result[block_idx * Q6_K_BLOCK_SIZE ..][0..Q6_K_BLOCK_SIZE];

                    // Process 2 groups of 128 elements
                    inline for (0..2) |group| {
                        const ql = ql_base[group * 64 ..];
                        const qh = qh_base[group * 32 ..];
                        const y_off = group * 128;
                        const sc_off = group * 8;

                        for (0..32) |l| {
                            const is: usize = l / 16;
                            const q1: i32 = @as(i32, (ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                            const q2: i32 = @as(i32, (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                            const q3: i32 = @as(i32, (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                            const q4: i32 = @as(i32, (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                            out[y_off + l + 0] = d * @as(f32, @floatFromInt(sc_base[sc_off + is + 0])) * @as(f32, @floatFromInt(q1));
                            out[y_off + l + 32] = d * @as(f32, @floatFromInt(sc_base[sc_off + is + 2])) * @as(f32, @floatFromInt(q2));
                            out[y_off + l + 64] = d * @as(f32, @floatFromInt(sc_base[sc_off + is + 4])) * @as(f32, @floatFromInt(q3));
                            out[y_off + l + 96] = d * @as(f32, @floatFromInt(sc_base[sc_off + is + 6])) * @as(f32, @floatFromInt(q4));
                        }
                    }
                }
                return result;
            },
            .Q4_K => {
                // Q4_K: 256 elements per block, 144 bytes per block
                // Layout: d[2] + dmin[2] + scales[12] + qs[128]
                // Matches GGML dequantize_row_q4_K reference implementation.
                if (data.len % Q4_K_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q4_K_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q4_K_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q4_K_BLOCK_BYTES ..][0..Q4_K_BLOCK_BYTES];
                    const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                    const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[2..4], .little))));
                    const scales_raw = block[4..16];
                    const qs = block[16..144];
                    const out = result[block_idx * Q4_K_BLOCK_SIZE ..][0..Q4_K_BLOCK_SIZE];

                    // Unpack 6-bit scales and mins from 12-byte packed array.
                    // Layout: scales_raw[0..3] hold lower 6 bits of scales 0-3,
                    // scales_raw[4..7] hold lower 6 bits of mins 0-3,
                    // scales_raw[8..11] hold upper 2 bits of scales 4-7 and mins 4-7.
                    var scales_arr: [8]u8 = undefined;
                    var mins_arr: [8]u8 = undefined;

                    scales_arr[0] = scales_raw[0] & 0x3F;
                    scales_arr[1] = scales_raw[1] & 0x3F;
                    scales_arr[2] = scales_raw[2] & 0x3F;
                    scales_arr[3] = scales_raw[3] & 0x3F;
                    mins_arr[0] = scales_raw[4] & 0x3F;
                    mins_arr[1] = scales_raw[5] & 0x3F;
                    mins_arr[2] = scales_raw[6] & 0x3F;
                    mins_arr[3] = scales_raw[7] & 0x3F;
                    scales_arr[4] = (scales_raw[8] & 0x0F) | ((scales_raw[0] >> 6) << 4);
                    scales_arr[5] = (scales_raw[9] & 0x0F) | ((scales_raw[1] >> 6) << 4);
                    scales_arr[6] = (scales_raw[10] & 0x0F) | ((scales_raw[2] >> 6) << 4);
                    scales_arr[7] = (scales_raw[11] & 0x0F) | ((scales_raw[3] >> 6) << 4);
                    mins_arr[4] = (scales_raw[8] >> 4) | ((scales_raw[4] >> 6) << 4);
                    mins_arr[5] = (scales_raw[9] >> 4) | ((scales_raw[5] >> 6) << 4);
                    mins_arr[6] = (scales_raw[10] >> 4) | ((scales_raw[6] >> 6) << 4);
                    mins_arr[7] = (scales_raw[11] >> 4) | ((scales_raw[7] >> 6) << 4);

                    // Dequantize 4 groups of 64 elements (32 bytes each).
                    // Each group uses 2 sub-block scales: low nibbles get scale[2j],
                    // high nibbles get scale[2j+1]. Matches GGML dequantize_row_q4_K.
                    for (0..4) |j| {
                        const sc1: f32 = d * @as(f32, @floatFromInt(scales_arr[j * 2]));
                        const m1: f32 = dmin * @as(f32, @floatFromInt(mins_arr[j * 2]));
                        const sc2: f32 = d * @as(f32, @floatFromInt(scales_arr[j * 2 + 1]));
                        const m2: f32 = dmin * @as(f32, @floatFromInt(mins_arr[j * 2 + 1]));
                        const q_base = qs[j * 32 ..];
                        const out_base = out[j * 64 ..];
                        for (0..32) |l| {
                            out_base[l] = sc1 * @as(f32, @floatFromInt(q_base[l] & 0x0F)) - m1;
                            out_base[l + 32] = sc2 * @as(f32, @floatFromInt(q_base[l] >> 4)) - m2;
                        }
                    }
                }
                return result;
            },
            .Q5_K => {
                // Q5_K: 256 elements per block, 176 bytes per block
                // Layout: d[2] + dmin[2] + scales[12] + qh[32] + qs[128]
                // Same scale/min unpacking as Q4_K, but 5-bit quants (4-bit in qs + 1 high bit in qh).
                if (data.len % Q5_K_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q5_K_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q5_K_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q5_K_BLOCK_BYTES ..][0..Q5_K_BLOCK_BYTES];
                    const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                    const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[2..4], .little))));
                    const scales_raw = block[4..16];
                    const qh = block[16..48];
                    const qs = block[48..176];
                    const out = result[block_idx * Q5_K_BLOCK_SIZE ..][0..Q5_K_BLOCK_SIZE];

                    var scales_arr: [8]u8 = undefined;
                    var mins_arr: [8]u8 = undefined;
                    scales_arr[0] = scales_raw[0] & 0x3F;
                    scales_arr[1] = scales_raw[1] & 0x3F;
                    scales_arr[2] = scales_raw[2] & 0x3F;
                    scales_arr[3] = scales_raw[3] & 0x3F;
                    mins_arr[0] = scales_raw[4] & 0x3F;
                    mins_arr[1] = scales_raw[5] & 0x3F;
                    mins_arr[2] = scales_raw[6] & 0x3F;
                    mins_arr[3] = scales_raw[7] & 0x3F;
                    scales_arr[4] = (scales_raw[8] & 0x0F) | ((scales_raw[0] >> 6) << 4);
                    scales_arr[5] = (scales_raw[9] & 0x0F) | ((scales_raw[1] >> 6) << 4);
                    scales_arr[6] = (scales_raw[10] & 0x0F) | ((scales_raw[2] >> 6) << 4);
                    scales_arr[7] = (scales_raw[11] & 0x0F) | ((scales_raw[3] >> 6) << 4);
                    mins_arr[4] = (scales_raw[8] >> 4) | ((scales_raw[4] >> 6) << 4);
                    mins_arr[5] = (scales_raw[9] >> 4) | ((scales_raw[5] >> 6) << 4);
                    mins_arr[6] = (scales_raw[10] >> 4) | ((scales_raw[6] >> 6) << 4);
                    mins_arr[7] = (scales_raw[11] >> 4) | ((scales_raw[7] >> 6) << 4);

                    for (0..4) |j| {
                        // Q5_K scale indexing: j for low nibble, j+4 for high nibble
                        // (differs from Q4_K which uses j*2 and j*2+1)
                        const sc1: f32 = d * @as(f32, @floatFromInt(scales_arr[j]));
                        const m1: f32 = dmin * @as(f32, @floatFromInt(mins_arr[j]));
                        const sc2: f32 = d * @as(f32, @floatFromInt(scales_arr[j + 4]));
                        const m2: f32 = dmin * @as(f32, @floatFromInt(mins_arr[j + 4]));
                        const q_base = qs[j * 32 ..];
                        const out_base = out[j * 64 ..];
                        for (0..32) |l| {
                            // Q5_K qh: qh[l] bit j for low nibble, bit j+4 for high nibble
                            const qh_lo: u5 = @intCast((qh[l] >> @intCast(j)) & 1);
                            const qh_hi: u5 = @intCast((qh[l] >> @intCast(j + 4)) & 1);
                            const lo5 = (q_base[l] & 0x0F) | (qh_lo << 4);
                            const hi5 = (q_base[l] >> 4) | (qh_hi << 4);
                            out_base[l] = sc1 * @as(f32, @floatFromInt(lo5)) - m1;
                            out_base[l + 32] = sc2 * @as(f32, @floatFromInt(hi5)) - m2;
                        }
                    }
                }
                return result;
            },
            else => {
                log.err("unsupported data type for F32 conversion: {d}", .{@intFromEnum(self)});
                return error.UnsupportedDataType;
            },
        }
    }

    pub fn toF16(self: @This(), allocator: std.mem.Allocator, data: []const u8) ![]const f16 {
        switch (self) {
            .BF16 => {
                if (data.len % 2 != 0) return error.InvalidData;
                const bf16_data: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, data));
                const result = try allocator.alloc(f16, bf16_data.len);
                for (bf16_data, 0..) |bf16, idx| {
                    const bits: u32 = @as(u32, bf16) << 16;
                    const f32_val: f32 = @bitCast(bits);
                    result[idx] = @floatCast(f32_val);
                }
                return result;
            },
            .FP32 => {
                if (data.len % 4 != 0) return error.InvalidData;
                const f32_data: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, data));
                const result = try allocator.alloc(f16, f32_data.len);
                for (f32_data, 0..) |f32_val, idx| {
                    result[idx] = @floatCast(f32_val);
                }
                return result;
            },
            .FP16 => {
                if (data.len % 2 != 0) return error.InvalidData;
                const f16_data: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, data));
                return try allocator.dupe(f16, f16_data);
            },
            .Q8_0, .Q4_0, .Q4_1, .Q5_0, .Q6_K, .Q4_K, .Q5_K => {
                const f32_data = try self.toF32(allocator, data);
                defer allocator.free(f32_data);
                const result = try allocator.alloc(f16, f32_data.len);
                for (f32_data, 0..) |f32_val, idx| {
                    result[idx] = @floatCast(f32_val);
                }
                return result;
            },
            else => {
                log.err("unsupported data type for F16 conversion: {d}", .{@intFromEnum(self)});
                return error.UnsupportedDataType;
            },
        }
    }
};

const Q8_0_BLOCK_SIZE = 32;
const Q8_0_BLOCK_BYTES = 2 + Q8_0_BLOCK_SIZE; // f16 scale + 32 i8 quants

const Q4_0_BLOCK_SIZE = 32;
const Q4_0_BLOCK_BYTES = 2 + Q4_0_BLOCK_SIZE / 2; // f16 scale + 16 packed bytes (2 nibbles each)

const Q4_1_BLOCK_SIZE = 32;
const Q4_1_BLOCK_BYTES = 2 + 2 + Q4_1_BLOCK_SIZE / 2; // f16 scale + f16 min + 16 packed bytes (2 nibbles each)

const Q5_0_BLOCK_SIZE = 32;
const Q5_0_BLOCK_BYTES = 2 + 4 + Q5_0_BLOCK_SIZE / 2; // f16 scale + uint32 high_bits + 16 packed nibbles

const Q5_K_BLOCK_SIZE = 256;
const Q5_K_BLOCK_BYTES = 2 + 2 + 12 + 32 + 128; // d[2] + dmin[2] + scales[12] + qh[32] + qs[128] = 176

const Q6_K_BLOCK_SIZE = 256;
const Q6_K_BLOCK_BYTES = 128 + 64 + 16 + 2; // ql[128] + qh[64] + scales[16] + d[2] = 210

const Q4_K_BLOCK_SIZE = 256;
const Q4_K_BLOCK_BYTES = 2 + 2 + 12 + 128; // d[2] + dmin[2] + scales[12] + qs[128] = 144

const log = std.log.scoped(.infer);

const std = @import("std");
