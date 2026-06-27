pub const FileType = enum {
    gguf,
    safetensors,
};

pub const Backend = enum {
    cpu,
    vulkan,
};

pub const DataTypes = enum {
    BF16,
    FP32,
    FP16,
    Q8_0,
    Q6_K,
    Q5_K,
    Q5_0,
    Q4_0,
    Q4_1,
    Q4_K,
};

pub const KVTypes = enum {
    BF16,
    F32,
    F16,
    Q8,
    TurboQuant,
};

pub const Subarch = enum {
    dense,
    moe,
};

pub const Profile = enum {
    base,
    instruct,
    thinking,
};
