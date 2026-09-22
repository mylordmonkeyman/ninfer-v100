#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::volta {

inline constexpr int kStateDim     = 128;
inline constexpr int kChunkSize    = 32;
inline constexpr int kWindowTokens = 1024;

inline constexpr int kQkHeads     = 16;
inline constexpr int kValueHeads  = 48;
inline constexpr int kGroupSize   = 3;

inline constexpr int kTile         = 8;
inline constexpr int kLowerTiles   = 10;
inline constexpr int kTileElements = 64;

inline constexpr std::size_t kWorkspaceAlign = 256;

struct TileCoord {
    int row;
    int col;
};

inline constexpr TileCoord kLowerTileCoords[kLowerTiles] = {
    {0, 0},
    {1, 0}, {1, 1},
    {2, 0}, {2, 1}, {2, 2},
    {3, 0}, {3, 1}, {3, 2}, {3, 3},
};

__host__ __device__ constexpr TileCoord lower_tile_coord(int tile) noexcept {
    if (tile < 0 || tile >= kLowerTiles) { return {0, 0}; }
    int row = 0;
    while ((row + 1) * (row + 2) / 2 <= tile) { ++row; }
    return {row, tile - row * (row + 1) / 2};
}

__host__ __device__ constexpr int lower_tile_index(int br, int bc) noexcept {
    return br * (br + 1) / 2 + bc;
}

__host__ __device__ constexpr int full_chunks(int tokens) noexcept {
    return tokens > 0 ? tokens / kChunkSize : 0;
}

__host__ __device__ constexpr int full_tokens(int tokens) noexcept {
    return full_chunks(tokens) * kChunkSize;
}

__host__ __device__ constexpr int workspace_tokens(int tokens) noexcept {
    const int full = full_tokens(tokens);
    return full < kWindowTokens ? full : kWindowTokens;
}

struct TensorRegion {
    std::size_t offset = 0;
    std::size_t bytes  = 0;
};

struct workspace_layout {
    TensorRegion q_inv_norm;
    TensorRegion k_inv_norm;
    TensorRegion kk_tiles;
    TensorRegion qk_tiles;

    int capacity_tokens = 0;
    int capacity_chunks = 0;

    std::size_t total_bytes = 0;
};

constexpr std::size_t align_workspace(std::size_t bytes) noexcept {
    return (bytes + (kWorkspaceAlign - 1)) & ~(kWorkspaceAlign - 1);
}

constexpr workspace_layout make_workspace_layout(int qk_heads, int tokens) noexcept {
    workspace_layout layout{};
    if (qk_heads <= 0) { return layout; }

    layout.capacity_tokens = workspace_tokens(tokens);
    layout.capacity_chunks = layout.capacity_tokens / kChunkSize;

    const std::size_t heads = static_cast<std::size_t>(qk_heads);
    const std::size_t tokens_capacity = static_cast<std::size_t>(layout.capacity_tokens);
    const std::size_t chunks_capacity = static_cast<std::size_t>(layout.capacity_chunks);

    layout.q_inv_norm.offset = 0;
    layout.q_inv_norm.bytes  = heads * tokens_capacity * sizeof(float);

    layout.k_inv_norm.offset = align_workspace(layout.q_inv_norm.offset + layout.q_inv_norm.bytes);
    layout.k_inv_norm.bytes  = heads * tokens_capacity * sizeof(float);

    const std::size_t lower_tile_bytes =
        chunks_capacity * heads * kLowerTiles * kTileElements * sizeof(float);

    layout.kk_tiles.offset = align_workspace(layout.k_inv_norm.offset + layout.k_inv_norm.bytes);
    layout.kk_tiles.bytes  = lower_tile_bytes;

    layout.qk_tiles.offset = align_workspace(layout.kk_tiles.offset + layout.kk_tiles.bytes);
    layout.qk_tiles.bytes  = lower_tile_bytes;

    layout.total_bytes = align_workspace(layout.qk_tiles.offset + layout.qk_tiles.bytes);
    return layout;
}

constexpr std::size_t workspace_bytes(int qk_heads, int tokens) noexcept {
    return make_workspace_layout(qk_heads, tokens).total_bytes;
}

static_assert(kValueHeads % kQkHeads == 0);
static_assert(kValueHeads / kQkHeads == kGroupSize);
static_assert(kStateDim == 128);
static_assert(kChunkSize == 32);
static_assert(kChunkSize == 4 * kTile);
static_assert(workspace_bytes(kQkHeads, kWindowTokens) == 2'752'512);

} // namespace ninfer::ops::detail::gated_delta_net::volta
