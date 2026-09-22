#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include <cstddef>

namespace ninfer::ops::detail::gated_delta_net::volta {

inline constexpr int kQkHalfLd    = 130;
inline constexpr int kKtHalfLd    = 34;
inline constexpr int kFp32TileLd  = 33;
inline constexpr int kMacroFp32Ld = 17;

constexpr std::size_t align128(std::size_t n) noexcept {
    return (n + 127) & ~std::size_t(127);
}

template <int DV_TILE>
struct FusedSharedLayout {
    static_assert(DV_TILE == 16 || DV_TILE == 32);

    static constexpr std::size_t A = 0;
    static constexpr std::size_t B = align128(A + 8704);
    static constexpr std::size_t C = align128(B + 8320);
    static constexpr std::size_t D = align128(C + (DV_TILE == 16 ? 4160 : 8320));
    static constexpr std::size_t E = align128(D + (DV_TILE == 16 ? 4352 : 8704));
    static constexpr std::size_t Bytes = align128(E + 1024);
};

struct GroupedDv16SharedLayout {
    static constexpr std::size_t Q = FusedSharedLayout<16>::Bytes;
    static constexpr std::size_t Bytes = align128(Q + 8320);
};

static_assert(FusedSharedLayout<16>::Bytes == 26624);
static_assert(FusedSharedLayout<32>::Bytes == 35072);
static_assert(GroupedDv16SharedLayout::Bytes == 34944);

} // namespace ninfer::ops::detail::gated_delta_net::volta
