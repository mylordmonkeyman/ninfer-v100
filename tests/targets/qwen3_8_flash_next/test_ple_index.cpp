#include "targets/qwen3_8_flash_next/impl/ple_index.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr PleIndexMetadata metadata{
        .multipliers      = {23'703'573'157'769LL, 20'109'073'645'365LL, 8'052'911'324'071LL},
        .head_offsets     = {0, 20'000'003, 40'000'026, 60'000'059, 80'000'106, 100'000'165,
                             120'000'228, 140'000'297, 160'000'374, 180'000'455, 200'000'548,
                             220'000'655, 240'000'802, 260'000'955, 280'001'114, 300'001'275},
        .head_vocab_sizes = {20'000'003, 20'000'023, 20'000'033, 20'000'047, 20'000'059, 20'000'063,
                             20'000'069, 20'000'077, 20'000'081, 20'000'093, 20'000'107, 20'000'147,
                             20'000'153, 20'000'159, 20'000'161, 20'000'171},
    };
    constexpr std::array tokens   = {42LL, 17LL, 248'044LL, 99LL};
    constexpr std::array expected = {
        std::array<std::int64_t, 16>{4'064'167, 21'428'206, 43'851'727, 63'435'096, 89'826'284,
                                     106'088'217, 121'229'381, 156'147'803, 175'176'707,
                                     188'463'481, 216'828'809, 228'933'352, 248'448'227,
                                     269'188'956, 283'041'664, 314'347'972},
        std::array<std::int64_t, 16>{16'682'585, 32'401'650, 50'261'714, 71'266'385, 80'699'480,
                                     103'844'016, 138'560'934, 144'850'220, 176'215'859,
                                     184'859'497, 203'428'549, 221'557'213, 249'655'255,
                                     278'112'931, 287'678'741, 316'107'010},
        std::array<std::int64_t, 16>{5'560'156, 23'348'898, 56'653'202, 64'218'243, 81'003'143,
                                     114'205'635, 124'891'229, 154'118'427, 174'409'918,
                                     184'121'467, 210'801'992, 230'209'321, 256'177'394,
                                     263'203'618, 299'114'471, 300'431'127},
        std::array<std::int64_t, 16>{1'074'634, 33'421'378, 53'334'060, 73'399'811, 88'773'591,
                                     108'029'273, 137'660'660, 158'565'115, 177'669'505,
                                     198'344'556, 205'330'745, 222'074'671, 255'286'581,
                                     269'724'462, 294'809'720, 302'278'574},
    };

    PleTokenHistory history;
    if (history.previous_token() != 248'044 || history.second_previous_token() != 248'044) {
        std::cerr << "Default PleTokenHistory not seeded with kPleBoundaryTokenId\n";
        return 1;
    }

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto actual = ple_indices(metadata, history, tokens[index]);
        if (actual != expected[index]) {
            std::cerr << "PLE index mismatch at token " << index << '\n';
            return 1;
        }
        history.commit(tokens[index]);
    }
    if (history.previous_token() != 99 || history.second_previous_token() != 248'044) {
        std::cerr << "PLE EOS-segment history was not reset\n";
        return 1;
    }

    // Test token 248046 is ordinary advance (not boundary reset)
    history.commit(248'046);
    if (history.previous_token() != 248'046 || history.second_previous_token() != 99) {
        std::cerr << "Token 248046 did not advance history normally\n";
        return 1;
    }

    // Test token 248044 resets boundary
    history.commit(248'044);
    if (history.previous_token() != 248'044 || history.second_previous_token() != 248'044) {
        std::cerr << "Token 248044 did not reset boundary history\n";
        return 1;
    }

    // Test chunk vs sequential PLE indices parity over a chunk of arbitrary tokens
    const std::vector<std::int32_t> test_tokens = {100, 200, 300, 400, 248'044, 500, 600, 248'046, 700};
    PleTokenHistory seq_hist;
    std::vector<std::array<std::int64_t, 16>> seq_indices;
    for (std::int32_t tok : test_tokens) {
        seq_indices.push_back(ple_indices(metadata, seq_hist, tok));
        seq_hist.commit(tok);
    }

    PleTokenHistory chunk_hist;
    std::vector<std::array<std::int64_t, 16>> chunk_indices(test_tokens.size());
    for (std::size_t t = 0; t < test_tokens.size(); ++t) {
        chunk_indices[t] = ple_indices(metadata, chunk_hist, test_tokens[t]);
        chunk_hist.commit(test_tokens[t]);
    }

    for (std::size_t t = 0; t < test_tokens.size(); ++t) {
        if (chunk_indices[t] != seq_indices[t]) {
            std::cerr << "Chunk vs sequential PLE index mismatch at t=" << t << '\n';
            return 1;
        }
    }

    return 0;
}
