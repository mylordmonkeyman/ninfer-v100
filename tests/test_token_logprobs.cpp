#include "runtime/contract/token_logprobs.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::TokenLogprobOptions;
using ninfer::TokenLogprobs;
using ninfer::runtime::compute_token_logprobs;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

// Exactly representable BF16 values only: BF16 keeps the binary32 exponent and the top 7
// mantissa bits, so small integers and halves round-trip.
std::uint16_t bf16(float value) {
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16U);
}

std::vector<std::uint16_t> column(const std::vector<float>& values) {
    std::vector<std::uint16_t> out;
    out.reserve(values.size());
    for (const float value : values) { out.push_back(bf16(value)); }
    return out;
}

bool close(float actual, double expected) { return std::fabs(actual - expected) < 1e-5; }

double log_softmax(const std::vector<float>& values, const std::vector<int>& support, int index) {
    double sum = 0.0;
    for (const int token : support) { sum += std::exp(static_cast<double>(values[token])); }
    return static_cast<double>(values[index]) - std::log(sum);
}

void test_bf16_round_trip() {
    for (const float value : {0.0F, 1.0F, -1.0F, 2.5F, -17.0F, 96.0F, 0.015625F}) {
        require(ninfer::runtime::bf16_to_float(bf16(value)) == value,
                "BF16 conversion changed an exactly representable value");
    }
    require(std::isinf(ninfer::runtime::bf16_to_float(bf16(-INFINITY))),
            "BF16 conversion lost negative infinity");
}

void test_unconstrained_distribution() {
    const std::vector<float> values{1.0F, 3.0F, -2.0F, 3.0F, 0.5F, -8.0F};
    const auto logits = column(values);
    const std::vector<int> everything{0, 1, 2, 3, 4, 5};
    const TokenLogprobs out =
        compute_token_logprobs(logits, {}, 4, TokenLogprobOptions{.enabled = true, .top = 3});

    require(out.sampled.token == 4 && close(out.sampled.raw_logprob, log_softmax(values, everything, 4)),
            "sampled raw logprob is not the log softmax");
    require(out.sampled.logprob == out.sampled.raw_logprob,
            "unconstrained logprob differs from the raw logprob");
    // Tokens 1 and 3 tie at the top: the lower id comes first.
    require(out.top.size() == 3 && out.top[0].token == 1 && out.top[1].token == 3 &&
                out.top[2].token == 0,
            "top alternatives are not ordered by logit then token id");
    double total = 0.0;
    for (const int token : everything) {
        total += std::exp(static_cast<double>(
            compute_token_logprobs(logits, {}, token, TokenLogprobOptions{.enabled = true})
                .sampled.raw_logprob));
    }
    require(std::fabs(total - 1.0) < 1e-5, "raw probabilities do not sum to one");
    require(out.candidates.empty(), "candidates were reported without a candidate list");
}

void test_masked_distribution_renormalises_over_allowed_tokens() {
    // The model prefers token 1, but only tokens 2 and 4 are allowed.
    const std::vector<float> values{1.0F, 9.0F, 2.0F, 0.0F, 3.0F, 8.0F};
    const auto logits = column(values);
    const std::vector<std::int32_t> mask{(1 << 2) | (1 << 4)};
    const TokenLogprobs out =
        compute_token_logprobs(logits, mask, 4, TokenLogprobOptions{.enabled = true, .top = 5});

    require(out.top.size() == 2 && out.top[0].token == 4 && out.top[1].token == 2,
            "masked top alternatives include a forbidden token");
    require(close(out.top[0].logprob, log_softmax(values, {2, 4}, 4)) &&
                close(out.top[1].logprob, log_softmax(values, {2, 4}, 2)),
            "masked logprob is not renormalised over the allowed tokens");
    require(std::fabs(std::exp(out.top[0].logprob) + std::exp(out.top[1].logprob) - 1.0) < 1e-5,
            "allowed probabilities do not sum to one");
    require(close(out.sampled.raw_logprob, log_softmax(values, {0, 1, 2, 3, 4, 5}, 4)),
            "raw logprob was affected by the mask");
}

void test_candidate_list_is_a_full_distribution() {
    const std::vector<float> values{1.0F, 9.0F, 2.0F, 0.0F, 3.0F, 8.0F};
    const auto logits = column(values);
    TokenLogprobOptions options{.enabled = true, .top = 1, .candidates = {5, 0, 3}};
    TokenLogprobs out = compute_token_logprobs(logits, {}, 1, options);

    require(out.candidates.size() == 3 && out.candidates[0].token == 5 &&
                out.candidates[1].token == 0 && out.candidates[2].token == 3,
            "candidates are not reported in request order");
    double total = 0.0;
    for (std::size_t index = 0; index < 3; ++index) {
        require(close(out.candidates[index].logprob,
                      log_softmax(values, {5, 0, 3}, options.candidates[index])),
                "candidate logprob is not renormalised over the candidate list");
        total += std::exp(out.candidates[index].logprob);
    }
    require(std::fabs(total - 1.0) < 1e-5, "candidate probabilities do not sum to one");
    require(close(out.candidates[0].raw_logprob, log_softmax(values, {0, 1, 2, 3, 4, 5}, 5)),
            "candidate raw logprob is not the vocabulary log softmax");

    // A candidate the mask forbids keeps its slot with probability zero.
    const std::vector<std::int32_t> mask{(1 << 5) | (1 << 3)};
    out = compute_token_logprobs(logits, mask, 5, options);
    require(std::isinf(out.candidates[1].logprob) && out.candidates[1].logprob < 0.0F,
            "forbidden candidate did not get probability zero");
    require(close(out.candidates[0].logprob, log_softmax(values, {5, 3}, 5)),
            "candidates were not renormalised over the allowed candidates");
}

void test_invalid_inputs_are_rejected() {
    const auto logits = column({0.0F, 1.0F});
    bool threw        = false;
    try {
        (void)compute_token_logprobs(logits, {}, 2, TokenLogprobOptions{.enabled = true});
    } catch (const std::logic_error&) { threw = true; }
    require(threw, "out-of-range sampled token was accepted");
    threw = false;
    try {
        (void)compute_token_logprobs(logits, {}, 0,
                                     TokenLogprobOptions{.enabled = true, .candidates = {7}});
    } catch (const std::logic_error&) { threw = true; }
    require(threw, "out-of-range candidate was accepted");
}

void run_test(const char* name, void (*test)()) {
    try {
        test();
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        std::exit(1);
    }
}

} // namespace

int main() {
    run_test("bf16 round trip", test_bf16_round_trip);
    run_test("unconstrained distribution", test_unconstrained_distribution);
    run_test("masked distribution", test_masked_distribution_renormalises_over_allowed_tokens);
    run_test("candidate list", test_candidate_list_is_a_full_distribution);
    run_test("invalid inputs", test_invalid_inputs_are_rejected);
    std::cout << "ok\n";
    return 0;
}
