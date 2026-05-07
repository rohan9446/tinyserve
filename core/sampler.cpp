#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <random>

struct TokenProb {
    int index;
    float prob;
};

static bool compare_desc(const TokenProb& a, const TokenProb& b) {
    return a.prob > b.prob;
}

extern "C" EXPORT int sample_token(
    float* logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    unsigned int seed
) {
    // temperature scaling
    if (temperature > 0.0f && temperature != 1.0f) {
        for (int i = 0; i < vocab_size; i++) {
            logits[i] /= temperature;
        }
    }

    // softmax
    float max_logit = *std::max_element(logits, logits + vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = std::exp(logits[i] - max_logit);
        sum += logits[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= sum;
    }

    // build token-prob pairs
    std::vector<TokenProb> tokens(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        tokens[i] = {i, logits[i]};
    }
    std::sort(tokens.begin(), tokens.end(), compare_desc);

    // top-k filtering
    int k = (top_k > 0 && top_k < vocab_size) ? top_k : vocab_size;
    tokens.resize(k);

    // top-p (nucleus) filtering
    if (top_p > 0.0f && top_p < 1.0f) {
        float cumulative = 0.0f;
        int cutoff = 0;
        for (int i = 0; i < (int)tokens.size(); i++) {
            cumulative += tokens[i].prob;
            cutoff = i + 1;
            if (cumulative >= top_p) break;
        }
        tokens.resize(cutoff);
    }

    // renormalize
    float total = 0.0f;
    for (auto& t : tokens) total += t.prob;
    for (auto& t : tokens) t.prob /= total;

    // greedy if temperature is 0
    if (temperature == 0.0f) {
        return tokens[0].index;
    }

    // weighted random sample
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (auto& t : tokens) {
        cum += t.prob;
        if (r <= cum) return t.index;
    }

    return tokens.back().index;
}