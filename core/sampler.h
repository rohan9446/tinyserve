#ifndef TINYSERVE_SAMPLER_H
#define TINYSERVE_SAMPLER_H

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
    EXPORT int sample_token(
        float* logits,
        int vocab_size,
        float temperature,
        int top_k,
        float top_p,
        unsigned int seed
    );
}

#endif