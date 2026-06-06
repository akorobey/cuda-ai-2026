#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include "gelu_omp.h"

inline float TanByExp(float x) {
    float exp2x = std::exp(2.f * x);
    return (exp2x - 1.f) / (exp2x + 1.f);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t size = input.size();
    std::vector<float> output(size);

    const float* inptr = input.data();
    float* outptr = output.data();

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        float x = inptr[i];
        float inner = 0.79788456f * x * (1.f + 0.044715f * x * x);
        outptr[i] = 0.5f * x * (1.f + TanByExp(inner));
    }

    return output;
}
