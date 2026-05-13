#include "gelu_omp.h"

#include <chrono>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <omp.h>
#include <immintrin.h>

static std::vector<float> GeluOMPRef(const std::vector<float>& input) {
    std::vector<float> output;
    output.reserve(input.size());
    for (const auto& val : input) {
        output.push_back(val * 0.5f * (1.0f + tanh(sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val))));
    }
    return output;
}

static __m256 exp256_ps(__m256 in) {
    float data[8];
    _mm256_storeu_ps(data, in);
    return _mm256_setr_ps(
        expf(data[0]),
        expf(data[1]),
        expf(data[2]),
        expf(data[3]),
        expf(data[4]),
        expf(data[5]),
        expf(data[6]),
        expf(data[7])
    );
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t inputSize = input.size();
    std::vector<float> output(inputSize);
    const float* inData = input.data();
    float* outData = output.data();

    constexpr float sqrtRes = sqrt(2.0f / M_PI);
    const size_t vectorizedSize = inputSize - inputSize % 8;

    const __m256 vecSqrt = _mm256_set1_ps(sqrtRes);
    const __m256 vecK = _mm256_set1_ps(0.044715f);
    const __m256 vecOne = _mm256_set1_ps(1.0f);
    const __m256 vecTwo = _mm256_set1_ps(2.0f);
    const __m256 vecHalf = _mm256_set1_ps(0.5f);

    #pragma omp parallel for
    for (int i = 0; i < vectorizedSize; i += 8) {
        __m256 val = _mm256_loadu_ps(inData + i);
        __m256 val3 = _mm256_mul_ps(_mm256_mul_ps(val, val), val);
        __m256 k = _mm256_fmadd_ps(vecK, val3, val);
        __m256 tanhArg = _mm256_mul_ps(vecSqrt, k);
        __m256 expRes = exp256_ps(_mm256_mul_ps(tanhArg, vecTwo));
        __m256 tanhRes = _mm256_sub_ps(vecOne, _mm256_div_ps(vecTwo, _mm256_add_ps(expRes, vecOne)));
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(val, vecHalf), _mm256_add_ps(vecOne, tanhRes));
        _mm256_storeu_ps(outData + i, res);
    }

    #pragma omp parallel for
    for (int i = vectorizedSize; i < inputSize; ++i) {
        const float val = inData[i];

        const float tanhArg = sqrtRes * (val + 0.044715f * val * val * val);
        const float tanhRes = 1 - 2.0f / (std::exp(2 * tanhArg) + 1);

        outData[i] = val * 0.5f * (1.0f + tanhRes);
    }

    return output;
}

int main() {
    constexpr size_t dataSize = 13687989;
    constexpr float minVal = 0.0f;
    constexpr float maxVal = 20.0f;

    std::vector<float> input(dataSize);
    std::generate(input.begin(), input.end(), [](){
        return minVal + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (maxVal - minVal);
    });

    auto outputRef = GeluOMPRef(input);
    auto output = GeluOMP(input);
    float error = 0.0f;
    for (size_t i = 0; i < dataSize; ++i) {
        error = std::max(std::abs(output[i] - outputRef[i]), error);
    }
    std::cout << "Absolute max error: " << error << std::endl;

    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        GeluOMP(input);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    std::cout << "Time: " << time << " seconds" << std::endl;
}