#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}

void sqrtAVX2(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;
    static const float one = 1.f;
    __m256 kThresholdVec = _mm256_broadcast_ss(&kThreshold);
    __m256 oneVec = _mm256_broadcast_ss(&one);
    __m256 sign_bit = _mm256_set1_ps(-0.0f);

    for (int i=0; i<N; i+=8) {

        __m256 x = _mm256_load_ps(&values[i]);
        __m256 guess = _mm256_broadcast_ss(&initialGuess);

        // Calculate error
        __m256 error = _mm256_mul_ps(guess, guess);
        error = _mm256_mul_ps(error, x);
        error = _mm256_sub_ps(error, oneVec);

        // Abs error
        error = _mm256_andnot_ps(sign_bit, error);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}
