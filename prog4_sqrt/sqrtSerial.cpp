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

// sqrt implementation using avx2 intrinsics
void sqrtAVX2(int N,
                float initialGuess,
                float values[],
                float output[])
{
    __m256 kThresholdVec = _mm256_set1_ps(0.00001f);;
    __m256 threeVec = _mm256_set1_ps(3.f);
    __m256 oneVec = _mm256_set1_ps(1.f);
    __m256 halfVec = _mm256_set1_ps(0.5f);
    __m256 signVec = _mm256_set1_ps(-0.0f);

    for (int i=0; i<N; i+=8) {
        __m256 x = _mm256_loadu_ps(&(values[i]));
        __m256 guess = _mm256_broadcast_ss(&initialGuess);

        // Calculate error
        __m256 error = _mm256_mul_ps(guess, guess);
        error = _mm256_mul_ps(error, x);
        error = _mm256_sub_ps(error, oneVec);
        // Abs error
        error = _mm256_andnot_ps(signVec, error);

        __m256 comparisonVec = _mm256_cmp_ps(error, kThresholdVec, _CMP_GT_OS);
        int comparisonBits = _mm256_movemask_ps(comparisonVec);  // If all 0, then the while loop is finished

        while (comparisonBits != 0) {
            __m256 newGuess = _mm256_mul_ps(guess, guess);
            newGuess = _mm256_mul_ps(newGuess, guess);
            newGuess = _mm256_mul_ps(x, newGuess);
            newGuess = _mm256_sub_ps(_mm256_mul_ps(threeVec, guess), newGuess);
            guess = _mm256_mul_ps(newGuess, halfVec);

            // Calculate error
            error = _mm256_mul_ps(guess, guess);
            error = _mm256_mul_ps(error, x);
            error = _mm256_sub_ps(error, oneVec);

            // Abs error
            error = _mm256_andnot_ps(signVec, error);

            comparisonVec = _mm256_cmp_ps(error, kThresholdVec, _CMP_GT_OS);
            comparisonBits = _mm256_movemask_ps(comparisonVec);  // If all 0, then the while loop is finished
        }
        
        __m256 result = _mm256_mul_ps(x, guess);
        _mm256_storeu_ps(&output[i], result);
    }
}
