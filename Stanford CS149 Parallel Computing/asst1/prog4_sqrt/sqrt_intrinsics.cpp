#include <immintrin.h>

void sqrt_intrinsics(int N,
                    float initialGuess,
                    float values[],
                    float output[])
{
    __m256 vkThreshold = _mm256_set1_ps(0.00001f);
    __m256 vone = _mm256_set1_ps(1.f);
    __m256 vhalf = _mm256_set1_ps(0.5f);
    __m256 vthree = _mm256_set1_ps(3.f);
    // 定义一个掩码：除符号位外全为 1， IEEE 754 浮点数标准，单精度浮点数的最高位（第 31 位）是符号位
    __m256 vAbsMask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    for(int i=0; i<N; i+=8)
    {
        // 带 u 的版本支持非对齐内存
        __m256 vx = _mm256_loadu_ps(values+i);
        __m256 vguess = _mm256_set1_ps(initialGuess);

        __m256 verror = _mm256_and_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(vguess, vguess), vx), vone), vAbsMask);

        __m256 vmask = _mm256_cmp_ps(verror, vkThreshold, _CMP_GT_OS);

         int iter = 0;
        // 增加最大迭代次数限制(如100次)，防止遇到 0.0f 时死循环
        while(_mm256_movemask_ps(vmask)!=0 && iter < 100) 
        {
            vguess = _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(vthree, vguess),_mm256_mul_ps(vx, _mm256_mul_ps(vguess, _mm256_mul_ps(vguess, vguess)))), vhalf);
            verror = _mm256_and_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(vguess, vguess), vx), vone), vAbsMask);
            vmask = _mm256_cmp_ps(verror, vkThreshold, _CMP_GT_OS);
            
            iter++;
        }

        _mm256_storeu_ps(output+i, _mm256_mul_ps(vx, vguess));
    }
}