#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <immintrin.h>

// FOR DEBUG: print vector content
void print_vec(__m256 x) {
  float* y = (float*)malloc(sizeof(float)*8);
  _mm256_store_ps(y, x);
  for (int i = 0; i < 8; i++) {
    printf("%g ", y[i]);
  }
  printf("\n");
  free(y);
}

float sum256(__m256 x) {
  const __m128 hi4 = _mm256_extractf128_ps(x, 1);
  const __m128 lo4 = _mm256_castps256_ps128(x);
  const __m128 sum4 = _mm_add_ps(lo4, hi4);
  const __m128 lo2 = sum4;
  const __m128 hi2 = _mm_movehl_ps(sum4, sum4);
  const __m128 sum2 = _mm_add_ps(lo2, hi2);
  const __m128 lo = sum2;
  const __m128 hi = _mm_shuffle_ps(sum2, sum2, 0x1);
  const __m128 sum = _mm_add_ss(lo, hi);
  return _mm_cvtss_f32(sum);
}

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m256 xi = _mm256_set1_ps(x[i]);
    __m256 yi = _mm256_set1_ps(y[i]);

    __m256 xj = _mm256_load_ps(x);
    __m256 yj = _mm256_load_ps(y);
    __m256 mj = _mm256_load_ps(m);

    __m256 rxj = _mm256_sub_ps(xi, xj);
    __m256 ryj = _mm256_sub_ps(yi, yj);

    __m256 r = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxj, rxj), _mm256_mul_ps(ryj, ryj)));

    // replace INF with 0
    __m256 mask = _mm256_cmp_ps(r, _mm256_set1_ps(INFINITY), 0);
    r = _mm256_blendv_ps(r, _mm256_setzero_ps(), mask);

    __m256 fxi = _mm256_mul_ps(_mm256_mul_ps(rxj, mj), _mm256_mul_ps(_mm256_mul_ps(r, r), r));
    __m256 fyi = _mm256_mul_ps(_mm256_mul_ps(ryj, mj), _mm256_mul_ps(_mm256_mul_ps(r, r), r));

    // sum
    float fxi_sum = sum256(fxi);
    float fyi_sum = sum256(fyi);

    fx[i] -= fxi_sum;
    fy[i] -= fyi_sum;

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
