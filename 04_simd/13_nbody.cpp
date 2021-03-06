#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

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
  __m256 zero = _mm256_setzero_ps();
  for(int i=0; i<N; i+=8) {
    __m256 xi = _mm256_load_ps(x+i);
    __m256 yi = _mm256_load_ps(y+i);
    __m256 fxi = zero;
    __m256 fyi = zero;
    for(int j=0; j<N; j++) {
      __m256 dx = _mm256_set1_ps(x[j]);
      __m256 dy = _mm256_set1_ps(y[j]);
      __m256 mj = _mm256_set1_ps(m[j]);
      __m256 r2 = zero;
      dx = _mm256_sub_ps(xi, dx);
      dy = _mm256_sub_ps(yi, dy);
      r2 = _mm256_fmadd_ps(dx, dx, r2);
      r2 = _mm256_fmadd_ps(dy, dy, r2);
      __m256 mask = _mm256_cmp_ps(r2, zero, _CMP_GT_OQ);
      __m256 invR = _mm256_rsqrt_ps(r2);
      invR = _mm256_blendv_ps(zero, invR, mask);
      mj = _mm256_mul_ps(mj, invR);
      invR = _mm256_mul_ps(invR, invR);
      mj = _mm256_mul_ps(mj, invR);
      fxi = _mm256_fmadd_ps(dx, mj, fxi);
      fyi = _mm256_fmadd_ps(dy, mj, fyi);
    }
    _mm256_store_ps(fx+i, fxi);
    _mm256_store_ps(fy+i, fyi);
  }
  for(int i=0; i<N; i++)
    printf("%d %g %g\n",i,fx[i],fy[i]);
}
