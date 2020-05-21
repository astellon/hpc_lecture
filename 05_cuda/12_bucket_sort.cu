#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_sort(int* key) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int bucket[];
  atomicAdd(bucket+key[i], 1);
  __syncthreads();  // need to wail filling bucket
  for (int j = 0, k=0; j <= i; k++) {  // O(range) loop
    key[i] = k;
    j += bucket[k];
  }
}

int main() {
  int n = 500;
  int range = 5;

  // allocation
  int* key;
  cudaMallocManaged(&key, n*sizeof(int));

  // initialization
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  bucket_sort<<<1, n, range>>>(key);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
