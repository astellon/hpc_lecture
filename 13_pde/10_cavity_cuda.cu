#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hdf5.h"

#include "timer.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// print array for test and debug
void show(float** data, int nx, int ny) {
  for(int x = 0; x < nx; x++) {
    for(int y = 0; y < ny; y++) {
      printf("%3.3f ", data[x][y]);
    }
    printf("\n");
  }
}

float** matrix(int nx, int ny) {
  float* data;
  float** ptr;
  cudaMallocManaged(&data, nx * ny * sizeof(float));
  cudaMallocManaged(&ptr, nx * sizeof(float*));

  // init
  cudaMemset(data, 0, nx * ny * sizeof(float));

  for (int i = 0; i != nx; i++) {
    ptr[i] = data + i * ny;
  }

  return ptr;
}

void save_matrix(const std::string& file_name, float** mat, int nx, int ny) {
  auto file = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t globaldim[2] = {hsize_t(nx), hsize_t(ny)};

  auto globalspace = H5Screate_simple(2, globaldim, NULL);

  auto dataset = H5Dcreate(file, "dataset", H5T_NATIVE_FLOAT, globalspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  float* data = (float*)malloc(nx*ny*sizeof(float));
  cudaMemcpy(data, mat[0], nx * ny * sizeof(float), cudaMemcpyDefault);

  H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  H5Dclose(dataset);
  H5Sclose(globalspace);
  H5Fclose(file);
  free(data);
}

__global__ void buildup_b(float** b, float rho, float dt, float** u, float** v, float dx, float dy, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // not include bound
  if (0 < i && i < nx - 1 && 0 < j && j < ny - 1) {
    b[i][j] = (
      rho * (
        1/dt * ((u[i+1][j] - u[i-1][j]) / (2 * dx) + (v[i][j+1] - v[i][j-1]) / (2 * dy))
        - (u[i+1][j] - u[i-1][j]) / (2 * dx) * (u[i+1][j] - u[i-1][j]) / (2 * dx)
        - (v[i][j+1] - v[i][j-1]) / (2 * dy) * (v[i][j+1] - v[i][j-1]) / (2 * dy)
        - 2 * (u[i][j+1] - u[i][j-1]) / (2 * dy) * (v[i+1][j] - v[i-1][j]) / (2 * dx)
      )
    );
  }
}

__global__ void pressure_poisson(float** p, float dx, float dy, float** b, int nit, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // not include bound
  if (0 < i && i < nx - 1 && 0 < j && j < ny - 1) {
    // n interation
    for (auto it = 0; it != nit; it++) {
      p[i][j] = (
        ((p[i+1][j] + p[i-1][j]) * dy*dy + (p[i][j+1] + p[i][j-1]) * dx*dx) / (2*(dx*dx + dy*dy))
        - (dx*dx * dy*dy) / (2*(dx*dx + dy*dy)) * b[i][j]
      );
    }
  }

  // include bound
  if (0 <= i && i <= nx - 1 && 0 <= j && j <= ny - 1) {
    // boundary condition
    if (i == 0) {
      p[0][j] = p[1][j];
    } else if (i == nx - 1) {
      p[nx-1][j] = p[nx-2][j];
    } else if (j == 0) {
      p[i][0] = p[i][1];
    } else if (j == ny - 1) {
      p[i][ny-1] = 0;
    }
  }
}

__global__ void step(float** u, float** v, float** un, float** vn, float** p, float dx, float dy, float dt, float rho, float nu, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // not include bound
  if (0 < i && i < nx - 1 && 0 < j && j < ny - 1) {
    u[i][j] = (
      un[i][j] - un[i][j] * dt/dx * (un[i][j] - un[i-1][j]) - vn[i][j] * dt/dy * (un[i][j] - un[i][j-1])
      - dt / (rho * 2 * dx) * (p[i+1][j] - p[i-1][j])
      + nu * (dt / (dx * dx) * (un[i+1][j] - 2*un[i][j] + un[i-1][j]) + dt / (dy * dy) * (un[i][j+1] - 2*un[i][j] + un[i][j-1]))
    );

    v[i][j] = (
      vn[i][j] - un[i][j] * dt/dx * (vn[i][j] - vn[i-1][j]) - vn[i][j] * dt/dy * (vn[i][j] - vn[i][j-1])
      - dt / (rho * 2 * dy) * (p[i][j+1] - p[i][j-1])
      + nu * (dt / (dx * dx) * (vn[i+1][j] - 2*vn[i][j] + vn[i-1][j]) + dt / (dy * dy) * (vn[i][j+1] - 2*vn[i][j] + vn[i][j-1]))
    );
  }

  // include bound
  if (0 <= i && i <= nx - 1 && 0 <= j && j <= ny - 1) {
    // boundary condition
    if (i == 0) {
      u[0][j] = 0;
      v[0][j] = 0;
    } else if (i == nx - 1) {
      u[nx-1][j] = 0;
      v[nx-1][j] = 0;
    }

    if (j == 0) {
      u[i][0] = 0;
      v[i][0] = 0;
    } else if (j == ny - 1) {
      u[i][ny-1] = 1;
      v[i][ny-1] = 0;
    }
  }
}

int main(int argc, char *argv[]) {
  auto nx = 41, ny = 41, nt = 500, nit = 50;
  auto dx = 2.0f / (nx - 1), dy = 2.0f / (ny - 1);

  auto rho = 1;
  auto nu = 0.1f, dt = 0.001f;

  auto u = matrix(nx, ny);
  auto v = matrix(nx, ny);
  auto p = matrix(nx, ny);
  auto b = matrix(nx, ny);

  // kernel function configuration
  int nthreadsx = 32;
  int nthreadsy = 32;

  int nblocksx = nx/nthreadsx+1;
  int nblocksy = ny/nthreadsy+1;

  dim3 grid(nblocksx, nblocksy), block(nthreadsx, nthreadsy);

  auto un = matrix(nx, ny);
  auto vn = matrix(nx, ny);

  int time = 0;
  {
    Timer<std::chrono::milliseconds> timer(&time);

    for (auto t = 0; t != nt; t++) {
      // SLOW!!
      cudaMemcpy(un[0], u[0], nx * ny * sizeof(float), cudaMemcpyDefault);
      cudaMemcpy(vn[0], v[0], nx * ny * sizeof(float), cudaMemcpyDefault);

      buildup_b<<<grid, block>>>(b, rho, dt, u, v, dx, dy, nx, ny);
      gpuErrchk( cudaPeekAtLastError() );
      cudaDeviceSynchronize();

      pressure_poisson<<<grid, block>>>(p, dx, dy, b, nit, nx, ny);
      gpuErrchk( cudaPeekAtLastError() );
      cudaDeviceSynchronize();

      step<<<grid, block>>>(u, v, un, vn, p, dx, dy, dt, rho, nu, nx, ny);
      gpuErrchk( cudaPeekAtLastError() );
      cudaDeviceSynchronize();
    }
  }

  std::cout << "took: " << time  << "ms" << std::endl;

  save_matrix("u.h5", u, nx, ny);
  save_matrix("v.h5", v, nx, ny);
  save_matrix("p.h5", p, nx, ny);

  cudaFree(u[0]);
  cudaFree(v[0]);
  cudaFree(p[0]);
  cudaFree(b[0]);
  cudaFree(un[0]);
  cudaFree(vn[0]);

  cudaFree(u);
  cudaFree(v);
  cudaFree(p);
  cudaFree(b);
  cudaFree(un);
  cudaFree(vn);

  return 0;
}