#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "timer.h"

template <class T>
using Mat = std::vector<std::vector<T>>;

template <class T>
using Vec = std::vector<T>;

template<typename T>
void save_matrix(const std::string& file_name, const Mat<float>& mat, int nx, int ny) {
  auto ofs = std::basic_ofstream<char>(file_name);

  if (!ofs) {
    std::cerr << "Cannot open file" << std::endl;
  }

  for (int i = 0; i != nx; i++) {
    for (int j = 0; j != ny; j++) {
      ofs << mat[i][j] << " ";
    }
    ofs << std::endl;
  }
}

template <typename T>
Mat<T> buildup_b(const Mat<T>& b, T rho, T dt, const Mat<T>& u, const Mat<T>& v, T dx, T dy, int nx, int ny) {
  auto bn = Mat<T>(b);

  for (int i = 1; i != nx - 1; i++) {
    for (int j = 1; j != ny - 1; j++) {
      bn[i][j] = (
        rho * (
          1/dt * ((u[i+1][j] - u[i-1][j]) / (2 * dx) + (v[i][j+1] - v[i][j-1]) / (2 * dy))
          - (u[i+1][j] - u[i-1][j]) / (2 * dx) * (u[i+1][j] - u[i-1][j]) / (2 * dx)
          - (v[i][j+1] - v[i][j-1]) / (2 * dy) * (v[i][j+1] - v[i][j-1]) / (2 * dy)
          - 2 * (u[i][j+1] - u[i][j-1]) / (2 * dy) * (v[i+1][j] - v[i-1][j]) / (2 * dx)
        )
      );
    }
  }

  return bn;
}

template<typename T>
Mat<T> pressure_poisson(const Mat<T>& p, T dx, T dy, const Mat<T>& b, int nit, int nx, int ny) {
  auto pn = Mat<T>(p);

  for (int it = 0; it != nit; it++) {
    for (int i = 1; i != nx - 1; i++) {
      for (int j = 1; j != ny - 1; j++) {
        pn[i][j] = (
          ((pn[i+1][j] + pn[i-1][j]) * dy*dy + (pn[i][j+1] + pn[i][j-1]) * dx*dx) / (2*(dx*dx + dy*dy))
          - (dx*dx * dy*dy) / (2*(dx*dx + dy*dy)) * b[i][j]
        );
      }
    }

    for (int i = 0; i != nx; i++) {
      pn[i][0] = pn[i][1];
      pn[i][ny-1] = 0;
    }

    for (int j = 0; j != ny; j++) {
      pn[0][j] = pn[1][j];
      pn[nx-1][j] = pn[nx-2][j];
    }
  }

  return pn;
}

int main(int argc, char *argv[]) {
  auto nx = 41, ny = 41, nt = 500, nit = 50, c = 1;
  auto dx = 2.0f / (nx - 1), dy = 2.0f / (ny - 1);

  auto rho = 1;
  auto nu = 0.1f, dt = 0.001f;

  auto u = Mat<float>(nx, Vec<float>(ny));
  auto v = Mat<float>(nx, Vec<float>(ny));
  auto p = Mat<float>(nx, Vec<float>(ny));
  auto b = Mat<float>(nx, Vec<float>(ny));

  int time = 0;
  {
    Timer<std::chrono::milliseconds> T(&time);

    for (int t = 0; t != nt; t++) {
      auto un = Mat<float>(u);
      auto vn = Mat<float>(v);

      b = buildup_b<float>(b, rho, dt, u, v, dx, dy, nx, ny);
      p = pressure_poisson<float>(p, dx, dy, b, nit, nx, ny);

      for (int i = 1; i != nx - 1; i++) {
        for (int j = 1; j != ny - 1; j++) {
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
      }

      for (int i = 0; i != nx; i++) {
        u[i][0] = 0;
        u[i][ny-1] = 1;
        v[i][0] = 0;
        v[i][ny-1] = 0;
      }

      for (int j = 0; j != ny; j++) {
        u[0][j] = 0;
        u[nx-1][j] = 0;
        v[0][j] = 0;
        v[nx-1][j] = 0;
      }
    }
  }

  std::cout << "took: " << time  << "ms" << std::endl;

  save_matrix<float>("u.txt", u, nx, ny);
  save_matrix<float>("v.txt", v, nx, ny);
  save_matrix<float>("p.txt", p, nx, ny);

  return 0;
}