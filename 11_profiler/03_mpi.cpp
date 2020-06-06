#include "mpi.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

int main(int argc, char ** argv) {
  struct timeval tic, toc;
  MPI_Init(&argc, &argv);
  int mpisize, mpirank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int * send = new int [1000000000];
  int * recv = new int [1000000000];
  int size = 1024*512;
  for (int i=0; i<10; i++) {
    size *= 2;
    int send_rank = (mpirank + 1) % mpisize;
    int recv_rank = (mpirank - 1 + mpisize) % mpisize;
    MPI_Request request[2];
    MPI_Status status[2];
    gettimeofday(&tic, NULL);
    MPI_Irecv(recv, size, MPI_INT, recv_rank, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Isend(send, size, MPI_INT, send_rank, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Waitall(2, request, status);
    gettimeofday(&toc, NULL);
    if(mpirank==0) printf("%d %lf\n",size,toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6);
  }
  delete[] send;
  delete[] recv;
  MPI_Finalize();
}
