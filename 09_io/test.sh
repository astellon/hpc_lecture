mpicxx 16_phdf5_write.cpp --std=c++11 -lhdf5
mpirun -n 4 a.out
mpicxx 17_phdf5_read.cpp --std=c++11 -lhdf5
mpirun -n 4 a.out