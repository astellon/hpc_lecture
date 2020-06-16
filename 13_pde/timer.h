#include <chrono>

template<typename T>
class Timer {
 public:
  Timer(int* result) : result_(result) {
    *result_ = 0;
    start_ = std::chrono::system_clock::now();
  }

  ~Timer() {
    end_ = std::chrono::system_clock::now();
    *result_ = std::chrono::duration_cast<T>(end_ - start_).count();
  }

 private:
  std::chrono::system_clock::time_point start_;
  std::chrono::system_clock::time_point end_;
  int* result_;
};