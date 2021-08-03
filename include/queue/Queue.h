// Uses the Accessor header for memory mapping on target
#include "../accessor/Accessor.h"

namespace libompx {

class queue {

  template <typename Func> void submit(Func);
};

class handler {

  /// Select target
  enum class targetType {
    cpu, // Execute (multithreaded if applicable) on host
    gpu  // Execute on GPU
  };

  /// Target is defaulted to cpu
  targetType _target = targetType::cpu;

  template <typename Func> void parallel_for_cpu(size_t size, Func logic) {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      logic(i);
    }
  }

  template <typename Func> void parallel_for_gpu(size_t size, Func logic) {
    // TODO : Parallel for --> GPU target
  }

public:
  template <typename Func> void parallel_for(size_t size, Func logic) {

    if (_target == targetType::cpu) {
      parallel_for_cpu(size, logic);
    } else {
      parallel_for_gpu(size, logic);
    }
  }

  template <typename Func> inline void single_task(Func logic) {
    if (_target == targetType::cpu) {
      logic();
    } else {
      // TODO: single task --> GPU target
    }
  }
};

} // namespace libompx
