#include <assert.h>
#include <atomic>
#include <stdio.h>
#include <vector>
namespace libompx {

/// Kernel Launch Configuration Struct (Assuming 1D for now)
typedef struct {
  int gridSize;    // Number of thread blocks per grid
  int blockSize;   // Number of threads per thread block
  size_t smemSize; // Shared Memory Size
  int stream;      // associated stream
} launchConfig;

static int *kernels = nullptr;
static std::atomic<unsigned long> num_kernels = {0};
static std::atomic<unsigned long> synced_kernels = {0};

// Block index within grid
#define blockIdx omp_get_team_num()

// Block dimension (no. of threads per block)
#define blockDim omp_get_num_threads()

// Thread index within thread block
#define threadIdx omp_get_thread_num()

/// used in cudaMemcpy to specify the copy direction
enum cudaMemcpyDir {
  cudaMemcpyHostToDevice, // From Host to Device
  cudaMemcpyDeviceToHost  // From Device to Host
};

launchConfig cfg;

/// Allocate memory on device. Takes a device pointer reference and size
template <typename Ty> void cudaMalloc(Ty **devicePtr, size_t size) {
  int num_devices = omp_get_num_devices();
  assert(num_devices > 0);

  // allocate on default device
  *devicePtr = (Ty *)omp_target_alloc(size, omp_get_default_device());
}

/// Copy memory from host to device or device to host.
template <typename Ty>
void cudaMemcpy(Ty *dst, Ty *src, size_t length, cudaMemcpyDir direction) {
  // First, make sure we have at least one nonhost device
  int num_devices = omp_get_num_devices();
  assert(num_devices > 0);

  // get the host device number (which is the initial device)
  int host_device_num = omp_get_initial_device();

  // use default device for gpu
  int gpu_device_num = omp_get_default_device();

  // default to copy from host to device
  int dst_device_num = gpu_device_num;
  int src_device_num = host_device_num;

  if (direction == cudaMemcpyDeviceToHost) {
    // copy from device to host
    dst_device_num = host_device_num;
    src_device_num = gpu_device_num;
  }

  // parameters are now set, call omp_target_memcpy
  omp_target_memcpy(dst, src, length, 0, 0, dst_device_num, src_device_num);
}

/// Kernel launch function
template <typename Ty, typename Func, Func kernel, typename... Args>
void launch(const std::vector<int> &config, Ty *ptrA, Ty *ptrB, Args... args) {
  // assert(cfg.gridSize > 0);
  // assert(cfg.blockSize > 0);
  // assert(config.size() > 1);

  // Capture configuration
  cfg.gridSize = config[0];
  cfg.blockSize = config[1];
  cfg.smemSize = config.size() > 2 ? config[2] : 0;
  cfg.stream = config.size() > 3 ? config[3] : 0;

  int kernel_no = num_kernels++;
#pragma omp target teams is_device_ptr(ptrA, ptrB) num_teams(cfg.gridSize)     \
    thread_limit(cfg.blockSize) depend(out                                     \
                                       : kernels[kernel_no]) nowait
  {

#pragma omp parallel
    { kernel(ptrA, ptrB, args...); }
  }
}

/// Device Synchronization
void cudaDeviceSynchronize() {
  unsigned long kernel_first = synced_kernels;
  unsigned long kernel_last = num_kernels;
  if (kernel_first < kernel_last) {
    for (unsigned long i = kernel_first; i < kernel_last; ++i) {
#pragma omp parallel
#pragma omp single
#pragma omp task depend(in : kernels[i])
      {}
    }
    synced_kernels.compare_exchange_strong(kernel_first, kernel_last);
  }
}

/// Free allocated memory on device. Takes a device pointer
template <typename Ty> void cudaFree(Ty *devicePtr) {

  assert(omp_get_num_devices() > 0);
  cudaDeviceSynchronize();
  omp_target_free(devicePtr, omp_get_default_device());
}

} // namespace libompx
