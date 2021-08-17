#include <assert.h>

namespace libompx {

/// Kernel Launch Configuration Struct (Assuming 1D for now)
typedef struct {
  int gridSize;    // Number of thread blocks per grid
  int blockSize;   // Number of threads per thread block
  size_t smemSize; // Shared Memory Size
  int stream;      // associated stream
} launchConfig;

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

void config(int gridSize, int blockSize, size_t smemSize = 0, int stream = 0) {

  cfg.gridSize = gridSize;
  cfg.blockSize = blockSize;
  cfg.smemSize = smemSize;
  cfg.stream = stream;
}

/// Allocate memory on device. Takes a device pointer reference and size
template <typename Ty> void cudaMalloc(Ty **devicePtr, size_t size) {
  int num_devices = omp_get_num_devices();
  assert(num_devices > 0);

  // device is device 0
  *devicePtr = (Ty *)omp_target_alloc(size, 0);
}

/// Copy memory from host to device or device to host.
template <typename Ty>
void cudaMemcpy(Ty *dst, Ty *src, size_t length, cudaMemcpyDir direction) {
  // First, make sure we have at least one nonhost device
  int num_devices = omp_get_num_devices();
  assert(num_devices > 0);

  // get the host device number (which is the initial device)
  int host_device_num = omp_get_initial_device();

  // default to device 0 as a GPU representative
  // (if we have num_devices > 0, then for sure device 0 exists)
  int gpu_device_num = 0;

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
void launch(int n , Ty *ptrA, Ty *ptrB, Args... args) {
  // assert(cfg.gridSize > 0);
  // assert(cfg.blockSize > 0);
#pragma omp target teams is_device_ptr(ptrA, ptrB) num_teams(cfg.gridSize) thread_limit(cfg.blockSize)   
  {
#pragma omp parallel
    { kernel(ptrA, ptrB, args...); }
  }
}

/// Free allocated memory on device. Takes a device pointer
template <typename Ty> void cudaFree(Ty *devicePtr) {
  assert(omp_get_num_devices() > 0);
  omp_target_free(devicePtr, 0);
}

} // namespace libompx
