namespace libompx {

/// Struct used to create blockIdx, blockDim, and threadIdx
typedef struct {
  int x; // x dimension
  int y; // y dimension
  int z; // z dimension
} d3;

/// Kernel Launch Configuration Struct
typedef struct {
  int gridSize;  // Number of thread blocks per grid
  int blockSize; // Number of threads per thread block
  int smemSize;  // Shared Memory Size
} launchConfig;

d3 blockIdx;  // Block index within grid
d3 blockDim;  // Block dimension (no. of threads per block)
d3 threadIdx; // Thread index within the thread block

/// used in cudaMemcpy to specify the copy direction
enum cudaMemcpyDir {
  cudaMemcpyHostToDevice, // From Host to Device
  cudaMemcpyDeviceToHost  // From Device to Host
};

/// A class to implement CUDA-style parallelism functions
class CUWrapper {

public:
  /// Allocate memory on device. Takes a device pointer reference and size
  template <typename Ty> void cudaMalloc(Ty *&devicePtr, size_t size) {
    // host is device 1
    // device is device 0
    devicePtr = (Ty *)omp_target_alloc(size, 0);
  }

  /// Copy memory from host to device or device to host.
  template <typename Ty>
  void cudaMemcpy(Ty *dst, Ty *src, size_t length, cudaMemcpyDir direction) {
    // default to copy from host to device
    // device = 0 , host = 1
    int dst_device_num = 0, src_device_num = 1;
    if (direction == cudaMemcpyDeviceToHost) {
      // copy from host to device
      dst_device_num = 1;
      src_device_num = 0;
    }
    omp_target_memcpy(dst, src, length, 0, 0, dst_device_num, src_device_num);
  }

  /// Kernel launch function
  template <typename Func> void launch(launchConfig cfg, Func kernel) {}

  /// Free allocated memory on device. Takes a device pointer
  template <typename Ty> void cudaFree(Ty *devicePtr) {
    omp_target_free(devicePtr);
  }
};

} // namespace libompx
