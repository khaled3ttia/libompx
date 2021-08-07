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
    int num_devices = omp_get_num_devices();
    assert(num_devices > 0);

    // device is device 0
    devicePtr = (Ty *)omp_target_alloc(size, 0);
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
  template <typename Func> void launch(launchConfig cfg, Func kernel) {}

  /// Free allocated memory on device. Takes a device pointer
  template <typename Ty> void cudaFree(Ty *devicePtr) {
    omp_target_free(devicePtr);
  }
};

} // namespace libompx
