#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ size_t convertIndex(size_t index, CudaVec shape, CudaVec strides, size_t offset){
  size_t res = 0;
  for(int i=shape.size-1; i>=0; i--){
    res += (index%shape.data[i])*strides.data[i];
    index /= shape.data[i];
  }
  return offset + res;
}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if(gid < size) {
    out[gid] = a[convertIndex(gid, shape, strides, offset)];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape, 
                                  CudaVec strides, size_t offset){
  size_t gid = blockDim.x*blockIdx.x + threadIdx.x;
  if(gid < size){
    out[convertIndex(gid, shape, strides, offset)] = a[gid];
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, CudaVec shape, 
                                  CudaVec strides, size_t offset){
  size_t gid = blockDim.x*blockIdx.x + threadIdx.x;
  if(gid < size){
    out[convertIndex(gid, shape, strides, offset)] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */
// 无法直接传递 Host 端的 Lambda
// 参数必须是可平凡复制的（Trivially Copyable）：传递给内核的参数必须是简单的、类似 C 语言的结构体
// 所以使用 macros
// 1. 定义逐元素运算的宏
#define CUDA_EWISE_OP(name, op) \
__global__ void name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){ \
  size_t gid = blockIdx.x*blockDim.x + threadIdx.x; \
  if(gid < size) out[gid] = a[gid] op b[gid]; \
} \
void name(const CudaArray& a, const CudaArray& b, CudaArray* out){ \
  CudaDims dim = CudaOneDim(out->size); \
  name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}

// 2. 定义标量运算的宏
#define CUDA_SCALAR_OP(name, op) \
__global__ void name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){ \
  size_t gid = blockIdx.x*blockDim.x + threadIdx.x; \
  if(gid < size) out[gid] = a[gid] op val; \
} \
void name(const CudaArray& a, scalar_t val, CudaArray* out){ \
  CudaDims dim = CudaOneDim(out->size); \
  name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

CUDA_EWISE_OP(EwiseMul, *)
CUDA_SCALAR_OP(ScalarMul, *)

CUDA_EWISE_OP(EwiseDiv, /)
CUDA_SCALAR_OP(ScalarDiv, /)

CUDA_EWISE_OP(EwiseEq, ==)
CUDA_SCALAR_OP(ScalarEq, ==)

CUDA_EWISE_OP(EwiseGe, >=)
CUDA_SCALAR_OP(ScalarGe, >=)

// 3. 定义逐元素函数的宏
#define CUDA_EWISE_FUNC(name, func) \
__global__ void name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){ \
  size_t gid = blockIdx.x*blockDim.x + threadIdx.x; \
  if(gid < size) out[gid] = func(a[gid], b[gid]); \
} \
void name(const CudaArray& a, const CudaArray& b, CudaArray* out){ \
  CudaDims dim = CudaOneDim(out->size); \
  name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}

// 4. 定义标量函数的宏
#define CUDA_SCALAR_FUNC(name, func) \
__global__ void name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){ \
  size_t gid = blockIdx.x*blockDim.x + threadIdx.x; \
  if(gid < size) out[gid] = func(a[gid], val); \
} \
void name(const CudaArray& a, scalar_t val, CudaArray* out){ \
  CudaDims dim = CudaOneDim(out->size); \
  name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

// 5. 定义一元函数的宏
#define CUDA_UNARY_FUNC(name, func) \
__global__ void name##Kernel(const scalar_t* a, scalar_t* out, size_t size){ \
  size_t gid = blockIdx.x*blockDim.x + threadIdx.x; \
  if(gid < size) out[gid] = func(a[gid]); \
} \
void name(const CudaArray& a, CudaArray* out){ \
  CudaDims dim = CudaOneDim(out->size); \
  name##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
}

CUDA_SCALAR_FUNC(ScalarPower, powf)

CUDA_EWISE_FUNC(EwiseMaximum, fmax)
CUDA_SCALAR_FUNC(ScalarMaximum, fmax)

CUDA_UNARY_FUNC(EwiseLog, logf)
CUDA_UNARY_FUNC(EwiseExp, expf)
CUDA_UNARY_FUNC(EwiseTanh, tanhf)

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
// 每个线程负责输出矩阵 C 的一个点 (row, col)
__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t N, size_t P){
  size_t row = blockIdx.y*blockDim.y+threadIdx.y;
  size_t col = blockIdx.x*blockDim.x+threadIdx.x;

  __shared__ scalar_t sa[TILE][TILE];
  __shared__ scalar_t sb[TILE][TILE];

  scalar_t acc = 0;

  // 沿 N 维度以 TILE 为单位滑动
  for(size_t k_block=0; k_block < (N+TILE-1)/TILE; k_block++){
    if(k_block*TILE+threadIdx.x<N&&row<M){
      sa[threadIdx.y][threadIdx.x] = a[row*N+k_block*TILE+threadIdx.x];
    }else{
      // 超出范围的点赋值为0
      sa[threadIdx.y][threadIdx.x] = 0;
    }
    if(k_block*TILE+threadIdx.y<N&&col<P){
      sb[threadIdx.y][threadIdx.x] = b[(k_block*TILE+threadIdx.y)*P+col];
    }else{
      sb[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
    for(int k=0; k<TILE; k++){
      acc += sa[threadIdx.y][k]*sb[k][threadIdx.x];
    }
    // 等待所有线程计算完成，才能进入下一轮加载
    __syncthreads();
  }
  if(row<M && col<P) out[row*P+col] = acc;
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  size_t num_blocks_x = (P + TILE - 1) / TILE; 
  size_t num_blocks_y = (M + TILE - 1) / TILE;
  dim3 grid = dim3(num_blocks_x, num_blocks_y, 1);
  dim3 block = dim3(TILE, TILE, 1);
  Fill(out, 0);
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
// 一个 thread 负责一个输出值
__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size){
  size_t gid = blockDim.x*blockIdx.x+threadIdx.x;
  if(gid < size){
    scalar_t value = a[gid*reduce_size];
    for(int i=0; i<reduce_size; i++){
      value = fmax(value, a[gid*reduce_size+i]);
    }
    out[gid] = value;
  }
}

// 一个 block 负责一个输出值
__global__ void ReduceMaxKernelOptimized(const scalar_t* a, scalar_t* out, size_t reduce_size){
  size_t blockid = blockIdx.x;
  size_t threadid = threadIdx.x;
  
  __shared__ scalar_t s[BASE_THREAD_NUM];

  // 每个线程负责一部分数据的最大值（Grid-stride loop 思想）
  // 关键：i 增加的方式保证了线程间的内存访问是连续的
  if(threadid < reduce_size){
    scalar_t value = a[blockid*reduce_size + threadid];
    for(int i=threadid+blockDim.x; i<reduce_size; i+=blockDim.x){
      value = fmax(value, a[blockid*reduce_size + i]);
    }
    s[threadid] = value;
    __syncthreads();
  }

  // 树状求最大值 (Tree Reduction)
  // 在共享内存中折半求最大值
  for(int i=blockDim.x/2; i>0; i>>=1){
    if(threadid<i){
      s[threadid] = fmax(s[threadid], s[threadid+i]);
    }
    __syncthreads(); // 必须同步，确保上一轮最大值全完成
  }

  // 由 0 号线程写回结果
  if(threadid == 0){
    out[blockid] = s[threadid];
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  // 根据 reduce_size 的大小动态选择 Kernel。如果 reduce_size > 128，用共享内存版；否则用简单版
  if(reduce_size > 128){
    ReduceMaxKernelOptimized<<<out->size, BASE_THREAD_NUM>>>(a.ptr, out->ptr, reduce_size);
  }else{
    CudaDims dim = CudaOneDim(out->size);
    ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  }
  /// END SOLUTION
}

// 一个 thread 负责一个输出值
__global__ void ReduceSumKernel(scalar_t* a, scalar_t* out, size_t size, size_t reduce_size){
  size_t gid = blockDim.x*blockIdx.x+threadIdx.x;
  if(gid < size){
    scalar_t value = 0;
    for(int i=0; i<reduce_size; i++){
      value += a[gid*reduce_size+i];
    }
    out[gid] = value;
  }
}

// 一个 block 负责一个输出值
__global__ void ReduceSumKernelOptimized(const scalar_t* a, scalar_t* out, size_t reduce_size){
  size_t blockid = blockIdx.x;
  size_t threadid = threadIdx.x;
  
  __shared__ scalar_t s[BASE_THREAD_NUM];

  // 每个线程负责一部分数据的求和（Grid-stride loop 思想）
  // 关键：i 增加的方式保证了线程间的内存访问是连续的
  scalar_t value = 0;
  for(int i=threadid; i<reduce_size; i+=blockDim.x){
    value += a[blockid*reduce_size + i];
  }
  s[threadid] = value;
  __syncthreads();

  // 树状求和 (Tree Reduction)
  // 在共享内存中折半求和
  for(int i=blockDim.x/2; i>0; i>>=1){
    if(threadid<i){
      s[threadid] += s[threadid+i];
    }
    __syncthreads(); // 必须同步，确保上一轮求和全完成
  }

  // 由 0 号线程写回结果
  if(threadid == 0){
    out[blockid] = s[threadid];
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  // 根据 reduce_size 的大小动态选择 Kernel。如果 reduce_size > 128，用共享内存版；否则用简单版
  if(reduce_size > 128){
    ReduceSumKernelOptimized<<<out->size, BASE_THREAD_NUM>>>(a.ptr, out->ptr, reduce_size);
  }else{
    CudaDims dim = CudaOneDim(out->size);
    ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  }
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
