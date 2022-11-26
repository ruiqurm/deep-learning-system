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
typedef ssize_t ptrdiff_t;

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

__device__ void GetOffset(CudaVec shape,CudaVec strides,size_t offset,size_t* transform_offset){
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t index[MAX_VEC_SIZE];
  
    int layer_num = 1;
    for(int i =0; i<shape.size;i++){
        layer_num *= shape.data[i];
    }
    size_t counter = gid;
    for(int i =0;i<shape.size;i++){
        layer_num /= shape.data[i];
        index[i] = counter / layer_num;
        counter %= layer_num;
    }
    *transform_offset = offset;
    for(int i=0;i<strides.size;i++){
        *transform_offset += strides.data[i] * index[i];
    }
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
  ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  if (gid >=size)return;
  size_t a_offset;
  GetOffset(shape,strides,offset,&a_offset);
  out[gid] = a[a_offset];
  /// END YOUR SOLUTION
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


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out,size_t size,CudaVec shape,CudaVec strides, size_t offset){

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size)return;
    size_t out_offset;
    GetOffset(shape,strides,offset,&out_offset);  
    out[out_offset] = a[gid];
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
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid,dim.block>>>(a.ptr,out->ptr,a.size,VecToCuda(shape),VecToCuda(strides),offset);
  /// END YOUR SOLUTION
}


__global__ void ScalarSetitemKernel(const scalar_t val,scalar_t* out,size_t size,CudaVec shape,CudaVec strides, size_t offset){
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size)return;
    size_t out_offset;
    GetOffset(shape,strides,offset,&out_offset);  
    out[out_offset] = val;
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
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid,dim.block>>>(val,out->ptr,size,VecToCuda(shape),VecToCuda(strides),offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

// __global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid < size) out[gid] = a[gid] + b[gid];
// }

// void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
//   /**
//    * Add together two CUDA array
//    */
//   CudaDims dim = CudaOneDim(out->size);
//   EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
// }

// __global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid < size) out[gid] = a[gid] + val;
// }

// void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
//   /**
//    * Add together a CUDA array and a scalar value.
//    */
//   CudaDims dim = CudaOneDim(out->size);
//   ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
// }

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

/// BEGIN YOUR SOLUTION
#define ScalarBinOp(Name,OP) \
__global__ void Scalar##Name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if (gid < size) out[gid] = (a[gid]) OP (val);\
}\
void Scalar##Name(const CudaArray& a, scalar_t val, CudaArray* out){\
  CudaDims dim = CudaOneDim(out->size);\
  Scalar##Name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

#define ScalarFunc(Name,F) \
__global__ void Scalar##Name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if (gid < size) out[gid] = F(a[gid],val);\
}\
void Scalar##Name(const CudaArray& a, scalar_t val, CudaArray* out){\
  CudaDims dim = CudaOneDim(out->size);\
  Scalar##Name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

#define EwiseBinOp(Name,OP) \
__global__ void Ewise##Name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if (gid < size) out[gid] = (a[gid]) OP (b[gid]);\
}\
void Ewise##Name(const CudaArray& a, const CudaArray& b, CudaArray* out){\
  CudaDims dim = CudaOneDim(out->size);\
  Ewise##Name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}

#define EwiseFuncOneParm(Name,F) \
__global__ void Ewise##Name##Kernel(const scalar_t* a, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if (gid < size) out[gid] = F(a[gid]);\
}\
void Ewise##Name(const CudaArray& a, CudaArray* out){\
  CudaDims dim = CudaOneDim(out->size);\
  Ewise##Name##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
}

#define EwiseFuncTwoParm(Name,F) \
__global__ void Ewise##Name##Kernel(const scalar_t* a,const scalar_t* b, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if (gid < size) out[gid] = F(a[gid],b[gid]);\
}\
void Ewise##Name(const CudaArray& a,const CudaArray& b, CudaArray* out){\
  CudaDims dim = CudaOneDim(out->size);\
  Ewise##Name##Kernel<<<dim.grid, dim.block>>>(a.ptr,b.ptr, out->ptr, out->size); \
}

__device__ scalar_t max(scalar_t a,scalar_t b){
  return a>b?a:b;
}

ScalarBinOp(Add,+)
ScalarBinOp(Mul,*)
ScalarBinOp(Div,/)
ScalarBinOp(Eq, == )
ScalarBinOp(Ge, >=)
ScalarFunc(Power,pow)
ScalarFunc(Maximum,max)
EwiseBinOp(Add,+)
EwiseBinOp(Mul,*)
EwiseBinOp(Div,/)
EwiseBinOp(Eq,==)
EwiseBinOp(Ge,>=)

EwiseFuncOneParm(Log,log)
EwiseFuncOneParm(Exp,exp)
EwiseFuncOneParm(Tanh,tanh)
EwiseFuncTwoParm(Maximum,max)
/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
#define MATMUL_BLOCK_S 16
#define MATMUL_BLOCK_L 16
#define MATMUL_TILE 8
#define BLOCK_SIZE (MATMUL_BLOCK_S*MATMUL_BLOCK_L*sizeof(scalar_t))
CudaDims CudaMatrixDim(size_t m,size_t n,size_t p) {
  CudaDims dim;
  size_t a_lines = (m + MATMUL_BLOCK_L - 1) / MATMUL_BLOCK_L;
  size_t b_columns = (p + MATMUL_BLOCK_L - 1) / MATMUL_BLOCK_L;
  dim.block = dim3(MATMUL_BLOCK_L/MATMUL_TILE, MATMUL_BLOCK_L/MATMUL_TILE, 1);
  dim.grid = dim3(b_columns, a_lines, 1);
  return dim;
}
 
__device__ scalar_t static inline get_data_or_zero(const scalar_t* data, size_t i, size_t j, size_t m, size_t n) {
  return i<m && j<n ? data[i*n+j] : 0;
}

__global__ void MatmulKernel(const scalar_t* A, const scalar_t* B,scalar_t* O,
                             uint32_t m, uint32_t n,uint32_t p) {
  __shared__ scalar_t sA[MATMUL_BLOCK_L][MATMUL_BLOCK_S],sB[MATMUL_BLOCK_S][MATMUL_BLOCK_L];
  scalar_t c[MATMUL_TILE][MATMUL_TILE] = {0};
  scalar_t a[MATMUL_TILE],b[MATMUL_TILE];
  int real_m = m,real_n = n,real_p = p;
  int ybase = blockIdx.y*blockDim.y +threadIdx.y;
  int xbase = blockIdx.x*blockDim.x +threadIdx.x;
  n = MATMUL_BLOCK_S*((n + MATMUL_BLOCK_S - 1) / MATMUL_BLOCK_S);

  for(int ko=0;ko < n;ko += MATMUL_BLOCK_S){
    __syncthreads();
    // block level
    // cp data to shared memory
    if (threadIdx.x == 0 && threadIdx.y ==0){
      for(int i =0;i<MATMUL_BLOCK_S;i++){
        for(int j=0;j<MATMUL_BLOCK_L;j++){
          sA[j][i] = get_data_or_zero(A,j+blockIdx.y*MATMUL_BLOCK_L,ko+i,real_m,real_n);
          sB[i][j] = get_data_or_zero(B,ko+i,j+blockIdx.x*MATMUL_BLOCK_L,real_n,real_p);
        }
      }

    }
      
    __syncthreads();
    // thread_level
    for(int ki=0;ki<MATMUL_BLOCK_S;++ki){
      // copy data to register
      for(int i=0;i<MATMUL_TILE;i++){
        a[i] = sA[threadIdx.y*MATMUL_TILE+i][ki]; // a vector of size MATMUL_TILE*1
        b[i] = sB[ki][threadIdx.x*MATMUL_TILE+i]; // a vector of size 1*MATMUL_TILE
      }
      for(int y=0;y<MATMUL_TILE;++y){
        for(int x =0;x<MATMUL_TILE;++x){
          c[y][x] += a[y]*b[x];
          // if (threadIdx.y==0&&threadIdx.x==0){
          //   printf("%f ",a[y]*b[x]);
          // }
        }
      }
    }
  }

  // copy data back to global memory
  // because each thread is responsible for specific MATMUL_TILE*MATMUL_TILE elements,
  // there's no need to use atomic operation
  for (int i = 0;i<MATMUL_TILE;i++){
    for(int j = 0;j<MATMUL_TILE;j++){
      if (ybase*MATMUL_TILE+i < real_m && xbase*MATMUL_TILE + j < real_p){
        O[(ybase*MATMUL_TILE+i)*real_p+xbase*MATMUL_TILE + j] = c[i][j];
      }
    }
  }
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

  /// BEGIN YOUR SOLUTION
  auto dim = CudaMatrixDim(M,N,P);
  MatmulKernel<<<dim.grid,dim.block,BLOCK_SIZE*2>>>(a.ptr,b.ptr,out->ptr,M,N,P);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////


__device__ void mat_reduce_max(const scalar_t a,const scalar_t b,scalar_t* out){
  *out = a>b?a:b;
}
__device__ void mat_reduce_sum(const scalar_t a,const scalar_t b,scalar_t* out){
  *out = a+b;
}

#define MAT_REDUCE_TILE 32
template<void REDUCE_FUNC(const scalar_t,const scalar_t,scalar_t*) >
__global__ void ReduceMaxKernel(const scalar_t *a,scalar_t* out,size_t reduce_size,scalar_t INIT_VALUE ){
  extern __shared__ scalar_t smem[]; // used to save result reduced by each thread
  // printf("blockDim.x:%d,blockIdx.x:%d,threadIdx.x:%d\n",blockDim.x,blockIdx.x,threadIdx.x);
  scalar_t max_v = INIT_VALUE;
  size_t base =  blockIdx.x * reduce_size;
  size_t offset = threadIdx.x * MAT_REDUCE_TILE;
  const scalar_t* a_with_offset = a + offset + base;
  for(int i=0;i<MAT_REDUCE_TILE;i++){
    if (offset + i >= reduce_size)break;
    REDUCE_FUNC(a_with_offset[i],max_v,&max_v);
  }
  smem[threadIdx.x] = max_v;
  __syncthreads();
  if (threadIdx.x == 0) {
    max_v = smem[0];
    for (int i = 1; i < blockDim.x; i++) {
      REDUCE_FUNC(max_v, smem[i],&max_v);
    }
    out[blockIdx.x] = max_v;
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
  /// BEGIN YOUR SOLUTION
  CudaDims dim;
  dim.grid = dim3(a.size / reduce_size, 1, 1);
  dim.block = dim3( (reduce_size+MAT_REDUCE_TILE-1) / MAT_REDUCE_TILE, 1, 1);
  size_t size_of_shared_memory = ((reduce_size+MAT_REDUCE_TILE-1) / MAT_REDUCE_TILE) * sizeof(scalar_t);
  ReduceMaxKernel<mat_reduce_max><<<dim.grid, dim.block,size_of_shared_memory>>>(a.ptr,out->ptr,reduce_size,std::numeric_limits<float>::lowest());
  /// END YOUR SOLUTION
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
  /// BEGIN YOUR SOLUTION
  CudaDims dim;
  dim.grid = dim3(a.size / reduce_size, 1, 1);
  dim.block = dim3( (reduce_size+MAT_REDUCE_TILE-1) / MAT_REDUCE_TILE, 1, 1);
  size_t size_of_shared_memory = ((reduce_size+MAT_REDUCE_TILE-1) / MAT_REDUCE_TILE) * sizeof(scalar_t);
  ReduceMaxKernel<mat_reduce_sum><<<dim.grid, dim.block,size_of_shared_memory>>>(a.ptr,out->ptr,reduce_size,0);

  /// END YOUR SOLUTION
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