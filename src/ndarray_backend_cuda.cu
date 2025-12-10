#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <vector>     
#include <algorithm>  
#include <cstring>    


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

// Utility function to convert contiguous index i to memory location from strides

__device__ size_t index_transform(size_t gid, CudaVec shape, CudaVec strides, size_t offset) {
  /**
   * Convert a linear index in a compact array to the corresponding position 
   * in a strided array.
   * 
   * Args:
   *   gid: linear index in the compact array
   *   shape: shape of the array
   *   strides: strides of the strided array
   *   offset: offset in the strided array
   * 
   * Returns:
   *   The position in the strided array corresponding to gid
   */
  size_t idx = offset;
  size_t remaining = gid;
  
  // Convert linear index to multi-dimensional coordinates, then apply strides
  // Process dimensions from right to left (fastest to slowest changing)
  for (int i = shape.size - 1; i >= 0; i--) {
    size_t coord = remaining % shape.data[i];
    remaining /= shape.data[i];
    idx += coord * strides.data[i];
  }
  
  return idx;
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
  if (gid < size) {
    // Convert compact index to strided array position
    size_t strided_idx = index_transform(gid, shape, strides, offset);
    // Copy from strided array to compact array
    out[gid] = a[strided_idx];
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
                                   CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for element-wise setitem operation.
   * This copies from a compact array to a strided (non-compact) array.
   * 
   * Args:
   *   a: CUDA pointer to compact input array
   *   out: CUDA pointer to strided output array
   *   size: size of input array (compact)
   *   shape: vector of shapes
   *   strides: vector of strides of the output array
   *   offset: offset of the output array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < size) {
    // Convert compact index to strided array position
    size_t strided_idx = index_transform(gid, shape, strides, offset);
    // Copy from compact array to strided array
    out[strided_idx] = a[gid];
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


__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for scalar setitem operation.
   * This writes a scalar value to a strided (non-compact) array.
   * 
   * Args:
   *   val: scalar value to write
   *   out: CUDA pointer to strided output array
   *   size: number of elements to write
   *   shape: vector of shapes
   *   strides: vector of strides of the output array
   *   offset: offset of the output array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < size) {
    // Convert compact index to strided array position
    size_t strided_idx = index_transform(gid, shape, strides, offset);
    // Write scalar value to strided array
    out[strided_idx] = val;
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


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Division
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Power
////////////////////////////////////////////////////////////////////////////////

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = powf(a[gid], val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Maximum
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = fmaxf(a[gid], b[gid]);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = fmaxf(a[gid], val);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Equality
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] == b[gid]) ? 1.0f : 0.0f;
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] == val) ? 1.0f : 0.0f;
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Greater or Equal
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] >= b[gid]) ? 1.0f : 0.0f;
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] >= val) ? 1.0f : 0.0f;
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Logarithm
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = logf(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Exponential
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = expf(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Tanh
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = tanhf(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, 
                             uint32_t M, uint32_t N, uint32_t P) {
  /**
   * CUDA kernel for matrix multiplication with shared memory tiling.
   * Computes C = A * B where A is M×N, B is N×P, and C is M×P.
   * Uses cooperative fetching and shared memory for efficiency.
   */
  
  // Allocate shared memory for tiles
  __shared__ scalar_t tile_a[TILE][TILE];
  __shared__ scalar_t tile_b[TILE][TILE];
  
  // Thread indices within the block
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  // Calculate the row and column of the output element this thread computes
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;
  
  // Accumulator for the dot product
  scalar_t sum = 0.0f;
  
  // Loop over tiles along the shared dimension N
  int num_tiles = (N + TILE - 1) / TILE;
  for (int t = 0; t < num_tiles; t++) {
    // Cooperatively load tile of A into shared memory
    // Each thread loads one element
    int a_row = row;
    int a_col = t * TILE + tx;
    if (a_row < M && a_col < N) {
      tile_a[ty][tx] = a[a_row * N + a_col];
    } else {
      tile_a[ty][tx] = 0.0f;  // Padding for out-of-bounds
    }
    
    // Cooperatively load tile of B into shared memory
    int b_row = t * TILE + ty;
    int b_col = col;
    if (b_row < N && b_col < P) {
      tile_b[ty][tx] = b[b_row * P + b_col];
    } else {
      tile_b[ty][tx] = 0.0f;  // Padding for out-of-bounds
    }
    
    // Synchronize to ensure all threads have loaded their data
    __syncthreads();
    
    // Compute partial dot product for this tile
    #pragma unroll
    for (int k = 0; k < TILE; k++) {
      sum += tile_a[ty][k] * tile_b[k][tx];
    }
    
    // Synchronize before loading the next tile
    __syncthreads();
  }
  
  // Write the result to global memory
  if (row < M && col < P) {
    out[row * P + col] = sum;
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.
   * Uses tiled matrix multiplication with shared memory for efficiency.
   * 
   * Args:
   *   a: compact 2D array of size M x N
   *   b: compact 2D array of size N x P
   *   out: compact 2D array of size M x P to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
  
  /// BEGIN SOLUTION
  // Configure 2D grid of 2D thread blocks
  dim3 blockDim(TILE, TILE);  // TILE x TILE threads per block
  dim3 gridDim((P + TILE - 1) / TILE, (M + TILE - 1) / TILE);  // Blocks to cover output
  
  MatmulKernel<<<gridDim, blockDim>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////


__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {
  /**
   * Kernel for reduction by taking maximum over contiguous blocks.
   * Each thread handles one complete reduction.
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < out_size) {
    // Each thread computes one output element by finding max over reduce_size elements
    size_t offset = gid * reduce_size;
    scalar_t max_val = a[offset];
    
    for (size_t i = 1; i < reduce_size; i++) {
      max_val = fmaxf(max_val, a[offset + i]);
    }
    
    out[gid] = max_val;
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
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}


__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {
  /**
   * Kernel for reduction by taking sum over contiguous blocks.
   * Each thread handles one complete reduction.
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < out_size) {
    // Each thread computes one output element by summing reduce_size elements
    size_t offset = gid * reduce_size;
    scalar_t sum = 0.0f;
    
    for (size_t i = 0; i < reduce_size; i++) {
      sum += a[offset + i];
    }
    
    out[gid] = sum;
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
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

// ===========================================================================
// EIGENDECOMPOSITION (host-side QR, with device I/O)
// ===========================================================================

void Eigh(const CudaArray& a, CudaArray* eigenvalues, CudaArray* eigenvectors, int n) {
  // Copy input matrix from device to host
  std::vector<scalar_t> matrix(n * n);
  cudaError_t err = cudaMemcpy(matrix.data(), a.ptr, n * n * ELEM_SIZE, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

  // Host-side buffers
  std::vector<scalar_t> diag(n);
  std::vector<scalar_t> offdiag(n);
  std::vector<scalar_t> vecs(n * n);

  const int max_iter = 100;
  const scalar_t eps = (scalar_t)1e-10;

  // Initialize eigenvectors to identity
  for (int i = 0; i < n * n; i++) vecs[i] = 0.0f;
  for (int i = 0; i < n; i++) vecs[i * n + i] = 1.0f;

  // Householder reduction to tridiagonal
  for (int k = 0; k < n - 2; k++) {
    scalar_t scale = 0.0f;
    for (int i = k + 1; i < n; i++) {
      scalar_t v = matrix[i * n + k];
      scale += v * v;
    }
    scale = std::sqrt(scale);
    if (scale < eps) continue;

    if (matrix[(k + 1) * n + k] > 0) scale = -scale;

    scalar_t h = scale * (scale - matrix[(k + 1) * n + k]);
    std::vector<scalar_t> v(n, 0.0f);
    v[k + 1] = matrix[(k + 1) * n + k] - scale;
    for (int i = k + 2; i < n; i++) {
      v[i] = matrix[i * n + k];
    }

    std::vector<scalar_t> w(n, 0.0f);
    for (int i = 0; i < n; i++) {
      scalar_t acc = 0.0f;
      for (int j = k + 1; j < n; j++) {
        acc += matrix[i * n + j] * v[j];
      }
      w[i] = acc / h;
    }

    scalar_t wv = 0.0f;
    for (int i = k + 1; i < n; i++) wv += w[i] * v[i];
    wv /= (2.0f * h);
    for (int i = k + 1; i < n; i++) w[i] -= wv * v[i];

    for (int i = k + 1; i < n; i++) {
      for (int j = k + 1; j < n; j++) {
        matrix[i * n + j] -= v[i] * w[j] + w[i] * v[j];
      }
    }

    matrix[(k + 1) * n + k] = scale;
    matrix[k * n + (k + 1)] = scale;
    for (int i = k + 2; i < n; i++) {
      matrix[i * n + k] = 0.0f;
      matrix[k * n + i] = 0.0f;
    }

    // accumulate into vecs
    for (int i = 0; i < n; i++) {
      scalar_t dot = 0.0f;
      for (int j = k + 1; j < n; j++) {
        dot += vecs[i * n + j] * v[j];
      }
      dot *= (scalar_t)(2.0) / h;
      for (int j = k + 1; j < n; j++) {
        vecs[i * n + j] -= dot * v[j];
      }
    }
  }

  // Extract tridiagonal
  for (int i = 0; i < n; i++) {
    diag[i] = matrix[i * n + i];
    if (i < n - 1) offdiag[i] = matrix[(i + 1) * n + i];
  }
  offdiag[n - 1] = 0.0f;

  // QR iterations with Wilkinson shifts
  for (int l = 0; l < n; l++) {
    int iter = 0;
    while (iter < max_iter) {
      int m;
      for (m = l; m < n - 1; m++) {
        scalar_t test = std::abs(diag[m]) + std::abs(diag[m + 1]);
        if (std::abs(offdiag[m]) < eps * test) break;
      }
      if (m == l) break;

      scalar_t d = (diag[m - 1] - diag[m]) / (2.0f * offdiag[m - 1]);
      scalar_t r = std::sqrt(d * d + 1.0f);
      scalar_t shift = diag[m] - offdiag[m - 1] / (d + (d >= 0 ? r : -r));

      scalar_t p = diag[l] - shift;
      scalar_t q = offdiag[l];

      for (int i = l; i < m; i++) {
        scalar_t r_val = std::sqrt(p * p + q * q);
        scalar_t c = p / r_val;
        scalar_t s = q / r_val;

        if (i > l) offdiag[i - 1] = r_val;

        scalar_t d1 = diag[i];
        scalar_t d2 = diag[i + 1];
        scalar_t e = offdiag[i];

        scalar_t cc = c * c;
        scalar_t ss = s * s;
        scalar_t cs2 = 2.0f * c * s;

        diag[i] = cc * d1 + cs2 * e + ss * d2;
        diag[i + 1] = ss * d1 - cs2 * e + cc * d2;
        offdiag[i] = c * s * (d1 - d2) + (cc - ss) * e;

        for (int k = 0; k < n; k++) {
          scalar_t t = vecs[k * n + i];
          scalar_t u = vecs[k * n + i + 1];
          vecs[k * n + i] = c * t + s * u;
          vecs[k * n + i + 1] = -s * t + c * u;
        }

        if (i < m - 1) {
          p = offdiag[i];
          q = s * offdiag[i + 1];
          offdiag[i + 1] *= c;
        } else {
          p = offdiag[i];
          q = 0.0f;
        }
      }

      offdiag[m - 1] = p;
      iter++;
    }

    if (iter >= max_iter) {
      throw std::runtime_error("Cuda Eigh failed to converge");
    }
  }

  // Sort eigenvalues and eigenvectors (ascending)
  std::vector<int> idx(n);
  for (int i = 0; i < n; i++) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int a_i, int b_i) {
    return diag[a_i] < diag[b_i];
  });

  std::vector<scalar_t> sorted_vals(n);
  std::vector<scalar_t> sorted_vecs(n * n);
  for (int i = 0; i < n; i++) {
    int src_col = idx[i];
    sorted_vals[i] = diag[src_col];
    for (int j = 0; j < n; j++) {
      sorted_vecs[j * n + i] = vecs[j * n + src_col];
    }
  }

  // Copy results back to device
  err = cudaMemcpy(eigenvalues->ptr, sorted_vals.data(), n * ELEM_SIZE, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMemcpy(eigenvectors->ptr, sorted_vecs.data(), n * n * ELEM_SIZE, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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

  m.def("eigh", Eigh);
}
