#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <vector>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
    size_t ndim = shape.size();
    std::vector<int32_t> indices(ndim, 0);
    
    // Calculate total number of elements
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
      total_elements *= shape[i];
    }
    
    size_t out_idx = 0;
    for (size_t cnt = 0; cnt < total_elements; cnt++) {
      // Calculate input index from strides and offset
      size_t in_idx = offset;
      for (size_t i = 0; i < ndim; i++) {
        in_idx += indices[i] * strides[i];
      }
      
      // Copy from non-compact to compact
      out->ptr[out_idx++] = a.ptr[in_idx];
      
      // Increment indices with carry (like odometer)
      for (int i = ndim - 1; i >= 0; i--) {
        indices[i]++;
        if (indices[i] < shape[i]) {
          break;  // No carry needed
        }
        indices[i] = 0;  // Reset and carry to next dimension
      }
    }
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  std::vector<int32_t> indices(ndim, 0);
  
  // Calculate total number of elements
  size_t total_elements = 1;
  for (size_t i = 0; i < ndim; i++) {
    total_elements *= shape[i];
  }
  
  size_t in_idx = 0;
  for (size_t cnt = 0; cnt < total_elements; cnt++) {
    // Calculate output index from strides and offset
    size_t out_idx = offset;
    for (size_t i = 0; i < ndim; i++) {
      out_idx += indices[i] * strides[i];
    }
    
    // Copy from compact to non-compact
    out->ptr[out_idx] = a.ptr[in_idx++];
    
    // Increment indices with carry
    for (int i = ndim - 1; i >= 0; i--) {
      indices[i]++;
      if (indices[i] < shape[i]) {
        break;
      }
      indices[i] = 0;
    }
  }
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  std::vector<int32_t> indices(ndim, 0);
  
  for (size_t cnt = 0; cnt < size; cnt++) {
    // Calculate output index from strides and offset
    size_t out_idx = offset;
    for (size_t i = 0; i < ndim; i++) {
      out_idx += indices[i] * strides[i];
    }
    
    // Set the scalar value
    out->ptr[out_idx] = val;
    
    // Increment indices with carry
    for (int i = ndim - 1; i >= 0; i--) {
      indices[i]++;
      if (indices[i] < shape[i]) {
        break;
      }
      indices[i] = 0;
    }
  }
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
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
 // Add these implementations after ScalarAdd() and before the Matmul() function

// Element-wise multiplication
void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  }
}

// Element-wise division
void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / b.ptr[i];
  }
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / val;
  }
}

// Scalar power
void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::pow(a.ptr[i], val);
  }
}

// Element-wise maximum
void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
  }
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], val);
  }
}

// Element-wise equality (returns 1.0 for true, 0.0 for false)
void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] == b.ptr[i]) ? 1.0f : 0.0f;
  }
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] == val) ? 1.0f : 0.0f;
  }
}

// Element-wise greater than or equal (returns 1.0 for true, 0.0 for false)
void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] >= b.ptr[i]) ? 1.0f : 0.0f;
  }
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] >= val) ? 1.0f : 0.0f;
  }
}

// Element-wise logarithm
void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::log(a.ptr[i]);
  }
}

// Element-wise exponential
void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::exp(a.ptr[i]);
  }
}

// Element-wise hyperbolic tangent
void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (uint32_t i = 0; i < m * p; i++) {
    out->ptr[i] = 0;
  }
  
  // Naive three-loop matrix multiplication
  for (uint32_t i = 0; i < m; i++) {
    for (uint32_t j = 0; j < p; j++) {
      for (uint32_t k = 0; k < n; k++) {
        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (uint32_t i = 0; i < TILE; i++) {
    for (uint32_t j = 0; j < TILE; j++) {
      for (uint32_t k = 0; k < TILE; k++) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  uint32_t m_tiles = m / TILE;
  uint32_t n_tiles = n / TILE;
  uint32_t p_tiles = p / TILE;
  
  // Initialize output to zero
  for (uint32_t i = 0; i < m_tiles * p_tiles * TILE * TILE; i++) {
    out->ptr[i] = 0;
  }
  
  // Tiled matrix multiplication
  // Iterate over output tiles
  for (uint32_t i = 0; i < m_tiles; i++) {
    for (uint32_t j = 0; j < p_tiles; j++) {
      // For each output tile (i, j), accumulate contributions from all k tiles
      for (uint32_t k = 0; k < n_tiles; k++) {
        // Get pointers to the relevant TILE x TILE blocks
        // a[i][k][...][...] is at offset: (i * n_tiles + k) * TILE * TILE
        const float* a_tile = a.ptr + (i * n_tiles + k) * TILE * TILE;
        
        // b[k][j][...][...] is at offset: (k * p_tiles + j) * TILE * TILE
        const float* b_tile = b.ptr + (k * p_tiles + j) * TILE * TILE;
        
        // out[i][j][...][...] is at offset: (i * p_tiles + j) * TILE * TILE
        float* out_tile = out->ptr + (i * p_tiles + j) * TILE * TILE;
        
        // Multiply the TILE x TILE blocks and add to output
        AlignedDot(a_tile, b_tile, out_tile);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  
  // Handle edge case
  if (reduce_size == 0) return;
  
  for (size_t i = 0; i < out->size; i++) {
    // Start with first element of the block
    size_t base_idx = i * reduce_size;
    scalar_t max_val = a.ptr[base_idx];
    
    // Find max over remaining elements in the block
    for (size_t j = 1; j < reduce_size; j++) {
      max_val = std::max(max_val, a.ptr[base_idx + j]);
    }
    
    out->ptr[i] = max_val;
  }
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  
  // Handle edge case
  if (reduce_size == 0) return;
  
  for (size_t i = 0; i < out->size; i++) {
    // Compute sum over the block
    size_t base_idx = i * reduce_size;
    scalar_t sum = 0;
    
    for (size_t j = 0; j < reduce_size; j++) {
      sum += a.ptr[base_idx + j];
    }
    
    out->ptr[i] = sum;
  }
}

// ===========================================================================
// EIGENDECOMPOSITION (Pure C++ implementation)
// ===========================================================================

void Eigh(const AlignedArray& a, AlignedArray* eigenvalues, AlignedArray* eigenvectors, int n) {
  /**
   * Compute eigenvalues and eigenvectors of a symmetric matrix using QR algorithm.
   */

  const int max_iter = 100;
  const float eps = 1e-10f;

  // Initialize eigenvectors to identity
  for (int i = 0; i < n * n; i++) eigenvectors->ptr[i] = 0.0f;
  for (int i = 0; i < n; i++) eigenvectors->ptr[i * n + i] = 1.0f;

  // Copy matrix to work with (will become tridiagonal)
  std::vector<float> diag(n);      // Main diagonal
  std::vector<float> offdiag(n);   // Off-diagonal (subdiagonal)
  std::vector<float> matrix(n * n);
  std::memcpy(matrix.data(), a.ptr, n * n * sizeof(float));
  
  // STEP 1: Householder reduction to tridiagonal form

  for (int k = 0; k < n - 2; k++) {
    // Compute Householder vector for column k (below diagonal)
    float scale = 0.0f;
    for (int i = k + 1; i < n; i++) {
      scale += matrix[i * n + k] * matrix[i * n + k];
    }
    scale = std::sqrt(scale);

    if (scale < eps) continue;

    // Choose sign to avoid cancellation
    if (matrix[(k + 1) * n + k] > 0) scale = -scale;

    float h = scale * (scale - matrix[(k + 1) * n + k]);
    std::vector<float> v(n, 0.0f);
    v[k + 1] = matrix[(k + 1) * n + k] - scale;
    for (int i = k + 2; i < n; i++) {
      v[i] = matrix[i * n + k];
    }

    // Apply Householder: A <- (I - 2vv^T/h) A (I - 2vv^T/h)
    // Since A is symmetric, this preserves symmetry

    // Compute w = A * v / h
    std::vector<float> w(n, 0.0f);
    for (int i = 0; i < n; i++) {
      for (int j = k + 1; j < n; j++) {
        w[i] += matrix[i * n + j] * v[j];
      }
      w[i] /= h;
    }

    // Compute correction: w <- w - (w^T v / 2h) * v
    float wv = 0.0f;
    for (int i = k + 1; i < n; i++) wv += w[i] * v[i];
    wv /= (2.0f * h);
    for (int i = k + 1; i < n; i++) w[i] -= wv * v[i];

    // Apply: A <- A - v w^T - w v^T
    for (int i = k + 1; i < n; i++) {
      for (int j = k + 1; j < n; j++) {
        matrix[i * n + j] -= v[i] * w[j] + w[i] * v[j];
      }
    }

    // Store the transformed column
    matrix[(k + 1) * n + k] = scale;
    for (int i = k + 2; i < n; i++) {
      matrix[i * n + k] = 0.0f;
      matrix[k * n + i] = 0.0f;
    }

    // Accumulate eigenvector transformation: Q <- Q * (I - 2vv^T/h)
    for (int i = 0; i < n; i++) {
      float dot = 0.0f;
      for (int j = k + 1; j < n; j++) {
        dot += eigenvectors->ptr[i * n + j] * v[j];
      }
      dot *= 2.0f / h;
      for (int j = k + 1; j < n; j++) {
        eigenvectors->ptr[i * n + j] -= dot * v[j];
      }
    }
  }

  // Extract tridiagonal elements
  for (int i = 0; i < n; i++) {
    diag[i] = matrix[i * n + i];
    if (i < n - 1) offdiag[i] = matrix[(i + 1) * n + i];
  }
  offdiag[n - 1] = 0.0f;

  // STEP 2: QR iteration on tridiagonal matrix (implicit shifts)

  for (int l = 0; l < n; l++) {
    int iter = 0;

    while (iter < max_iter) {
      // Find small off-diagonal element to split
      int m;
      for (m = l; m < n - 1; m++) {
        float test = std::abs(diag[m]) + std::abs(diag[m + 1]);
        if (std::abs(offdiag[m]) < eps * test) break;
      }

      if (m == l) break;  // Converged

      // Wilkinson shift: eigenvalue of trailing 2x2 closer to diag[m]
      float d = (diag[m - 1] - diag[m]) / (2.0f * offdiag[m - 1]);
      float r = std::sqrt(d * d + 1.0f);
      float shift = diag[m] - offdiag[m - 1] / (d + (d >= 0 ? r : -r));

      // Implicit QR step with shift
      float c = 1.0f, s = 0.0f;
      float p = diag[l] - shift;
      float q = offdiag[l];

      for (int i = l; i < m; i++) {
        // Givens rotation to zero out element
        float r_val = std::sqrt(p * p + q * q);
        float c_new = p / r_val;
        float s_new = q / r_val;

        // Update tridiagonal elements
        if (i > l) offdiag[i - 1] = r_val;

        float d1 = diag[i];
        float d2 = diag[i + 1];
        float e = offdiag[i];

        diag[i] = c_new * c_new * d1 + 2.0f * c_new * s_new * e + s_new * s_new * d2;
        diag[i + 1] = s_new * s_new * d1 - 2.0f * c_new * s_new * e + c_new * c_new * d2;
        offdiag[i] = c_new * s_new * (d1 - d2) + (c_new * c_new - s_new * s_new) * e;

        // Accumulate eigenvector rotation
        for (int k = 0; k < n; k++) {
          float tmp = eigenvectors->ptr[k * n + i];
          eigenvectors->ptr[k * n + i] = c_new * tmp + s_new * eigenvectors->ptr[k * n + i + 1];
          eigenvectors->ptr[k * n + i + 1] = -s_new * tmp + c_new * eigenvectors->ptr[k * n + i + 1];
        }

        if (i < m - 1) {
          p = offdiag[i];
          q = s_new * offdiag[i + 1];
          offdiag[i + 1] *= c_new;
        }

        c = c_new;
        s = s_new;
      }

      offdiag[m - 1] = s * p;
      iter++;
    }

    if (iter >= max_iter) {
      throw std::runtime_error("Eigendecomposition failed to converge");
    }
  }

  // Copy eigenvalues to output
  for (int i = 0; i < n; i++) {
    eigenvalues->ptr[i] = diag[i];
  }

  // STEP 3: Sort eigenvalues (ascending) and reorder eigenvectors
  
  std::vector<int> idx(n);
  for (int i = 0; i < n; i++) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return eigenvalues->ptr[a] < eigenvalues->ptr[b];
  });

  std::vector<float> sorted_vals(n);
  std::vector<float> sorted_vecs(n * n);

  for (int i = 0; i < n; i++) {
    sorted_vals[i] = eigenvalues->ptr[idx[i]];
    for (int j = 0; j < n; j++) {
      sorted_vecs[j * n + i] = eigenvectors->ptr[j * n + idx[i]];
    }
  }

  std::memcpy(eigenvalues->ptr, sorted_vals.data(), n * sizeof(float));
  std::memcpy(eigenvectors->ptr, sorted_vecs.data(), n * n * sizeof(float));
}  

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
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
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
  m.def("eigh", Eigh);
}
