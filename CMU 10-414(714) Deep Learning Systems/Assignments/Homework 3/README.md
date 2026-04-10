# Homework 3

**作业原址**：[dlsyscourse/hw3](https://github.com/dlsyscourse/hw3)
非常感谢老师的付出和开源，以下是我的实现(特别感谢 Google AI Studio 提供远程指导😝)   

## Part 1: Python array operations

```python
	def reshape(self, new_shape: tuple[int, ...]) -> "NDArray":
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.

        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.

        Args:
            new_shape (tuple): new shape of the array

        Returns:
            NDArray : reshaped array; this will point to thep
        """

        ### BEGIN YOUR SOLUTION
        if prod(self._shape) != prod(new_shape) or not self.is_compact():
          raise ValueError("product of current shape is not equal to the product of the new shape or the matrix is not compact")
        
        return self.make(new_shape, self.compact_strides(new_shape), self._device, self._handle, self._offset)
        ### END YOUR SOLUTION

    def permute(self, new_axes: tuple[int, ...]) -> "NDArray":
        """
        Permute order of the dimensions.  new_axes describes a permuation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memroy as the original array.

        Args:
            new_axes (tuple): permuation order of the dimensions

        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """

        ### BEGIN YOUR SOLUTION
        new_shape = []
        new_strides = []
        for axe in new_axes:
          new_shape.append(self._shape[axe])
          new_strides.append(self._strides[axe])
        return self.make(tuple(new_shape), tuple(new_strides), self._device, self._handle, self._offset)
        ### END YOUR SOLUTION

    def broadcast_to(self, new_shape: tuple[int, ...]) -> "NDArray":
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1

        Args:
            new_shape (tuple): shape to broadcast to

        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """

        ### BEGIN YOUR SOLUTION
        assert len(new_shape) == len(self._shape), "dimension"
        new_strides = list(self._strides)
        for i, shape in enumerate(new_shape):
          assert self._shape[i] == 1 or shape == self._shape[i], "error"
          if shape != self._shape[i]:
            new_strides[i] = 0
        
        return self.make(new_shape, tuple(new_strides), self._device, self._handle, self._offset)

        ### END YOUR SOLUTION]

    ### Get and set elements

    def process_slice(self, sl: slice, dim: int) -> slice:
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start is None:
            start = 0
        if start < 0:
            # start = self.shape[dim]
            # 潜在错误修复
            start = self.shape[dim] + start
        if stop is None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step is None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs: int | slice | tuple[int | slice, ...]) -> "NDArray":
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.

        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory

        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.

        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            coresponding to the subset of the matrix to get

        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memroy but just
            manipulate the shape/strides/offset of the new array, referecing
            the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        slices = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(slices) == self.ndim, "Need indexes equal to number of dimensions"

        ### BEGIN YOUR SOLUTION
        new_offset = self._offset
        # size = 1
        new_strides = list(self._strides)
        new_shape = list(self._shape)
        for i in range(1, len(self._shape) + 1):
          # new_offset += size*slices[-i].start
          # 非连续（non-compact）数组上再次切片
          new_offset += self._strides[-i] * slices[-i].start
          # size *= new_shape[-i]
          # // 整除
          # 用 (stop-start+step-1)//step 处理整除向上取整
          new_shape[-i] = (slices[-i].stop - slices[-i].start + slices[-i].step - 1) // slices[-i].step
          new_strides[-i] *= slices[-i].step
        
        return self.make(tuple(new_shape), tuple(new_strides), self._device, self._handle, new_offset)
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "(permute or reshape or broadcast or getitem) and cpu and not compact"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw3
plugins: typeguard-4.5.1, langsmith-0.7.23, anyio-4.13.0
collected 136 items / 126 deselected / 10 selected                             

tests/hw3/test_ndarray.py::test_permute[cpu-params0] PASSED              [ 10%]
tests/hw3/test_ndarray.py::test_permute[cpu-params1] PASSED              [ 20%]
tests/hw3/test_ndarray.py::test_permute[cpu-params2] PASSED              [ 30%]
tests/hw3/test_ndarray.py::test_reshape[cpu-params0] PASSED              [ 40%]
tests/hw3/test_ndarray.py::test_reshape[cpu-params1] PASSED              [ 50%]
tests/hw3/test_ndarray.py::test_getitem[cpu-params0] PASSED              [ 60%]
tests/hw3/test_ndarray.py::test_getitem[cpu-params1] PASSED              [ 70%]
tests/hw3/test_ndarray.py::test_getitem[cpu-params2] PASSED              [ 80%]
tests/hw3/test_ndarray.py::test_getitem[cpu-params3] PASSED              [ 90%]
tests/hw3/test_ndarray.py::test_broadcast_to[cpu-params0] PASSED         [100%]

===================== 10 passed, 126 deselected in 20.35s ======================
```

## Part 2: CPU Backend - Compact and setitem

```C++
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
  int dim = shape.size();
  std::vector<int32_t> index(dim, 0);
  uint32_t gid = 0;
  for(int i=0; i<out->size; i++)
  {
    out->ptr[i] = a.ptr[offset + gid];
    for(int j=dim-1; j>=0; j--)
    {
      index[j]++;
      gid += strides[j];
      if (index[j] < shape[j]) {
          break;
      } else {
          index[j] = 0;
          gid -= shape[j] * strides[j]; // 发生进位，物理地址回退到当前维度的起始
          // 继续循环到高维度进行 +1
      }
      // index[j] += cnt;
      // cnt = index[j] / shape[j];
      // index[j] %= shape[j];
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
  int dim = shape.size();
  std::vector<int32_t> index(dim, 0);
  for(int i=0; i<a.size; i++)
  {
    int32_t now = offset;
    int cnt = 1;
    for(int j=dim-1; j>=0; j--)
    {
      now += strides[j]*index[j];
      index[j] += cnt;
      cnt = index[j] / shape[j];
      index[j] %= shape[j];
    }
    out->ptr[now] = a.ptr[i];
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
  int dim = shape.size();
  std::vector<int32_t> index(dim, 0);
  for(int i=0; i<size; i++)
  {
    int32_t now = offset;
    int cnt = 1;
    for(int j=dim-1; j>=0; j--)
    {
      now += strides[j]*index[j];
      index[j] += cnt;
      cnt = index[j] / shape[j];
      index[j] %= shape[j];
    }
    out->ptr[now] = val;
  }
  /// END SOLUTION
}
```

```bash
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Found pybind11: /usr/local/lib/python3.12/dist-packages/pybind11/include (found version "3.0.3")
CUDA_TOOLKIT_ROOT_DIR not found or specified
-- Could NOT find CUDA (missing: CUDA_TOOLKIT_ROOT_DIR CUDA_NVCC_EXECUTABLE CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY) 
-- Configuring done (0.3s)
-- Generating done (0.4s)
-- Build files have been written to: /content/drive/MyDrive/10714/hw3/build
make[1]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[2]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[-50%] Building CXX object CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o
[  0%] Linking CXX shared module /content/drive/MyDrive/10714/hw3/python/needle/backend_ndarray/ndarray_backend_cpu.cpython-312-x86_64-linux-gnu.so
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[  0%] Built target ndarray_backend_cpu
make[2]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[1]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw3
plugins: typeguard-4.5.1, langsmith-0.7.23, anyio-4.13.0
collected 136 items / 121 deselected / 15 selected                             

tests/hw3/test_ndarray.py::test_compact[cpu-transpose] PASSED            [  6%]
tests/hw3/test_ndarray.py::test_compact[cpu-broadcast_to] PASSED         [ 13%]
tests/hw3/test_ndarray.py::test_compact[cpu-reshape1] PASSED             [ 20%]
tests/hw3/test_ndarray.py::test_compact[cpu-reshape2] PASSED             [ 26%]
tests/hw3/test_ndarray.py::test_compact[cpu-reshape3] PASSED             [ 33%]
tests/hw3/test_ndarray.py::test_compact[cpu-getitem1] PASSED             [ 40%]
tests/hw3/test_ndarray.py::test_compact[cpu-getitem2] PASSED             [ 46%]
tests/hw3/test_ndarray.py::test_compact[cpu-transposegetitem] PASSED     [ 53%]
tests/hw3/test_ndarray.py::test_setitem_ewise[cpu-params0] PASSED        [ 60%]
tests/hw3/test_ndarray.py::test_setitem_ewise[cpu-params1] PASSED        [ 66%]
tests/hw3/test_ndarray.py::test_setitem_ewise[cpu-params2] PASSED        [ 73%]
tests/hw3/test_ndarray.py::test_setitem_scalar[cpu-params0] PASSED       [ 80%]
tests/hw3/test_ndarray.py::test_setitem_scalar[cpu-params1] PASSED       [ 86%]
tests/hw3/test_ndarray.py::test_setitem_scalar[cpu-params2] PASSED       [ 93%]
tests/hw3/test_ndarray.py::test_setitem_scalar[cpu-params3] PASSED       [100%]

====================== 15 passed, 121 deselected in 0.45s ======================
```

## Part 3: CPU Backend - Elementwise and scalar operations

**内联 (Inlining)**：当 EwiseApply 被调用并传入一个 Lambda 时，编译器会创建一个该模板的特化版本。它不再去跳转地址找函数，而是把 x * y 或 std::log(x) 直接写进 for 循环里  
**循环展开与向量化 (SIMD)**：由于循环体现在是“透明”的，现代编译器（如 GCC/Clang）能够识别出这是连续内存的操作，从而使用 **AVX/SSE 指令集** 一次性处理 4 到 8 个浮点数。这比函数指针快 **5-10 倍**  
**类型安全**：比宏（Macro）更安全，能够利用 C++ 的类型检查和 std 库的优化  

```C++
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
// 将操作函数定义为模板参数 F，编译器会为每一个不同的算子生成一个专门的循环版本
/**
 * 逐元素二元运算模板
 */
template <typename F>
void EwiseApply(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, F&& func) {
    // 使用 OpenMP 加速
    #pragma omp parallel for
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = func(a.ptr[i], b.ptr[i]);
    }
}

/**
 * 标量二元运算模板
 */
template <typename F>
void ScalarApply(const AlignedArray& a, scalar_t val, AlignedArray* out, F&& func) {
    // 使用 OpenMP 加速
    #pragma omp parallel for
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = func(a.ptr[i], val);
    }
}

/**
 * 一元运算模板
 */
template <typename F>
void UnaryApply(const AlignedArray& a, AlignedArray* out, F&& func) {
    // 使用 OpenMP 加速
    #pragma omp parallel for
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = func(a.ptr[i]);
    }
}

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out){
  EwiseApply(a, b, out, std::multiplies<scalar_t>());
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out){
  ScalarApply(a, val, out, std::multiplies<scalar_t>());
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out){
  EwiseApply(a, b, out, std::divides<scalar_t>());
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out){
  ScalarApply(a, val, out, std::divides<scalar_t>());
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out){
  ScalarApply(a, val, out, [](scalar_t x, scalar_t val) -> scalar_t{
    return std::pow(x, val);
  });
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out){
  EwiseApply(a, b, out, [](scalar_t x, scalar_t y) -> scalar_t{
    return std::max(x, y);
  });
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out){
  ScalarApply(a, val, out, [](scalar_t x, scalar_t val) -> scalar_t{
    return std::max(x, val);
  });
}

// 逻辑运算 (注意返回类型虽是 scalar_t，但逻辑是 0 或 1)
void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out){
  EwiseApply(a, b, out, [](scalar_t x, scalar_t y) -> scalar_t{
    return static_cast<scalar_t>(x == y);
  });
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out){
  ScalarApply(a, val, out, [](scalar_t x, scalar_t val) -> scalar_t{
    return static_cast<scalar_t>(x == val);
  });
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out){
  EwiseApply(a, b, out, [](scalar_t x, scalar_t y) -> scalar_t{
    return static_cast<scalar_t>(x >= y);
  });
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out){
  ScalarApply(a, val, out, [](scalar_t x, scalar_t val) -> scalar_t{
    return static_cast<scalar_t>(x >= val);
  });
}

void EwiseLog(const AlignedArray& a, AlignedArray* out){
  UnaryApply(a, out, [](scalar_t x) -> scalar_t{
    return log(x);
  });
}

void EwiseExp(const AlignedArray& a, AlignedArray* out){
  UnaryApply(a, out, [](scalar_t x) -> scalar_t{
    return exp(x);
  });
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out){
  UnaryApply(a, out, [](scalar_t x) -> scalar_t{
    return tanh(x);
  });
}
```

```bash
!make
!python3 -m pytest -v -k "(ewise_fn or ewise_max or log or exp or tanh or (scalar and not setitem)) and cpu"
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Found pybind11: /usr/local/lib/python3.12/dist-packages/pybind11/include (found version "3.0.3")
CUDA_TOOLKIT_ROOT_DIR not found or specified
-- Could NOT find CUDA (missing: CUDA_TOOLKIT_ROOT_DIR CUDA_NVCC_EXECUTABLE CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY) 
-- Configuring done (0.3s)
-- Generating done (0.4s)
-- Build files have been written to: /content/drive/MyDrive/10714/hw3/build
make[1]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[2]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[-50%] Building CXX object CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o
[  0%] Linking CXX shared module /content/drive/MyDrive/10714/hw3/python/needle/backend_ndarray/ndarray_backend_cpu.cpython-312-x86_64-linux-gnu.so
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[  0%] Built target ndarray_backend_cpu
make[2]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[1]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw3
plugins: typeguard-4.5.1, langsmith-0.7.23, anyio-4.13.0
collected 136 items / 113 deselected / 23 selected                             

tests/hw3/test_ndarray.py::test_ewise_fn[cpu-shape0-multiply] PASSED     [  4%]
tests/hw3/test_ndarray.py::test_ewise_fn[cpu-shape0-divide] PASSED       [  8%]
tests/hw3/test_ndarray.py::test_ewise_fn[cpu-shape0-add] PASSED          [ 13%]
tests/hw3/test_ndarray.py::test_ewise_fn[cpu-shape0-subtract] PASSED     [ 17%]
tests/hw3/test_ndarray.py::test_ewise_fn[cpu-shape0-equal] PASSED        [ 21%]
tests/hw3/test_ndarray.py::test_ewise_fn[cpu-shape0-greater_than] PASSED [ 26%]
tests/hw3/test_ndarray.py::test_ewise_fn[cpu-shape1-multiply] PASSED     [ 30%]
tests/hw3/test_ndarray.py::test_ewise_fn[cpu-shape1-divide] PASSED       [ 34%]
tests/hw3/test_ndarray.py::test_ewise_fn[cpu-shape1-add] PASSED          [ 39%]
tests/hw3/test_ndarray.py::test_ewise_fn[cpu-shape1-subtract] PASSED     [ 43%]
tests/hw3/test_ndarray.py::test_ewise_fn[cpu-shape1-equal] PASSED        [ 47%]
tests/hw3/test_ndarray.py::test_ewise_fn[cpu-shape1-greater_than] PASSED [ 52%]
tests/hw3/test_ndarray.py::test_ewise_max[cpu-shape0] PASSED             [ 56%]
tests/hw3/test_ndarray.py::test_ewise_max[cpu-shape1] PASSED             [ 60%]
tests/hw3/test_ndarray.py::test_scalar_mul[cpu] PASSED                   [ 65%]
tests/hw3/test_ndarray.py::test_scalar_div[cpu] PASSED                   [ 69%]
tests/hw3/test_ndarray.py::test_scalar_power[cpu] PASSED                 [ 73%]
tests/hw3/test_ndarray.py::test_scalar_maximum[cpu] PASSED               [ 78%]
tests/hw3/test_ndarray.py::test_scalar_eq[cpu] PASSED                    [ 82%]
tests/hw3/test_ndarray.py::test_scalar_ge[cpu] PASSED                    [ 86%]
tests/hw3/test_ndarray.py::test_ewise_log[cpu] PASSED                    [ 91%]
tests/hw3/test_ndarray.py::test_ewise_exp[cpu] PASSED                    [ 95%]
tests/hw3/test_ndarray.py::test_ewise_tanh[cpu] PASSED                   [100%]

=============================== warnings summary ===============================
tests/hw3/test_ndarray.py::test_scalar_power[cpu]
  /content/drive/MyDrive/10714/hw3/tests/hw3/test_ndarray.py:390: RuntimeWarning: invalid value encountered in power
    np.power(A, 0.5), (B**0.5).numpy(), atol=1e-5, rtol=1e-5

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================ 23 passed, 113 deselected, 1 warning in 0.54s =================
```

## Part 4: CPU Backend - Reductions

```C++
void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  if (reduce_size == 0) return;
  // 使用 OpenMP 加速
  if(out->size > 1){
    // 情况 A：有很多个归约任务
    // 在外层并行，每个线程负责一行，效率最高
    #pragma omp parallel for
    for(int i=0; i<out->size; i++){
      scalar_t value=a.ptr[i*reduce_size];
      for(int j=1; j<reduce_size; j++){
        value = std::max(value, a.ptr[i*reduce_size+j]);
      }
      out->ptr[i] = value;
    }
  }else{
    // 情况 B：只有一个巨大的归约任务
    // 在内层并行，使用 reduction 关键字让多个线程分担同一个任务
    scalar_t value=a.ptr[0];
    #pragma omp parallel for reduction(max:value)
    for(int j=1; j<reduce_size; j++){
      value = std::max(value, a.ptr[j]);
    }
    out->ptr[0] = value;
  }
  /// END SOLUTION
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

  /// BEGIN SOLUTION
  // 使用 OpenMP 加速
  if(out->size > 1){
    // 情况 A：有很多个归约任务（比如对矩阵的每一行求和）
    // 在外层并行，每个线程负责一行，效率最高
    #pragma omp parallel for
    for(int i=0; i<out->size; i++){
      scalar_t value=0;
      for(int j=0; j<reduce_size; j++){
        value += a.ptr[i*reduce_size+j];
      }
      out->ptr[i] = value;
    }
  }else{
    // 情况 B：只有一个巨大的归约任务（比如全局求和）
    // 在内层并行，使用 reduction 关键字让多个线程分担同一个累加任务
    scalar_t value=0;
    #pragma omp parallel for reduction(+:value)
    for(int j=0; j<reduce_size; j++){
      value += a.ptr[j];
    }
    out->ptr[0] = value;
  }
  /// END SOLUTION
}
```

```bash
!make
!python3 -m pytest -v -k "reduce and cpu"
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Found pybind11: /usr/local/lib/python3.12/dist-packages/pybind11/include (found version "3.0.3")
CUDA_TOOLKIT_ROOT_DIR not found or specified
-- Could NOT find CUDA (missing: CUDA_TOOLKIT_ROOT_DIR CUDA_NVCC_EXECUTABLE CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY) 
-- Configuring done (0.4s)
-- Generating done (0.3s)
-- Build files have been written to: /content/drive/MyDrive/10714/hw3/build
make[1]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[2]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[-50%] Building CXX object CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o
[  0%] Linking CXX shared module /content/drive/MyDrive/10714/hw3/python/needle/backend_ndarray/ndarray_backend_cpu.cpython-312-x86_64-linux-gnu.so
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[  0%] Built target ndarray_backend_cpu
make[2]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[1]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw3
plugins: typeguard-4.5.1, langsmith-0.7.23, anyio-4.13.0
collected 136 items / 128 deselected / 8 selected                              

tests/hw3/test_ndarray.py::test_reduce_sum[params0-cpu] PASSED           [ 12%]
tests/hw3/test_ndarray.py::test_reduce_sum[params1-cpu] PASSED           [ 25%]
tests/hw3/test_ndarray.py::test_reduce_sum[params2-cpu] PASSED           [ 37%]
tests/hw3/test_ndarray.py::test_reduce_sum[params3-cpu] PASSED           [ 50%]
tests/hw3/test_ndarray.py::test_reduce_max[params0-cpu] PASSED           [ 62%]
tests/hw3/test_ndarray.py::test_reduce_max[params1-cpu] PASSED           [ 75%]
tests/hw3/test_ndarray.py::test_reduce_max[params2-cpu] PASSED           [ 87%]
tests/hw3/test_ndarray.py::test_reduce_max[params3-cpu] PASSED           [100%]

====================== 8 passed, 128 deselected in 0.47s =======================
```

## Part 5: CPU Backend - Matrix multiplication

**改用** **i -> k -> j** **顺序**  
	**内存连续性**：最内层循环 j 访问的是 b[k*TILE + j] 和 out[i*TILE + j]，这两个都是连续的内存地址  
	**向量化 (SIMD)**：由于访问是连续的，编译器能更容易地生成 AVX/SSE 指令，一次处理 8 个浮点数  
	**寄存器复用**：a[i*TILE+k] 被提到了 j 循环外面，只需读取一次寄存器  

```C++
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
  Fill(out, 0);
  // 使用 OpenMP 加速
  #pragma omp parallel for
  for(int i=0; i<m; i++){
    for(int k=0; k<n; k++){
      // s 在 j 循环中是常量
      scalar_t s = a.ptr[i*n+k];
      for(int j=0; j<p; j++){
        // 此时 out 和 b 的访问都是连续的 (stride 1)！
        out->ptr[i*p+j] += s*b.ptr[k*p+j];
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
  for(int i=0; i<TILE; i++){
    for(int k=0; k<TILE; k++){
      float s = a[i*TILE+k];
      for(int j=0; j<TILE; j++){
        out[i*TILE+j] += s*b[k*TILE+j];
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
  Fill(out, 0);
  for(int i=0; i<m/TILE; i++){
    for(int j=0; j<p/TILE; j++){
      for(int k=0; k<n/TILE; k++){
        AlignedDot(a.ptr+i*n*TILE+k*TILE*TILE, b.ptr+k*p*TILE+j*TILE*TILE, out->ptr+i*p*TILE+j*TILE*TILE);
      }
    }
  }
  /// END SOLUTION
}
```

```bash
!export CXX=/usr/bin/clang++ && make
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Found pybind11: /usr/local/lib/python3.12/dist-packages/pybind11/include (found version "3.0.3")
CUDA_TOOLKIT_ROOT_DIR not found or specified
-- Could NOT find CUDA (missing: CUDA_TOOLKIT_ROOT_DIR CUDA_NVCC_EXECUTABLE CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY) 
-- Configuring done (0.3s)
-- Generating done (0.3s)
-- Build files have been written to: /content/drive/MyDrive/10714/hw3/build
make[1]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[2]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[-50%] Building CXX object CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o
[  0%] Linking CXX shared module /content/drive/MyDrive/10714/hw3/python/needle/backend_ndarray/ndarray_backend_cpu.cpython-312-x86_64-linux-gnu.so
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[  0%] Built target ndarray_backend_cpu
make[2]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[1]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
```

```bash
!python3 -m pytest -v -k "matmul and cpu"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw3
plugins: typeguard-4.5.1, langsmith-0.7.23, anyio-4.13.0
collected 136 items / 126 deselected / 10 selected                             

tests/hw3/test_ndarray.py::test_matmul[16-16-16-cpu] PASSED              [ 10%]
tests/hw3/test_ndarray.py::test_matmul[8-8-8-cpu] PASSED                 [ 20%]
tests/hw3/test_ndarray.py::test_matmul[1-2-3-cpu] PASSED                 [ 30%]
tests/hw3/test_ndarray.py::test_matmul[3-4-5-cpu] PASSED                 [ 40%]
tests/hw3/test_ndarray.py::test_matmul[5-4-3-cpu] PASSED                 [ 50%]
tests/hw3/test_ndarray.py::test_matmul[64-64-64-cpu] PASSED              [ 60%]
tests/hw3/test_ndarray.py::test_matmul[72-72-72-cpu] PASSED              [ 70%]
tests/hw3/test_ndarray.py::test_matmul[72-73-74-cpu] PASSED              [ 80%]
tests/hw3/test_ndarray.py::test_matmul[74-73-72-cpu] PASSED              [ 90%]
tests/hw3/test_ndarray.py::test_matmul[128-128-128-cpu] PASSED           [100%]

====================== 10 passed, 126 deselected in 0.45s ======================
```

## Part 6: CUDA Backend - Compact and setitem

```C++
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
```

```bash
!make
!python3 -m pytest -v -k "(compact or setitem) and cuda"
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Found pybind11: /usr/local/lib/python3.12/dist-packages/pybind11/include (found version "3.0.3")
-- Found cuda, building cuda backend
Thu Apr  9 10:15:17 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   44C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
-- Autodetected CUDA architecture(s):  7.5
-- Configuring done (2.4s)
-- Generating done (5.2s)
-- Build files have been written to: /content/drive/MyDrive/10714/hw3/build
make[1]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[2]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[-25%] Building CXX object CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o
[  0%] Linking CXX shared module /content/drive/MyDrive/10714/hw3/python/needle/backend_ndarray/ndarray_backend_cpu.cpython-312-x86_64-linux-gnu.so
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[  0%] Built target ndarray_backend_cpu
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[ 25%] Building NVCC (Device) object CMakeFiles/ndarray_backend_cuda.dir/src/ndarray_backend_cuda_generated_ndarray_backend_cuda.cu.o
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[ 50%] Linking CXX shared module /content/drive/MyDrive/10714/hw3/python/needle/backend_ndarray/ndarray_backend_cuda.cpython-312-x86_64-linux-gnu.so
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[ 50%] Built target ndarray_backend_cuda
make[2]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[1]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw3
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 136 items / 121 deselected / 15 selected                             

tests/hw3/test_ndarray.py::test_compact[cuda-transpose] PASSED           [  6%]
tests/hw3/test_ndarray.py::test_compact[cuda-broadcast_to] PASSED        [ 13%]
tests/hw3/test_ndarray.py::test_compact[cuda-reshape1] PASSED            [ 20%]
tests/hw3/test_ndarray.py::test_compact[cuda-reshape2] PASSED            [ 26%]
tests/hw3/test_ndarray.py::test_compact[cuda-reshape3] PASSED            [ 33%]
tests/hw3/test_ndarray.py::test_compact[cuda-getitem1] PASSED            [ 40%]
tests/hw3/test_ndarray.py::test_compact[cuda-getitem2] PASSED            [ 46%]
tests/hw3/test_ndarray.py::test_compact[cuda-transposegetitem] PASSED    [ 53%]
tests/hw3/test_ndarray.py::test_setitem_ewise[cuda-params0] PASSED       [ 60%]
tests/hw3/test_ndarray.py::test_setitem_ewise[cuda-params1] PASSED       [ 66%]
tests/hw3/test_ndarray.py::test_setitem_ewise[cuda-params2] PASSED       [ 73%]
tests/hw3/test_ndarray.py::test_setitem_scalar[cuda-params0] PASSED      [ 80%]
tests/hw3/test_ndarray.py::test_setitem_scalar[cuda-params1] PASSED      [ 86%]
tests/hw3/test_ndarray.py::test_setitem_scalar[cuda-params2] PASSED      [ 93%]
tests/hw3/test_ndarray.py::test_setitem_scalar[cuda-params3] PASSED      [100%]

====================== 15 passed, 121 deselected in 8.17s ======================
```

## Part 7: CUDA Backend - Elementwise and scalar operations

```C++
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
```

```bash
!make
!python3 -m pytest -v -k "(ewise_fn or ewise_max or log or exp or tanh or (scalar and not setitem)) and cuda"
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Found pybind11: /usr/local/lib/python3.12/dist-packages/pybind11/include (found version "3.0.3")
-- Found cuda, building cuda backend
Thu Apr  9 12:03:51 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   42C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
-- Autodetected CUDA architecture(s):  7.5
-- Configuring done (0.5s)
-- Generating done (0.4s)
-- Build files have been written to: /content/drive/MyDrive/10714/hw3/build
make[1]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[2]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[  0%] Built target ndarray_backend_cpu
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[ 25%] Building NVCC (Device) object CMakeFiles/ndarray_backend_cuda.dir/src/ndarray_backend_cuda_generated_ndarray_backend_cuda.cu.o
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[ 50%] Linking CXX shared module /content/drive/MyDrive/10714/hw3/python/needle/backend_ndarray/ndarray_backend_cuda.cpython-312-x86_64-linux-gnu.so
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[ 50%] Built target ndarray_backend_cuda
make[2]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[1]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw3
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 136 items / 113 deselected / 23 selected                             

tests/hw3/test_ndarray.py::test_ewise_fn[cuda-shape0-multiply] PASSED    [  4%]
tests/hw3/test_ndarray.py::test_ewise_fn[cuda-shape0-divide] PASSED      [  8%]
tests/hw3/test_ndarray.py::test_ewise_fn[cuda-shape0-add] PASSED         [ 13%]
tests/hw3/test_ndarray.py::test_ewise_fn[cuda-shape0-subtract] PASSED    [ 17%]
tests/hw3/test_ndarray.py::test_ewise_fn[cuda-shape0-equal] PASSED       [ 21%]
tests/hw3/test_ndarray.py::test_ewise_fn[cuda-shape0-greater_than] PASSED [ 26%]
tests/hw3/test_ndarray.py::test_ewise_fn[cuda-shape1-multiply] PASSED    [ 30%]
tests/hw3/test_ndarray.py::test_ewise_fn[cuda-shape1-divide] PASSED      [ 34%]
tests/hw3/test_ndarray.py::test_ewise_fn[cuda-shape1-add] PASSED         [ 39%]
tests/hw3/test_ndarray.py::test_ewise_fn[cuda-shape1-subtract] PASSED    [ 43%]
tests/hw3/test_ndarray.py::test_ewise_fn[cuda-shape1-equal] PASSED       [ 47%]
tests/hw3/test_ndarray.py::test_ewise_fn[cuda-shape1-greater_than] PASSED [ 52%]
tests/hw3/test_ndarray.py::test_ewise_max[cuda-shape0] PASSED            [ 56%]
tests/hw3/test_ndarray.py::test_ewise_max[cuda-shape1] PASSED            [ 60%]
tests/hw3/test_ndarray.py::test_scalar_mul[cuda] PASSED                  [ 65%]
tests/hw3/test_ndarray.py::test_scalar_div[cuda] PASSED                  [ 69%]
tests/hw3/test_ndarray.py::test_scalar_power[cuda] PASSED                [ 73%]
tests/hw3/test_ndarray.py::test_scalar_maximum[cuda] PASSED              [ 78%]
tests/hw3/test_ndarray.py::test_scalar_eq[cuda] PASSED                   [ 82%]
tests/hw3/test_ndarray.py::test_scalar_ge[cuda] PASSED                   [ 86%]
tests/hw3/test_ndarray.py::test_ewise_log[cuda] PASSED                   [ 91%]
tests/hw3/test_ndarray.py::test_ewise_exp[cuda] PASSED                   [ 95%]
tests/hw3/test_ndarray.py::test_ewise_tanh[cuda] PASSED                  [100%]

=============================== warnings summary ===============================
tests/hw3/test_ndarray.py::test_scalar_power[cuda]
  /content/drive/MyDrive/10714/hw3/tests/hw3/test_ndarray.py:390: RuntimeWarning: invalid value encountered in power
    np.power(A, 0.5), (B**0.5).numpy(), atol=1e-5, rtol=1e-5

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================ 23 passed, 113 deselected, 1 warning in 0.79s =================
```

## Part 8: CUDA Backend - Reductions

```C++
////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
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
```

```bash
!make
!python3 -m pytest -v -k "reduce and cuda"
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Found pybind11: /usr/local/lib/python3.12/dist-packages/pybind11/include (found version "3.0.3")
-- Found cuda, building cuda backend
Thu Apr  9 14:33:47 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   37C    P8             15W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
-- Autodetected CUDA architecture(s):  7.5
-- Configuring done (0.6s)
-- Generating done (0.4s)
-- Build files have been written to: /content/drive/MyDrive/10714/hw3/build
make[1]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[2]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[  0%] Built target ndarray_backend_cpu
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[ 50%] Built target ndarray_backend_cuda
make[2]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[1]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw3
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 136 items / 128 deselected / 8 selected                              

tests/hw3/test_ndarray.py::test_reduce_sum[params0-cuda] PASSED          [ 12%]
tests/hw3/test_ndarray.py::test_reduce_sum[params1-cuda] PASSED          [ 25%]
tests/hw3/test_ndarray.py::test_reduce_sum[params2-cuda] PASSED          [ 37%]
tests/hw3/test_ndarray.py::test_reduce_sum[params3-cuda] PASSED          [ 50%]
tests/hw3/test_ndarray.py::test_reduce_max[params0-cuda] PASSED          [ 62%]
tests/hw3/test_ndarray.py::test_reduce_max[params1-cuda] PASSED          [ 75%]
tests/hw3/test_ndarray.py::test_reduce_max[params2-cuda] PASSED          [ 87%]
tests/hw3/test_ndarray.py::test_reduce_max[params3-cuda] PASSED          [100%]

====================== 8 passed, 128 deselected in 0.64s =======================
```

**优化方案：Block-level 共享内存归约**  
	一个 Block 里的所有线程共同协作来处理一行的归约  
	1.每个线程先通过循环，跳跃式地读取全局内存（保证内存合并读取），计算一个局部和  
	2.将局部和存入 \__shared__ 内存  
	3.在 shared 内存中进行树状求和 (Tree Reduction)  
	4.最后由该 Block 的线程 0 将总和写入 out  

```C++
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
```

```bash
!make
!python3 -m pytest -v -k "reduce and cuda"
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Found pybind11: /usr/local/lib/python3.12/dist-packages/pybind11/include (found version "3.0.3")
-- Found cuda, building cuda backend
Thu Apr  9 15:54:39 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   39C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
-- Autodetected CUDA architecture(s):  7.5
-- Configuring done (2.5s)
-- Generating done (4.4s)
-- Build files have been written to: /content/drive/MyDrive/10714/hw3/build
make[1]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[2]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[-25%] Building CXX object CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o
[  0%] Linking CXX shared module /content/drive/MyDrive/10714/hw3/python/needle/backend_ndarray/ndarray_backend_cpu.cpython-312-x86_64-linux-gnu.so
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[  0%] Built target ndarray_backend_cpu
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[ 25%] Building NVCC (Device) object CMakeFiles/ndarray_backend_cuda.dir/src/ndarray_backend_cuda_generated_ndarray_backend_cuda.cu.o
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[ 50%] Linking CXX shared module /content/drive/MyDrive/10714/hw3/python/needle/backend_ndarray/ndarray_backend_cuda.cpython-312-x86_64-linux-gnu.so
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[ 50%] Built target ndarray_backend_cuda
make[2]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[1]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw3
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 136 items / 128 deselected / 8 selected                              

tests/hw3/test_ndarray.py::test_reduce_sum[params0-cuda] PASSED          [ 12%]
tests/hw3/test_ndarray.py::test_reduce_sum[params1-cuda] PASSED          [ 25%]
tests/hw3/test_ndarray.py::test_reduce_sum[params2-cuda] PASSED          [ 37%]
tests/hw3/test_ndarray.py::test_reduce_sum[params3-cuda] PASSED          [ 50%]
tests/hw3/test_ndarray.py::test_reduce_max[params0-cuda] PASSED          [ 62%]
tests/hw3/test_ndarray.py::test_reduce_max[params1-cuda] PASSED          [ 75%]
tests/hw3/test_ndarray.py::test_reduce_max[params2-cuda] PASSED          [ 87%]
tests/hw3/test_ndarray.py::test_reduce_max[params3-cuda] PASSED          [100%]

====================== 8 passed, 128 deselected in 7.64s =======================
```

**进阶优化：使用 Warp Shuffle（寄存器直接通信）**  
	现代 NVIDIA GPU 支持 __shfl_down_sync 指令，它允许线程直接从另一个线程的寄存器中读数据，完全绕过共享内存（Shared Memory）。这能减少 L1 缓存压力并提升速度  

```C++
// 专门处理最后 32 个元素的 Warp 归约
__device__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void ReduceSumKernelOptimized(const scalar_t* a, scalar_t* out, size_t reduce_size) {
    size_t blockid = blockIdx.x;
    size_t threadid = threadIdx.x;
    
    // 动态分配或固定大小的共享内存
    extern __shared__ scalar_t sdata[];

    // 1. Grid-stride loop + 边界检查
    scalar_t psum = 0;
    for (size_t i = threadid; i < reduce_size; i += blockDim.x) {
        psum += a[blockid * reduce_size + i];
    }
    sdata[threadid] = psum;
    __syncthreads();

    // 2. 树状归约
    for (int i = blockDim.x / 2; i > 32; i >>= 1) {
        if (threadid < i) {
            sdata[threadid] += sdata[threadid + i];
        }
        __syncthreads();
    }

    // 3. 最后 32 个元素进入 Warp 归约，无需 syncthreads
    if (threadid < 32) {
        scalar_t val = sdata[threadid];
        // 展开循环或使用 shuffle
        val = warpReduceSum(val);
        if (threadid == 0) out[blockid] = val;
    }
}
```

**还能优化的点 (面向竞赛/生产)**  
	1.**完全展开循环 (Manual Unrolling)**：如果 BASE_THREAD_NUM 是固定的（如 256），可以使用 if(blockDim.x >= 256) 配合模板来手动展开树状循环，消除循环开销  
	2.**多元素读取 (Vectorized Memory Access)**：让每个线程使用 float4 类型一次读取 4 个 scalar_t。这能更有效地利用 GPU 显存控制器的带宽（Memory Throughtput）  
	3.**原子操作 (Atomic Add)**：如果 out->size 很大，目前的实现是给每个输出启动一个 Block。如果 reduce_size 极大，可以让多个 Block 协作处理一个输出，最后用 atomicAdd 汇总到 out[blockid]  

## Part 9: CUDA Backend - Matrix multiplication

```C++
// 每个线程负责输出矩阵 C 的一个点 (row, col)
__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t N, size_t P){
  size_t gidx = blockIdx.x*blockDim.x+threadIdx.x;
  size_t gidy = blockIdx.y*blockDim.y+threadIdx.y;

  __shared__ scalar_t sa[TILE][TILE];
  __shared__ scalar_t sb[TILE][TILE];

  scalar_t acc = 0;

  // 沿 N 维度以 TILE 为单位滑动
  for(size_t k_block=0; k_block < (N+TILE-1)/TILE; k_block++){
    if(k_block*TILE+threadIdx.y<N&&gidx<M){
      sa[threadIdx.x][threadIdx.y] = a[gidx*N+k_block*TILE+threadIdx.y];
    }else{
      // 超出范围的点赋值为0
      sa[threadIdx.x][threadIdx.y] = 0;
    }
    if(k_block*TILE+threadIdx.x<N&&gidy<P){
      sb[threadIdx.x][threadIdx.y] = b[(k_block*TILE+threadIdx.x)*P+gidy];
    }else{
      sb[threadIdx.x][threadIdx.y] = 0;
    }
    __syncthreads();
    for(int k=0; k<TILE; k++){
      acc += sa[threadIdx.x][k]*sb[k][threadIdx.y];
    }
    // 等待所有线程计算完成，才能进入下一轮加载
    __syncthreads();
  }
  if(gidx<M && gidy<P) out[gidx*P+gidy] = acc;
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
  size_t num_blocks_x = (M + TILE - 1) / TILE; 
  size_t num_blocks_y = (P + TILE - 1) / TILE;
  dim3 grid = dim3(num_blocks_x, num_blocks_y, 1);
  dim3 block = dim3(TILE, TILE, 1);
  Fill(out, 0);
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}
```

```bash
!make
!python3 -m pytest -v -k "matmul and cuda"
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Found pybind11: /usr/local/lib/python3.12/dist-packages/pybind11/include (found version "3.0.3")
-- Found cuda, building cuda backend
Fri Apr 10 03:49:22 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   42C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
-- Autodetected CUDA architecture(s):  7.5
-- Configuring done (5.5s)
-- Generating done (12.2s)
-- Build files have been written to: /content/drive/MyDrive/10714/hw3/build
make[1]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[2]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[-25%] Building CXX object CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o
[  0%] Linking CXX shared module /content/drive/MyDrive/10714/hw3/python/needle/backend_ndarray/ndarray_backend_cpu.cpython-312-x86_64-linux-gnu.so
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[  0%] Built target ndarray_backend_cpu
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[ 25%] Building NVCC (Device) object CMakeFiles/ndarray_backend_cuda.dir/src/ndarray_backend_cuda_generated_ndarray_backend_cuda.cu.o
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[ 50%] Linking CXX shared module /content/drive/MyDrive/10714/hw3/python/needle/backend_ndarray/ndarray_backend_cuda.cpython-312-x86_64-linux-gnu.so
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[ 50%] Built target ndarray_backend_cuda
make[2]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[1]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw3
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 136 items / 126 deselected / 10 selected                             

tests/hw3/test_ndarray.py::test_matmul[16-16-16-cuda] PASSED             [ 10%]
tests/hw3/test_ndarray.py::test_matmul[8-8-8-cuda] PASSED                [ 20%]
tests/hw3/test_ndarray.py::test_matmul[1-2-3-cuda] PASSED                [ 30%]
tests/hw3/test_ndarray.py::test_matmul[3-4-5-cuda] PASSED                [ 40%]
tests/hw3/test_ndarray.py::test_matmul[5-4-3-cuda] PASSED                [ 50%]
tests/hw3/test_ndarray.py::test_matmul[64-64-64-cuda] PASSED             [ 60%]
tests/hw3/test_ndarray.py::test_matmul[72-72-72-cuda] PASSED             [ 70%]
tests/hw3/test_ndarray.py::test_matmul[72-73-74-cuda] PASSED             [ 80%]
tests/hw3/test_ndarray.py::test_matmul[74-73-72-cuda] PASSED             [ 90%]
tests/hw3/test_ndarray.py::test_matmul[128-128-128-cuda] PASSED          [100%]

===================== 10 passed, 126 deselected in 22.05s ======================
```

**核心改进点：访存合并 (Memory Coalescing)**  
	在 CUDA 中，threadIdx.x 是最内层的维度。为了实现访存合并，threadIdx.x 应该对应于内存中连续的维度（即矩阵的列）  
	否则，写入 out[gidx * P + gidy] 时，同一个 Warp 里的线程（threadIdx.x 不同）访问的是不同行的同一列。物理地址跨度很大（差了一个 P），这会导致非合并访存，写入速度会慢 5-10 倍  

```c++
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
```

```bash
!make
!python3 -m pytest -v -k "matmul and cuda"
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Found pybind11: /usr/local/lib/python3.12/dist-packages/pybind11/include (found version "3.0.3")
-- Found cuda, building cuda backend
Fri Apr 10 04:09:21 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   43C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
-- Autodetected CUDA architecture(s):  7.5
-- Configuring done (0.6s)
-- Generating done (0.4s)
-- Build files have been written to: /content/drive/MyDrive/10714/hw3/build
make[1]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[2]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[  0%] Built target ndarray_backend_cpu
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[ 25%] Building NVCC (Device) object CMakeFiles/ndarray_backend_cuda.dir/src/ndarray_backend_cuda_generated_ndarray_backend_cuda.cu.o
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[3]: Entering directory '/content/drive/MyDrive/10714/hw3/build'
[ 50%] Linking CXX shared module /content/drive/MyDrive/10714/hw3/python/needle/backend_ndarray/ndarray_backend_cuda.cpython-312-x86_64-linux-gnu.so
make[3]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
[ 50%] Built target ndarray_backend_cuda
make[2]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
make[1]: Leaving directory '/content/drive/MyDrive/10714/hw3/build'
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw3
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 136 items / 126 deselected / 10 selected                             

tests/hw3/test_ndarray.py::test_matmul[16-16-16-cuda] PASSED             [ 10%]
tests/hw3/test_ndarray.py::test_matmul[8-8-8-cuda] PASSED                [ 20%]
tests/hw3/test_ndarray.py::test_matmul[1-2-3-cuda] PASSED                [ 30%]
tests/hw3/test_ndarray.py::test_matmul[3-4-5-cuda] PASSED                [ 40%]
tests/hw3/test_ndarray.py::test_matmul[5-4-3-cuda] PASSED                [ 50%]
tests/hw3/test_ndarray.py::test_matmul[64-64-64-cuda] PASSED             [ 60%]
tests/hw3/test_ndarray.py::test_matmul[72-72-72-cuda] PASSED             [ 70%]
tests/hw3/test_ndarray.py::test_matmul[72-73-74-cuda] PASSED             [ 80%]
tests/hw3/test_ndarray.py::test_matmul[74-73-72-cuda] PASSED             [ 90%]
tests/hw3/test_ndarray.py::test_matmul[128-128-128-cuda] PASSED          [100%]

====================== 10 passed, 126 deselected in 0.78s ======================
```

**高级优化：寄存器分块 (Register Tiling / Thread-level Tiling)**  
	1 个线程负责计算一个 **`V×V`（如 `4×4` 或 `8×8`）** 的输出小矩阵  
	在原来的 for(int k=0; k<TILE; k++) 循环中，每计算一次乘加，都要从 Shared Memory 读两次数据。如果一个线程算 4 个元素，可以把 sa 的值存在寄存器里，重复利用它去乘 sb 的 4 个不同元素。这样，访存次数减少了，计算指令的占比（计算密度）提高了  
	原来版本处理一个位置要读 Shared Memory 2N 次，现在只要读 2N/V 次

```C++
#define V 4 // 每个线程处理 4x4 的块

__global__ void MatmulKernelOptimized(const scalar_t* A, const scalar_t* B, scalar_t* C, size_t M, size_t N, size_t P) {
    // 此时 Block 的维度应该是 (TILE/V, TILE/V)
    size_t row_start = blockIdx.y * TILE + threadIdx.y * V;
    size_t col_start = blockIdx.x * TILE + threadIdx.x * V;

    __shared__ scalar_t sA[TILE][TILE];
    __shared__ scalar_t sB[TILE][TILE];

    // 寄存器缓存输出块
    scalar_t acc[V][V] = {0};

    for (size_t k_block = 0; k_block < (N + TILE - 1) / TILE; k_block++) {
        // 1. 协作搬运数据（每个线程需要搬运 V*V 个点，或者重新分配搬运逻辑）
        // ... 此处省略复杂的搬运逻辑，确保内存合并读取 ...
        
        __syncthreads();

        // 2. 寄存器级别计算
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            // 预读取到寄存器
            scalar_t a_val[V];
            scalar_t b_val[V];
            for(int i=0; i<V; i++) a_val[i] = sA[threadIdx.y * V + i][k];
            for(int i=0; i<V; i++) b_val[i] = sB[k][threadIdx.x * V + i];

            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    acc[i][j] += a_val[i] * b_val[j];
                }
            }
        }
        __syncthreads();
    }

    // 3. 写回 C
    // ... 将 acc[V][V] 写回全局内存 ...
}
```

**性能细节优化**  
	消除共享内存 Bank Conflict  
		如果 TILE 是 32，sa\[threadIdx.x][threadIdx.y] 的访问可能会触发 Bank Conflict（因为同一列的元素正好落在同一个 Bank）  
		**技巧**：定义共享内存时多加一列：\__shared__ scalar_t sa\[TILE][TILE + 1];。这个简单的“填充”可以错开内存地址，消除冲突  
	使用循环展开 (Pragma Unroll)  
		在内层计算循环前加上 #pragma unroll，编译器会尝试展开循环，减少分支跳转开销  

