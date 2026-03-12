# Assignment 3: A Simple CUDA Renderer

**作业原址**：https://github.com/stanford-cs149/asst3  
非常感谢老师的付出和开源，以下是作业介绍和我的实现(特别感谢 Google AI Studio 提供远程指导😝)  

![My Image](handout/teaser.jpg?raw=true)

## Overview

In this assignment you will write a parallel renderer in CUDA that draws colored circles.
While this renderer is very simple, parallelizing the renderer will require you to design and implement data structures
that can be efficiently constructed and manipulated in parallel. This is a challenging
assignment so you are advised to start early. **Seriously, you are advised to start early.** Good luck!

## Environment Setup

1. You will collect results (i.e. run performance tests) for this assignment on GPU-enabled VMs on Amazon Web Services (AWS). Please follow the instructions in [cloud_readme.md](cloud_readme.md) for setting up a machine to run the assignment.

2. Download the Assignment starter code from the course Github using:

`git clone https://github.com/stanford-cs149/asst3`

The CUDA C programmer's guide [PDF version](http://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) or [web version](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) is an excellent reference for learning how to program in CUDA. There are a wealth of CUDA tutorials and SDK examples on the web (just Google!) and on the [NVIDIA developer site](http://docs.nvidia.com/cuda/). In particular, you may enjoy the free Udacity course [Introduction to Parallel Programming in CUDA](https://www.udacity.com/blog/2014/01/update-on-udacity-cs344-intro-to.html).  
学习文档 [Contents — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html)  

Table 21 in the [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities) is a handy reference for the maximum number of CUDA threads per thread block, size of thread block, shared memory, etc for the NVIDIA T4 GPUs you will used in this assignment. NVIDIA T4 GPUs support CUDA compute capability 7.5.  
	电脑配置：
		处理器	Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz   2.50 GHz	内核：4	逻辑处理器：8  
		显卡	NVIDIA GeForce GTX 1660 Ti (6 GB)  
	查询GPU计算能力为7.5 [CUDA GPU 计算能力 | NVIDIA 开发者](https://developer.nvidia.cn/cuda/gpus)  

For C++ questions (like what does the _virtual_ keyword mean), the [C++ Super-FAQ](https://isocpp.org/faq) is a great resource that explains things in a way that's detailed yet easy to understand (unlike a lot of C++ resources), and was co-written by Bjarne Stroustrup, the creator of C++!

`make`Part 3 时，报错`fatal error: GL/glut.h: No such file or directory`  
使用`sudo apt install -y freeglut3-dev`  

版本问题，ref参考程序无法运行，按照asst1 环境配置安装Ubuntu24.04和Cuda12.4  
[nvidia - Installing CUDA on Ubuntu 23.10 - libt5info not installable - Ask Ubuntu](https://askubuntu.com/questions/1491254/installing-cuda-on-ubuntu-23-10-libt5info-not-installable)  

## Part 1: CUDA Warm-Up 1: SAXPY (5 pts)

To gain a bit of practice writing CUDA programs your warm-up task is to re-implement the SAXPY function
from Assignment 1 in CUDA. Starter code for this part of the assignment is located in the `/saxpy` directory
of the assignment repository. You can build and run the saxpy CUDA program by calling `make` and `./cudaSaxpy` in the `/saxpy` directory.

Please finish off the implementation of SAXPY in the function `saxpyCuda` in `saxpy.cu`. You will need to allocate device global memory arrays and copy the contents of the host input arrays `X`, `Y`, and `result` into CUDA device memory prior to performing the computation. After the CUDA computation is complete, the result must be copied back into host memory. Please see the definition of `cudaMemcpy` function in Section 3.2.2 of the Programmer's Guide (web version), or take a look at the helpful tutorial pointed to in the assignment starter code.

As part of your implementation, add timers around the CUDA kernel invocation in `saxpyCuda`. After your additions, your program should time two executions:

- The provided starter code contains timers that measure **the entire process** of copying data to the GPU, running the kernel, and copying data back to the CPU.

- You should also insert timers the measure _only the time taken to run the kernel_. (They should not include the time of CPU-to-GPU data transfer or transfer of results from the GPU back to the CPU.)

**When adding your timing code in the latter case, you'll need to be careful:** By defult a CUDA kernel's execution on the GPU is _asynchronous_ with the main application thread running on the CPU. For example, if you write code that looks like this:

```C++
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
double endTime = CycleTimer::currentSeconds();
```

You'll measure a kernel execution time that seems amazingly fast! (Because you are only timing the cost of the API call itself, not the cost of actually executing the resulting computation on the GPU.

Therefore, you will want to place a call to `cudaDeviceSynchronize()` following the
kernel call to wait for completion of all CUDA work on the GPU. This call to `cudaDeviceSynchronize()` returns when all prior CUDA work on the GPU has completed. Note that `cudaDeviceSynchronize()` is not necessary after the `cudaMemcpy()` to ensure the memory transfer to the GPU is complete, since `cudaMempy()` is synchronous under the conditions we are using it. (For those that wish to know more, see [this documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync).)

```C++
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
cudaDeviceSynchronize();
double endTime = CycleTimer::currentSeconds();
```

Note that in your measurements that include the time to transfer to and from the CPU, a call to `cudaDeviceSynchronize()` **is not** necessary before the final timer (after your call to `cudaMemcopy()` that returns data to the CPU) because `cudaMemcpy()` will not return to the calling thread until after the copy is complete.

**Question 1.** What performance do you observe compared to the sequential CPU-based implementation of
SAXPY (recall your results from saxpy on Program 5 from Assignment 1)?

```C++
// saxpyCuda --
//
// This function is regular C code running on the CPU.  It allocates
// memory on the GPU using CUDA API functions, uses CUDA API functions
// to transfer data from the CPU's memory address space to GPU memory
// address space, and launches the CUDA kernel function on the GPU.
void saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    // must read both input arrays (xarray and yarray) and write to
    // output array (resultarray)
    int totalBytes = sizeof(float) * 3 * N;

    // compute number of blocks and threads per block.  In this
    // application we've hardcoded thread blocks to contain 512 CUDA
    // threads.
    const int threadsPerBlock = 512;

    // Notice the round up here.  The code needs to compute the number
    // of threads blocks needed such that there is one thread per
    // element of the arrays.  This code is written to work for values
    // of N that are not multiples of threadPerBlock.
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // These are pointers that will be pointers to memory allocated
    // *one the GPU*.  You should allocate these pointers via
    // cudaMalloc.  You can access the resulting buffers from CUDA
    // device kernel code (see the kernel function saxpy_kernel()
    // above) but you cannot access the contents these buffers from
    // this thread. CPU threads cannot issue loads and stores from GPU
    // memory!
    float* device_x = nullptr;
    float* device_y = nullptr;
    float* device_result = nullptr;
    
    //
    // CS149 TODO: allocate device memory buffers on the GPU using cudaMalloc.
    //
    // We highly recommend taking a look at NVIDIA's
    // tutorial, which clearly walks you through the few lines of code
    // you need to write for this part of the assignment:
    //
    // https://devblogs.nvidia.com/easy-introduction-cuda-c-and-c/
    //
    cudaMalloc(&device_x, N*sizeof(float));
    cudaMalloc(&device_y, N*sizeof(float));
    cudaMalloc(&device_result, N*sizeof(float));
    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    //
    // CS149 TODO: copy input arrays to the GPU using cudaMemcpy
    //
    cudaMemcpy(device_x, xarray, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, N*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemset(device_result, 0, N*sizeof(float));
   
    // run CUDA kernel. (notice the <<< >>> brackets indicating a CUDA
    // kernel launch) Execution on the GPU occurs here.
    double runKernelStartTime = CycleTimer::currentSeconds();
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);

    //
    // CS149 TODO: copy result from GPU back to CPU using cudaMemcpy
    //
    cudaDeviceSynchronize();
    double runKernelEndTime = CycleTimer::currentSeconds();
    cudaMemcpy(resultarray, device_result, N, cudaMemcpyDeviceToHost);
    
    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
		errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    double runKernelOverallDuration = runKernelEndTime - runKernelStartTime;
    printf("Effective BW by CUDA saxpy: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, GBPerSec(totalBytes, overallDuration));
    printf("Effective BW by CUDA saxpy(only the time taken to run the kernel): %.3f ms\t\t[%.3f GB/s]\n", 1000.f * runKernelOverallDuration, GBPerSec(totalBytes, runKernelOverallDuration));
    //
    // CS149 TODO: free memory buffers on the GPU using cudaFree
    //
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
}
```

```bash
# results from saxpy on Program 5 from Assignment 1
# ./saxpy
[saxpy ispc]:           [12.155] ms     [24.519] GB/s   [3.291] GFLOPS
[saxpy task ispc]:      [11.480] ms     [25.960] GB/s   [3.484] GFLOPS
                                (1.06x speedup from use of tasks)
```

```bash
# ./cudaSaxpy
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Running 3 timing tests:
Effective BW by CUDA saxpy: 175.380 ms          [6.372 GB/s]
Effective BW by CUDA saxpy(only the time taken to run the kernel): 5.655 ms
[197.641 GB/s]
Effective BW by CUDA saxpy: 172.630 ms          [6.474 GB/s]
Effective BW by CUDA saxpy(only the time taken to run the kernel): 4.865 ms
[229.707 GB/s]
Effective BW by CUDA saxpy: 162.645 ms          [6.871 GB/s]
Effective BW by CUDA saxpy(only the time taken to run the kernel): 4.861 ms
[229.924 GB/s]
```

cuda上运行 saxpy 算上数据传输时间（CPU到GPU 和 GPU到CPU）慢于CPU上运行，不算数据传输时间，快于CPU上运行

**Question 2.** Compare and explain the difference between the results
provided by two sets of timers (timing only the kernel execution vs. timing the entire process of moving data to the GPU and back in addition to the kernel execution). Are the bandwidth values observed _roughly_ consistent with the reported bandwidths available to the different components of the machine? (You should use the web to track down the memory bandwidth of an NVIDIA T4 GPU. Hint: <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf>. The expected bandwidth of memory bus of AWS is 5.3 GB/s, which does not match that of a 16-lane [PCIe 3.0](https://en.wikipedia.org/wiki/PCI_Express). Several factors prevent peak bandwidth, including CPU motherboard chipset performance and whether or not the host CPU memory used as the source of the transfer is “pinned” — the latter allows the GPU to directly access memory without going through virtual memory address translation. If you are interested, you can find more info here: <https://kth.instructure.com/courses/12406/pages/optimizing-host-device-data-communication-i-pinned-host-memory>)
	**两种计时器结果的对比与解释**  
		Kernel 执行时间（Kernel Execution Time）：仅衡量了 GPU 内核函数（Kernel Function） 实际运行的时间。这部分是 SAXPY 核心的计算时间，也就是 GPU 真正进行计算的时间   
		总体时间 (Total Time)：包含了所有操作，包括  
			数据传输时间 (Data Transfer Time)：将数据从 CPU 内存（Host Memory）复制到 GPU 显存（Device Memory），以及将结果复制回 CPU 内存  
			内核启动开销 (Kernel Launch Overhead)：启动 GPU 内核函数，配置 grid/block 等的开销  
			内核执行时间 (Kernel Execution Time)：GPU 内部的计算  
		由于 SAXPY 属于内存受限型 (Memory-bound) 任务，数据传输时间占据了主要部分，而内核计算时间非常短  
	**Bandwidth 数值的合理性分析**  
		CUDA saxpy 的 Effective BW：6.372 GB/s - 6.871 GB/s  
			这是通过测量 SAXPY 程序的整体运行时间（包含数据传输）和计算出的带宽。这个数值代表了 “有效带宽”，也就是程序实际能达到的内存带宽  
		内核的 Effective BW：197.641 GB/s - 229.924 GB/s  
			这个数值是通过测量内核执行时间计算出的带宽。这并非 GPU 显存的实际带宽，而是 GPU 内核内部（例如 SIMD 单元）的计算能力	  
		**一致性分析（与硬件理论带宽的比较）： **  
			GPU 显存带宽 (Global Memory Bandwidth)：  
				根据 NVIDIA 的官方规格, https://www.techpowerup.com/gpu-specs/geforce-gtx-1660-ti.c3364, NVIDIA GeForce GTX 1660 Ti 的显存带宽约为 288 GB/s  
				对比结果：内核带宽结果约在 200 - 230 GB/s，这表明 GPU 内核的计算效率很高，可以有效地利用 GPU 内部的计算资源  
				有效带宽：有效带宽是 6-7 GB/s，说明 瓶颈在数据传输，而不是在 GPU 内核的计算  
			数据传输带宽 (Data Transfer Bandwidth)：  
				通过 GPU-Z 查看物理上限, PCIe 3.0 x16：理论单向峰值约为 15.75 GB/s，实验结果(6.372 GB/s - 6.871 GB/s)远低于 PCle 理论值，可能原因:  
			非钉内存（Pageable vs. Pinned Memory）：  
				默认情况下，C++ 用 malloc 分配的是 Pageable Memory。GPU 无法直接访问它，驱动必须先在后台把数据拷贝到一个临时的“锁页内存”缓冲区，再通过 DMA 传输。这个过程会消耗大量时间，导致带宽减半  
				如果改用 cudaMallocHost（即 **Pinned Memory**），带宽通常能提升  
			PCIe 协议开销：  
				数据包头、校验位、以及指令往返延迟（Latency）会吃掉一部分有效带宽  
			有效带宽结果偏低，因为包含了数据传输，而SAXPY程序计算量少。因此可以推断，数据传输时间占用了大部分时间     

## Part 2: CUDA Warm-Up 2: Parallel Prefix-Sum (10 pts)

Now that you're familiar with the basic structure and layout of CUDA programs, as a second exercise you are asked to come up with parallel implementation of the function `find_repeats` which, given a list of integers `A`, returns a list of all indices `i` for which `A[i] == A[i+1]`.

For example, given the array `{1,2,2,1,1,1,3,5,3,3}`, your program should output the array `{1,3,4,8}`.

#### Exclusive Prefix Sum

We want you to implement `find_repeats` by first implementing parallel exclusive prefix-sum operation.

Exlusive prefix sum takes an array `A` and produces a new array `output` that has, at each index `i`, the sum of all elements up to but not including `A[i]`. For example, given the array `A={1,4,6,8,2}`, the output of exclusive prefix sum `output={0,1,5,11,19}`.

The following "C-like" code is an iterative version of scan. In the pseudocode before, we use `parallel_for` to indicate potentially parallel loops. This is the same algorithm we discussed in class: <https://gfxcourses.stanford.edu/cs149/fall25/lecture/dataparallel/slide_17>

```C++
void exclusive_scan_iterative(int* start, int* end, int* output) {

    int N = end - start;
    memmove(output, start, N*sizeof(int));

    // upsweep phase
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            output[i+two_dplus1-1] += output[i+two_d-1];
        }
    }

    output[N-1] = 0;

    // downsweep phase
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            int t = output[i+two_d-1];
            output[i+two_d-1] = output[i+two_dplus1-1];
            output[i+two_dplus1-1] += t;
        }
    }
}
```

We would like you to use this algorithm to implement a version of parallel prefix sum in CUDA. You must implement `exclusive_scan` function in `scan/scan.cu`. Your implementation will consist of both host and device code. The implementation will require multiple CUDA kernel launches (one for each parallel_for loop in the pseudocode above).

**Note:** In the starter code, the reference solution scan implementation above assumes that the input array's length (`N`) is a power of 2. In the `cudaScan` function, we solve this problem by rounding the input array length to the next power of 2 when allocating the corresponding buffers on the GPU. However, the code only copies back `N` elements from the GPU buffer back to the CPU buffer. This fact should simplify your CUDA implementation.

Compilation produces the binary `cudaScan`. Commandline usage is as follows:

```
Usage: ./cudaScan [options]

Program Options:
  -m  --test <TYPE>      Run specified function on input.  Valid tests are: scan, find_repeats (default: scan)
  -i  --input <NAME>     Run test on given input type. Valid inputs are: ones, random (default: random)
  -n  --arraysize <INT>  Number of elements in arrays
  -t  --thrust           Use Thrust library implementation
  -?  --help             This message
```

#### Implementing "Find Repeats" Using Prefix Sum

Once you have written `exclusive_scan`, implement the function `find_repeats` in `scan/scan.cu`. This will involve writing more device code, in addition to one or more calls to `exclusive_scan()`. Your code should write the list of repeated elements into the provided output pointer (in device memory), and then return the size of the output list.

When calling your `exclusive_scan` implementation, remember that the contents of the `start` array are copied over to the `output` array. Also, the arrays passed to `exclusive_scan` are assumed to be in `device` memory.

**Grading:** We will test your code for correctness and performance on random input arrays.

For reference, a scan score table is provided below, showing the performance of a simple CUDA implementation on a K80 GPU. To check the correctness and performance score of your `scan` and `find_repeats` implementation, run **`./checker.py scan`** and **`./checker.py find_repeats`** respectively. Doing so will produce a reference table as shown below; your score is based solely on the performance of your code. In order to get full credit, your code must perform within 20% of the provided reference solution.

```
-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 0.766           | 0.143 (F)       | 0               |
| 10000000        | 8.876           | 0.165 (F)       | 0               |
| 20000000        | 17.537          | 0.157 (F)       | 0               |
| 40000000        | 34.754          | 0.139 (F)       | 0               |
-------------------------------------------------------------------------
|                                   | Total score:    | 0/5             |
-------------------------------------------------------------------------
```

This part of the assignment is largely about getting more practice with writing CUDA and thinking in a data parallel manner, and not about performance tuning code. Getting full performance points on this part of the assignment should not require much (or really any) performance tuning, just a direct port of the algorithm pseudocode to CUDA. However, there's one trick: a naive implementation of scan might launch N CUDA threads for each iteration of the parallel loops in the pseudocode, and using conditional execution in the kernel to determine which threads actually need to do work. Such a solution will not be performant! (Consider the last outer-most loop iteration of the upsweep phase, where only two threads would do work!). A full credit solution will only launch one CUDA thread for each iteration of the innermost parallel loops.

**Test Harness:** By default, the test harness runs on a pseudo-randomly generated array that is the same every time
the program is run, in order to aid in debugging. You can pass the argument `-i random` to run on a random array - we
will do this when grading. We encourage you to come up with alternate inputs to your program to help you evaluate it.
You can also use the `-n <size>` option to change the length of the input array.

The argument `--thrust` will use the [Thrust Library's](http://thrust.github.io/) implementation of [exclusive scan](https://docs.nvidia.com/cuda/archive/12.2.2/thrust/index.html?highlight=group%20prefix%20sums#prefix-sums). **Up to two points of extra credit for anyone that can create an implementation is competitive with Thrust.**

直接实现：a direct port of the algorithm pseudocode to CUDA

```C++
__global__ void upsweep(int N, int* result, int two_d, int two_dplus1)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N && index%two_dplus1 == 0)
        result[index + two_dplus1-1] += result[index + two_d-1];
}

__global__ void downsweep(int N, int* result, int two_d, int two_dplus1)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N && index%two_dplus1 == 0) {
        int t = result[index+two_d-1];
        result[index+two_d-1] = result[index+two_dplus1-1];
        result[index+two_dplus1-1] += t;
    }
}

void exclusive_scan(int* input, int N, int* result)
{
    N = nextPow2(N);
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    for(int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        upsweep<<<blocks, threadsPerBlock>>>(N, result, two_d, two_dplus1);
        cudaDeviceSynchronize();
    }
    cudaMemset(result+N-1, 0, sizeof(int));
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        downsweep<<<blocks, threadsPerBlock>>>(N, result, two_d, two_dplus1);
        cudaDeviceSynchronize();
    }
}
```

```bash
# ./cudaScan
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Array size: 64
Student GPU time: 0.586 ms
Scan outputs are correct!
```

```bash
# ./checker.py scan
Test: scan

--------------
Running tests:
--------------

Element Count: 1000000
Correctness passed!
Student Time: 3.853
Ref Time: 1.547

Element Count: 10000000
Correctness passed!
Student Time: 22.006
Ref Time: 11.227

Element Count: 20000000
Correctness passed!
Student Time: 34.736
Ref Time: 21.942

Element Count: 40000000
Correctness passed!
Student Time: 67.59
Ref Time: 42.138

-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 1.547           | 3.853           | 0.5018816506618219 |
| 10000000        | 11.227          | 22.006          | 0.6377238025992911 |
| 20000000        | 21.942          | 34.736          | 0.7895986872409029 |
| 40000000        | 42.138          | 67.59           | 0.7792942743009321 |
-------------------------------------------------------------------------
|                                   | Total score:    | 2.708498414802948/5.0 |
-------------------------------------------------------------------------
```

```C++
__global__ void mark(int N, int* flags, int* input)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    if(index<N-1)
    {
        if(input[index] == input[index+1]) flags[index] = 1;
    }
}

__global__ void collect(int N, int* flags, int* output)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    if(index<N-1)
    {
        if(flags[index]!=flags[index+1])
        {
            output[flags[index]]=index;
        }
    }
}

int find_repeats(int* device_input, int length, int* device_output) {
    int* device_flags;
    int rounded_length = nextPow2(length);
    const int threadsPerBlock = 512;
    const int blocks = (rounded_length + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc(&device_flags, rounded_length*sizeof(int));
    cudaMemset(device_flags, 0, sizeof(int)*rounded_length);
    // 标记重复元素
    mark<<<blocks, threadsPerBlock>>>(length, device_flags, device_input);
    cudaDeviceSynchronize();
    // 标记数组进行exclusive scan, 算出每个重复元素在输出数组中的位置
    exclusive_scan(device_flags, rounded_length, device_flags);
    // 收集重复元素
    collect<<<blocks, threadsPerBlock>>>(length, device_flags, device_output);
    cudaDeviceSynchronize();
    int* res = new int;
    cudaMemcpy(res, device_flags+length-1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_flags);
    return *res; 
}
```

```bash
# ./cudaScan -m find_repeats
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Array size: 64
Student GPU time: 0.821 ms
Find_repeats outputs are correct!
```

```bash
# ./checker.py find_repeats
Test: find_repeats

--------------
Running tests:
--------------

Element Count: 1000000
Correctness passed!
Student Time: 4.846
Ref Time: 2.767

Element Count: 10000000
Correctness passed!
Student Time: 27.559
Ref Time: 17.14

Element Count: 20000000
Correctness passed!
Student Time: 40.701
Ref Time: 32.639

Element Count: 40000000
Correctness passed!
Student Time: 80.595
Ref Time: 63.717

-------------------------
Find_repeats Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 2.767           | 4.846           | 0.7137329756500206 |
| 10000000        | 17.14           | 27.559          | 0.7774229834173954 |
| 20000000        | 32.639          | 40.701          | 1.0024016608928528 |
| 40000000        | 63.717          | 80.595          | 0.9882281779266704 |
-------------------------------------------------------------------------
|                                   | Total score:    | 3.481785797886939/5.0 |
-------------------------------------------------------------------------
```

优化：只为进行运算的 thread 生成，only launch one CUDA thread for each iteration of the innermost parallel loops  

```C++
__global__ void upsweep(int* result, int two_d)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int a = index*two_d*2 + two_d*2-1;
    int b = index*two_d*2 + two_d-1;
    result[a] += result[b];       
}

__global__ void downsweep(int* result, int two_d)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int a = index*two_d*2 + two_d-1;
    int b = index*two_d*2 + two_d*2-1;
    int t = result[a];
    result[a] = result[b];
    result[b] += t;
}

void exclusive_scan(int* input, int N, int* result)
{
    N = nextPow2(N);
    int threadsPerBlock = 512;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int loop = 0;
    int two_dplus1 = 0;
    for(int two_d = 1; two_d <= N/2; two_d*=2) {
        two_dplus1 = 2*two_d;

        loop = (N+two_dplus1-1)/two_dplus1;
        threadsPerBlock = min(512, loop);
        blocks = (loop + threadsPerBlock - 1) / threadsPerBlock;
        upsweep<<<blocks, threadsPerBlock>>>(result, two_d);
    }
    cudaMemset(result+N-1, 0, sizeof(int));
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        two_dplus1 = 2*two_d;
        loop = (N+two_dplus1-1)/two_dplus1;
        threadsPerBlock = min(512, loop);
        blocks = (loop + threadsPerBlock - 1) / threadsPerBlock;
        downsweep<<<blocks, threadsPerBlock>>>(result, two_d);
    }
}
```

`./checker.py scan` 和 `./checker.py find_repeats` 无法使用，版本不一致(我使用的cuda 13，文件支持cuda 12)，Ref Time: None，安装Ubuntu24.04和Cuda12.4后，可以运行  

```bash
# ./checker.py scan
Test: scan

--------------
Running tests:
--------------

Element Count: 1000000
Correctness passed!
Student Time: 0.818
Ref Time: 1.605

Element Count: 10000000
Correctness passed!
Student Time: 7.373
Ref Time: 12.018

Element Count: 20000000
Correctness passed!
Student Time: 14.52
Ref Time: 21.395

Element Count: 40000000
Correctness passed!
Student Time: 29.053
Ref Time: 41.407

-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 1.605           | 0.818           | 1.25            |
| 10000000        | 12.018          | 7.373           | 1.25            |
| 20000000        | 21.395          | 14.52           | 1.25            |
| 40000000        | 41.407          | 29.053          | 1.25            |
-------------------------------------------------------------------------
|                                   | Total score:    | 5.0/5.0         |
-------------------------------------------------------------------------

# ./checker.py find_repeats
Test: find_repeats

--------------
Running tests:
--------------

Element Count: 1000000
Correctness passed!
Student Time: 1.458
Ref Time: 2.653

Element Count: 10000000
Correctness passed!
Student Time: 11.776
Ref Time: 17.54

Element Count: 20000000
Correctness passed!
Student Time: 21.183
Ref Time: 31.59

Element Count: 40000000
Correctness passed!
Student Time: 41.247
Ref Time: 62.493

-------------------------
Find_repeats Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 2.653           | 1.458           | 1.25            |
| 10000000        | 17.54           | 11.776          | 1.25            |
| 20000000        | 31.59           | 21.183          | 1.25            |
| 40000000        | 62.493          | 41.247          | 1.25            |
-------------------------------------------------------------------------
|                                   | Total score:    | 5.0/5.0         |
-------------------------------------------------------------------------
```

```bash
# ./cudaScan
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Array size: 64
Student GPU time: 0.105 ms
Scan outputs are correct!

# ./cudaScan -t
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Array size: 64
Thrust GPU time: 0.139 ms
Scan outputs are correct!
```

```bash
# ./cudaScan -n 1000000
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Array size: 1000000
Student GPU time: 0.828 ms
Scan outputs are correct!

# ./cudaScan -t -n 1000000
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Array size: 1000000
Thrust GPU time: 0.814 ms
Scan outputs are correct!
```

```bash
# ./cudaScan -n 10000000
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Array size: 10000000
Student GPU time: 7.369 ms
Scan outputs are correct!

# ./cudaScan -t -n 10000000
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Array size: 10000000
Thrust GPU time: 1.581 ms
Scan outputs are correct!
```

```bash
# ./cudaScan -n 20000000
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Array size: 20000000
Student GPU time: 14.390 ms
Scan outputs are correct!

# ./cudaScan -t -n 20000000
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Array size: 20000000
Thrust GPU time: 2.120 ms
Scan outputs are correct!
```

```bash
# ./cudaScan -n 40000000
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Array size: 40000000
Student GPU time: 28.973 ms
Scan outputs are correct!

# ./cudaScan -t -n 40000000
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1660 Ti
   SMs:        24
   Global mem: 6144 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Array size: 40000000
Thrust GPU time: 2.575 ms
Scan outputs are correct!
```

当数据规模小于1000000时，实现的`scan`性能与`Thrust library implementation`性能相当，甚至略优，大于1000000时，性能远不及`Thrust library implementation`  

CPU端不需要同步，GPU端隐式同步，CUDA 在同一个流（Stream）中发射的内核，其执行在 GPU 侧是隐式串行（顺序）的  
GPU 侧同步（隐式）：流机制确保了 Kernel 1 `→` Kernel 2 `→` Kernel 3 的顺序  
CPU 侧同步（显式）：当调用 cudaDeviceSynchronize() 或 cudaMemcpy() 时，让 **CPU** 停下来等待 **GPU** 把队列里所有的活都干完  

## Part 3: A Simple Circle Renderer (85 pts)

Now for the real show!

The directory `/render` of the assignment starter code contains an implementation of renderer that draws colored
circles. Build the code, and run the render with the following command line: `./render -r cpuref rgb`. The program will output an image `output_0000.ppm` containing three circles. Now run the renderer with the command line `./render -r cpuref snow`. Now the output image will be falling snow. PPM images can be viewed directly on OSX via preview. For windows you might need to download a viewer.

Note: you can also use the `-i` option to send renderer output to the display instead of a file. (In the case of snow, you'll see an animation of falling snow.) However, to use interactive mode you'll need to be able to setup X-windows forwarding to your local machine. ([This reference](http://atechyblog.blogspot.com/2014/12/google-cloud-compute-x11-forwarding.html) or [this reference](https://stackoverflow.com/questions/25521486/x11-forwarding-from-debian-on-google-compute-engine) may help.)

The assignment starter code contains two versions of the renderer: a sequential, single-threaded C++
reference implementation, implemented in `refRenderer.cpp`, and an _incorrect_ parallel CUDA implementation in
`cudaRenderer.cu`.

### Renderer Overview

We encourage you to familiarize yourself with the structure of the renderer codebase by inspecting the reference
implementation in `refRenderer.cpp`. The method `setup` is called prior to rendering the first frame. In your CUDA-accelerated
renderer, this method will likely contain all your renderer initialization code (allocating buffers, etc). `render`
is called each frame and is responsible for drawing all circles into the output image. The other main function of
the renderer, `advanceAnimation`, is also invoked once per frame. It updates circle positions and velocities.
You will not need to modify `advanceAnimation` in this assignment.

The renderer accepts an array of circles (3D position, velocity(速度), radius, color) as input. The basic sequential
algorithm for rendering each frame is:

    Clear image
    for each circle
        update position and velocity
    for each circle
        compute screen bounding box
        for all pixels in bounding box
            compute pixel center point
            if center point is within the circle
                compute color of circle at point
                blend contribution of circle into image for this pixel

The figure below illustrates the basic algorithm for computing circle-pixel coverage using point-in-circle tests. Notice that a circle contributes color to an output pixel only if the pixel's center lies within the circle.

![Point in circle test](handout/point_in_circle.jpg?raw=true "A simple algorithm for computing the contribution of a circle to the output image: All pixels within the circle's bounding box are tested for coverage. For each pixel in the bounding box, the pixel is considered to be covered by the circle if its center point (black dots) is contained within the circle. Pixel centers that are inside the circle are colored red. The circle's contribution to the image will be computed only for covered pixels.")

An important detail of the renderer is that it renders **semi-transparent** circles. Therefore, the color of any one pixel is not the color of a single circle, but the result of blending the contributions of all the semi-transparent circles overlapping the pixel (note the "blend contribution" part of the pseudocode above). The renderer represents the color of a circle via a 4-tuple of red (R), green (G), blue (B), and opacity (alpha) values (RGBA). Alpha = 1 corresponds to a fully opaque circle. Alpha = 0 corresponds to a fully transparent circle. To draw a semi-transparent circle with color `(C_r, C_g, C_b, C_alpha)` on top of a pixel with color `(P_r, P_g, P_b)`, the renderer uses the following math:

<pre>
   result_r = C_alpha * C_r + (1.0 - C_alpha) * P_r
   result_g = C_alpha * C_g + (1.0 - C_alpha) * P_g
   result_b = C_alpha * C_b + (1.0 - C_alpha) * P_b
</pre>

Notice that composition is not commutative (object X over Y does not look the same as object Y over X), so it's important that the render draw circles in a manner that follows the order they are provided by the application. (You can assume the application provides the circles in depth order.) For example, consider the two images below where a blue circle is drawn OVER a green circle which is drawn OVER a red circle. In the image on the left, the circles are drawn into the output image in the correct order. In the image on the right, the circles are drawn in a different order, and the output image does not look correct.

![Ordering](handout/order.jpg?raw=true "The renderer must be careful to generate output that is the same as what is generated when sequentially drawing all circles in the order provided by the application.")

### CUDA Renderer

After familiarizing yourself with the circle rendering algorithm as implemented in the reference code, now
study the CUDA implementation of the renderer provided in `cudaRenderer.cu`. You can run the CUDA
implementation of the renderer using the `--renderer cuda (or -r cuda)` cuda program option.

The provided CUDA implementation parallelizes computation across all input circles, assigning one circle to
each CUDA thread. While this CUDA implementation is a complete implementation of the mathematics(数学原理) of
a circle renderer, it contains several major errors that you will fix in this assignment. Specifically: the current
implementation does not ensure image update is an atomic operation and it does not preserve the required
order of image updates (the ordering requirement will be described below).

### Renderer Requirements

Your parallel CUDA renderer implementation must maintain two invariants(不变量) that are preserved trivially in
the sequential implementation.

1. **Atomicity:** All image update operations must be atomic. The critical region includes reading the
   four 32-bit floating-point values (the pixel's rgba color), blending the contribution of the current circle with
   the current image value, and then writing the pixel's color back to memory.
2. **Order:** Your renderer must perform updates to an image pixel in _circle input order_. That is, if
   circle 1 and circle 2 both contribute to pixel P, any image updates to P due to circle 1 must be applied to the
   image before updates to P due to circle 2. As discussed above, preserving the ordering requirement
   allows for correct rendering of transparent circles. (It has a number of other benefits for graphics
   systems. If curious, talk to Kayvon.) **A key observation is that the definition of order only specifies the order of updates to the same pixel.** Thus, as shown below, there are no ordering requirements between circles that do not contribute to the same pixel. These circles can be processed independently.

![Dependencies](handout/dependencies.jpg?raw=true "The contributions of circles 1, 2, and 3 must be applied to overlapped pixels in the order the circles are provided to the renderer.")

Since the provided CUDA implementation does not satisfy either of these requirements, the result of not correctly
respecting order or atomicity can be seen by running the CUDA renderer implementation on the rgb and circles scenes.
You will see horizontal streaks(条纹) through the resulting images, as shown below. These streaks will change with each frame.

![Order_errors](handout/bug_example.jpg?raw=true "Errors in the output due to lack of atomicity in frame-buffer update (notice streaks in bottom of image).")

### What You Need To Do

**Your job is to write the fastest, correct CUDA renderer implementation you can**. You may take any approach you
see fit, but your renderer must adhere(遵守) to the atomicity and order requirements specified above. A solution that does not meet both requirements will be given no more than 12 points on part 3 of the assignment. We have already given you such a solution!

A good place to start would be to read through `cudaRenderer.cu` and convince yourself that it _does not_ meet the correctness requirement. In particular, look at how `CudaRenderer:render` launches the CUDA kernel `kernelRenderCircles`. (`kernelRenderCircles` is where all the work happens.) To visually see the effect of violation of above two requirements, compile the program with `make`. Then run `./render -r cuda rand10k` which should display the image with 10K circles, shown in the bottom row of the image above. Compare this (incorrect) image with the image generated by sequential code by running `./render -r cpuref rand10k`.

We recommend that you:

1. First rewrite the CUDA starter code implementation so that it is logically correct when running in parallel (we recommend an approach that does not require locks or synchronization)
2. Then determine what performance problem is with your solution.
3. At this point the real thinking on the assignment begins... (Hint: the circle-intersects-box tests provided to you in `circleBoxTest.cu_inl` are your friend. You are encouraged to use these subroutines.)

Following are commandline options to `./render`:

```
Usage: ./render [options] scenename
Valid scenenames are: rgb, rgby, rand10k, rand100k, rand1M, biglittle, littlebig, pattern, micro2M,
                      bouncingballs, fireworks, hypnosis, snow, snowsingle
Program Options:
  -r  --renderer <cpuref/cuda>  Select renderer: ref or cuda (default=cuda)
  -s  --size  <INT>             Rendered image size: <INT>x<INT> pixels (default=1024)
  -b  --bench <START:END>       Run for frames [START,END) (default=[0,1))
  -c  --check                   Check correctness of CUDA output against CPU reference
  -i  --interactive             Render output to interactive display
  -f  --file  <FILENAME>        Output file name (FILENAME_xxxx.ppm) (default=output)
  -?  --help                    This message
```

**Checker code:** To detect correctness of the program, `render` has a convenient `--check` option. This option runs the sequential version of the reference CPU renderer along with your CUDA renderer and then compares the resulting images to ensure correctness. The time taken by your CUDA renderer implementation is also printed.

We provide a total of eight circle datasets you will be graded on. However, in order to receive full credit, your code must pass all of our correctness-tests. To check the correctness and performance score of your code, run **`./checker.py`** (notice the .py extension) in the `/render` directory. If you run it on the starter code, the program will print a table like the following, along with the results of our entire test set:

```
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2622           | (F)             | 0               |
| rand10k         | 3.0658           | (F)             | 0               |
| rand100k        | 29.6144          | (F)             | 0               |
| pattern         | 0.4043           | (F)             | 0               |
| snowsingle      | 19.7155          | (F)             | 0               |
| biglittle       | 15.2422          | (F)             | 0               |
| rand1M          | 230.478          | (F)             | 0               |
| micro2M         | 439.9369         | (F)             | 0               |
--------------------------------------------------------------------------
|                                    | Total score:    | 0/72            |
--------------------------------------------------------------------------
```

Note: on some runs, you _may_ receive credit for some of these scenes, since the provided renderer's runtime is non-deterministic sometimes it might be correct. This doesn't change the fact that the current CUDA renderer is in general incorrect.

"Ref time" is the performance of our reference solution on your current machine (in the provided `render_ref` executable). "Your time" is the performance of your current CUDA renderer solution, where an `(F)` indicates an incorrect solution. Your grade will depend on the performance of your implementation compared to these reference implementations (see Grading Guidelines).

Along with your code, we would like you to hand in a clear, high-level description of how your implementation works as well as a brief description of how you arrived at this solution. Specifically address(解决) approaches you tried along the way, and how you went about determining how to optimize your code (For example, what measurements did you perform to guide your optimization efforts?).

Aspects of your work that you should mention in the write-up include:

1. Include both partners names and SUNet id's at the top of your write-up.
2. Replicate the score table generated for your solution and specify which machine you ran your code on.
3. Describe how you decomposed the problem and how you assigned work to CUDA thread blocks and threads (and maybe even warps).
4. Describe where synchronization occurs in your solution.
5. What, if any, steps did you take to reduce communication requirements (e.g., synchronization or main memory bandwidth requirements)?
6. Briefly describe how you arrived at your final solution. What other approaches did you try along the way. What was wrong with them?

```C++
// 并行计算渲染的每个像素点
__global__ void kernelRenderPixels(){
    int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    int indexy = blockIdx.y*blockDim.y + threadIdx.y;

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;

    if(indexx >= imageWidth || indexy >= imageHeight) return;

    int numCircles = cuConstRendererParams.numCircles;
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (indexy * imageWidth + indexx)]);
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(indexx) + 0.5f),
                                                 invHeight * (static_cast<float>(indexy) + 0.5f));
    for(int i=0; i<numCircles; i++)
    {
        int index3 = 3 * i;
        // read position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float  rad = cuConstRendererParams.radius[i];
        // compute the bounding box of the circle. The bound is in integer
        // screen coordinates, so it's clamped to the edges of the screen.
        short minX = static_cast<short>(imageWidth * (p.x - rad));
        short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
        short minY = static_cast<short>(imageHeight * (p.y - rad));
        short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;
        if(indexx<minX || indexx>maxX) continue;
        if(indexy<minY || indexy>maxY) continue;
        shadePixel(i, pixelCenterNorm, p, imgPtr);
    }

}

void CudaRenderer::render() {

    dim3 blockDim(16, 16);
    dim3 gridDim((image->width + blockDim.x - 1) / blockDim.x, (image->height + blockDim.y - 1) / blockDim.y);
    kernelRenderPixels<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();   
}
```

```bash
# ./checker.py

Running scene: rgb...
[rgb] Correctness passed!
[rgb] Student times:  [0.5855, 0.7084, 0.5416]
[rgb] Reference times:  [0.7526, 0.6638, 0.6283]

Running scene: rand10k...
[rand10k] Correctness passed!
[rand10k] Student times:  [76.1714, 71.3985, 71.0334]
[rand10k] Reference times:  [4.4718, 4.3868, 5.4531]

Running scene: rand100k...
[rand100k] Correctness passed!
[rand100k] Student times:  [755.1889, 729.9354, 747.3187]
[rand100k] Reference times:  [44.3781, 43.0792, 47.4508]

Running scene: pattern...
[pattern] Correctness passed!
[pattern] Student times:  [8.4621, 10.8631, 11.4988]
[pattern] Reference times:  [0.9744, 1.0774, 1.0029]

Running scene: snowsingle...
[snowsingle] Correctness passed!
[snowsingle] Student times:  [713.1776, 668.3803, 622.2528]
[snowsingle] Reference times:  [26.4982, 31.5826, 33.9116]

Running scene: biglittle...
[biglittle] Correctness passed!
[biglittle] Student times:  [98.936, 96.4138, 97.5675]
[biglittle] Reference times:  [21.0294, 23.2805, 26.9064]

Running scene: rand1M...
[rand1M] Correctness passed!
[rand1M] Student times:  [7310.7074, 7402.4089, 7266.8676]
[rand1M] Reference times:  [344.8312, 289.1865, 281.7732]

Running scene: micro2M...
[micro2M] Correctness passed!
[micro2M] Student times:  [14609.0626, 14460.708, 14460.8661]
[micro2M] Reference times:  [561.4761, 576.7197, 577.104]
------------
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.6283           | 0.5416          | 9               |
| rand10k         | 4.3868           | 71.0334         | 2               |
| rand100k        | 43.0792          | 729.9354        | 2               |
| pattern         | 0.9744           | 8.4621          | 3               |
| snowsingle      | 26.4982          | 622.2528        | 2               |
| biglittle       | 21.0294          | 96.4138         | 4               |
| rand1M          | 281.7732         | 7266.8676       | 2               |
| micro2M         | 561.4761         | 14460.708       | 2               |
--------------------------------------------------------------------------
|                                    | Total score:    | 26/72           |
--------------------------------------------------------------------------
```

分批次处理图，将位置和半径存到共享内存，性能优化较小  

```C++
// 并行计算渲染的每个像素点
__global__ void kernelRenderPixels(){
    int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    int indexy = blockIdx.y*blockDim.y + threadIdx.y;

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;

    if(indexx >= imageWidth || indexy >= imageHeight) return;

    int numCircles = cuConstRendererParams.numCircles;
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (indexy * imageWidth + indexx)]);
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(indexx) + 0.5f), invHeight * (static_cast<float>(indexy) + 0.5f));
    int total = blockDim.x*blockDim.y;

    extern __shared__ float array[];
    float3* p = (float3*)array;
    float* rad = (float*)&p[total];

    // 分批次处理图，将位置和半径存到共享内存
    for(int b=0; b<numCircles; b+=total)
    {
        int index = threadIdx.y*blockDim.x + threadIdx.x;
        if(b+index < numCircles)
        {
            int index3 = 3 * (b+index);
            // read position and radius
            p[index] = *(float3*)(&cuConstRendererParams.position[index3]);
            rad[index] = cuConstRendererParams.radius[b+index];
        }
        __syncthreads();
        for(int i=0; i<total&&b+i<numCircles; i++)
        {   
            // compute the bounding box of the circle. The bound is in integer
            // screen coordinates, so it's clamped to the edges of the screen.
            short minX = static_cast<short>(imageWidth * (p[i].x - rad[i]));
            short maxX = static_cast<short>(imageWidth * (p[i].x + rad[i])) + 1;
            short minY = static_cast<short>(imageHeight * (p[i].y - rad[i]));
            short maxY = static_cast<short>(imageHeight * (p[i].y + rad[i])) + 1;
            if(indexx<minX || indexx>maxX) continue;
            if(indexy<minY || indexy>maxY) continue;
            shadePixel(b+i, pixelCenterNorm, p[i], imgPtr);
        }
        __syncthreads();
    }
}

void CudaRenderer::render() {
    dim3 blockDim(16, 16);
    dim3 gridDim((image->width + blockDim.x - 1) / blockDim.x, (image->height + blockDim.y - 1) / blockDim.y);
    kernelRenderPixels<<<gridDim, blockDim, blockDim.x*blockDim.y*(sizeof(float3)+sizeof(float))>>>();
    cudaDeviceSynchronize();
}
```

```bash
# ./checker.py

Running scene: rgb...
[rgb] Correctness passed!
[rgb] Student times:  [0.6378, 0.7829, 0.5894]
[rgb] Reference times:  [0.6185, 0.58, 0.6557]

Running scene: rand10k...
[rand10k] Correctness passed!
[rand10k] Student times:  [71.9452, 67.5599, 66.9562]
[rand10k] Reference times:  [4.932, 4.54, 5.2868]

Running scene: rand100k...
[rand100k] Correctness passed!
[rand100k] Student times:  [750.41, 688.3847, 697.3045]
[rand100k] Reference times:  [43.193, 43.0471, 43.6084]

Running scene: pattern...
[pattern] Correctness passed!
[pattern] Student times:  [7.9105, 8.5752, 9.7302]
[pattern] Reference times:  [1.1413, 1.1702, 1.013]

Running scene: snowsingle...
[snowsingle] Correctness passed!
[snowsingle] Student times:  [595.5534, 588.1766, 592.019]
[snowsingle] Reference times:  [29.1881, 35.7866, 36.1933]

Running scene: biglittle...
[biglittle] Correctness passed!
[biglittle] Student times:  [94.4037, 92.7923, 91.0411]
[biglittle] Reference times:  [21.6778, 21.8703, 27.0993]

Running scene: rand1M...
[rand1M] Correctness passed!
[rand1M] Student times:  [6706.8021, 6759.4377, 6748.0969]
[rand1M] Reference times:  [308.4922, 306.4452, 307.1411]

Running scene: micro2M...
[micro2M] Correctness passed!
[micro2M] Student times:  [13277.5114, 13291.9472, 13296.4664]
[micro2M] Reference times:  [593.9437, 596.8935, 596.5265]
------------
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.58             | 0.5894          | 9               |
| rand10k         | 4.54             | 66.9562         | 2               |
| rand100k        | 43.0471          | 688.3847        | 2               |
| pattern         | 1.013            | 7.9105          | 3               |
| snowsingle      | 29.1881          | 588.1766        | 2               |
| biglittle       | 21.6778          | 91.0411         | 4               |
| rand1M          | 306.4452         | 6706.8021       | 2               |
| micro2M         | 593.9437         | 13277.5114      | 2               |
--------------------------------------------------------------------------
|                                    | Total score:    | 26/72           |
--------------------------------------------------------------------------
```

使用寄存器执行累加，性能优化较小  

```C++
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// 并行计算渲染的每个像素点
__global__ void kernelRenderPixels(){
    int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    int indexy = blockIdx.y*blockDim.y + threadIdx.y;

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;

    if(indexx >= imageWidth || indexy >= imageHeight) return;

    int numCircles = cuConstRendererParams.numCircles;
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (indexy * imageWidth + indexx)]);
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(indexx) + 0.5f), invHeight * (static_cast<float>(indexy) + 0.5f));
    int total = blockDim.x*blockDim.y;

    extern __shared__ float array[];
    float3* p = (float3*)array;
    float* rad = (float*)&p[total];
    float4 imgReg = *imgPtr;

    // 分批次处理图，将位置和半径存到共享内存
    for(int b=0; b<numCircles; b+=total)
    {
        int index = threadIdx.y*blockDim.x + threadIdx.x;
        if(b+index < numCircles)
        {
            int index3 = 3 * (b+index);
            // read position and radius
            p[index] = *(float3*)(&cuConstRendererParams.position[index3]);
            rad[index] = cuConstRendererParams.radius[b+index];
        }
        __syncthreads();
        for(int i=0; i<total&&b+i<numCircles; i++)
        {   
            // compute the bounding box of the circle. The bound is in integer
            // screen coordinates, so it's clamped to the edges of the screen.
            short minX = static_cast<short>(imageWidth * (p[i].x - rad[i]));
            short maxX = static_cast<short>(imageWidth * (p[i].x + rad[i])) + 1;
            short minY = static_cast<short>(imageHeight * (p[i].y - rad[i]));
            short maxY = static_cast<short>(imageHeight * (p[i].y + rad[i])) + 1;
            if(indexx<minX || indexx>maxX) continue;
            if(indexy<minY || indexy>maxY) continue;
            shadePixel(b+i, pixelCenterNorm, p[i], &imgReg);
        }
        __syncthreads();
    }
    (*imgPtr) = imgReg;
}
```

```bash
# ./checker.py

Running scene: rgb...
[rgb] Correctness passed!
[rgb] Student times:  [0.5638, 0.5748, 0.6341]
[rgb] Reference times:  [0.6002, 0.8954, 0.6575]

Running scene: rand10k...
[rand10k] Correctness passed!
[rand10k] Student times:  [74.8062, 65.8119, 66.1143]
[rand10k] Reference times:  [4.6321, 6.1137, 4.6402]

Running scene: rand100k...
[rand100k] Correctness passed!
[rand100k] Student times:  [688.593, 688.5919, 693.3179]
[rand100k] Reference times:  [45.0659, 45.1973, 45.1702]

Running scene: pattern...
[pattern] Correctness passed!
[pattern] Student times:  [7.7787, 7.6238, 9.9626]
[pattern] Reference times:  [1.2145, 1.003, 1.4029]

Running scene: snowsingle...
[snowsingle] Correctness passed!
[snowsingle] Student times:  [695.7218, 698.1584, 651.4813]
[snowsingle] Reference times:  [29.7673, 30.0593, 37.5142]

Running scene: biglittle...
[biglittle] Correctness passed!
[biglittle] Student times:  [103.4406, 86.9835, 87.3047]
[biglittle] Reference times:  [22.6612, 28.0711, 25.338]

Running scene: rand1M...
[rand1M] Correctness passed!
[rand1M] Student times:  [6722.1093, 6833.6195, 6849.3356]
[rand1M] Reference times:  [319.0858, 319.4691, 342.8659]

Running scene: micro2M...
[micro2M] Correctness passed!
[micro2M] Student times:  [13337.0154, 13360.3456, 13371.7431]
[micro2M] Reference times:  [624.15, 626.4777, 621.196]
------------
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.6002           | 0.5638          | 9               |
| rand10k         | 4.6321           | 65.8119         | 2               |
| rand100k        | 45.0659          | 688.5919        | 2               |
| pattern         | 1.003            | 7.6238          | 3               |
| snowsingle      | 29.7673          | 651.4813        | 2               |
| biglittle       | 22.6612          | 86.9835         | 4               |
| rand1M          | 319.0858         | 6722.1093       | 2               |
| micro2M         | 621.196          | 13337.0154      | 2               |
--------------------------------------------------------------------------
|                                    | Total score:    | 26/72           |
--------------------------------------------------------------------------
```

很多圆无法覆盖当前像素块区域，在将圆点和半径存到共享内存时，找出那些与当前像素块相交的圆（先记录相交的位置，然后用前缀和和反转索引和数组值得到实际相交圆的数组），性能大幅提升  

```C++
// 并行计算渲染的每个像素点
__global__ void kernelRenderPixels(){
    int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    int indexy = blockIdx.y*blockDim.y + threadIdx.y;

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;

    if(indexx >= imageWidth || indexy >= imageHeight) return;

    int numCircles = cuConstRendererParams.numCircles;
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (indexy * imageWidth + indexx)]);
    // local memory
    float4 imgReg = *imgPtr;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(indexx) + 0.5f), invHeight * (static_cast<float>(indexy) + 0.5f));
    // 归一化边界
    float boxL = (blockIdx.x * blockDim.x) * invWidth;
    float boxR = (min((blockIdx.x + 1) * blockDim.x, (int)imageWidth)) * invWidth;
    float boxB = (blockIdx.y * blockDim.y) * invHeight;
    float boxT = (min((blockIdx.y + 1) * blockDim.y, (int)imageHeight)) * invHeight;

    int blockSize = blockDim.x*blockDim.y;
    int index = threadIdx.y*blockDim.x + threadIdx.x;

    // local memory
    extern __shared__ float array[];
    // 坐标
    float3* p = (float3*)array;
    // 半径
    float* rad = (float*)&p[blockSize];
    // 具有当前block上的像素点的圆
    int* activeCircle = (int*)&rad[blockSize];
    // 前缀和数组
    uint* prefixSumInput = (uint*)&activeCircle[blockSize];
    uint* prefixSumOutput = (uint*)&prefixSumInput[blockSize];
    uint* prefixSumScratch = (uint*)&prefixSumOutput[blockSize];
    // activeCircle 实际长度
    int* activeNum = (int*)&prefixSumScratch[2*blockSize];

    // 分批次处理图，将位置和半径存到共享内存
    for(int b=0; b<numCircles; b+=blockSize)
    {
        int circleIdx = b + index;
        if(circleIdx < numCircles)
        {
            int index3 = 3 * (circleIdx);
            // read position and radius
            p[index] = *(float3*)(&cuConstRendererParams.position[index3]);
            rad[index] = cuConstRendererParams.radius[circleIdx];
            if(circleInBoxConservative(p[index].x, p[index].y, rad[index], boxL, boxR, boxT, boxB)
                &&circleInBox(p[index].x, p[index].y, rad[index], boxL, boxR, boxT, boxB))
            {
                prefixSumInput[index] = 1;
            }else{
                prefixSumInput[index] = 0;
            }
        }
        __syncthreads();
        sharedMemExclusiveScan(index, prefixSumInput, prefixSumOutput, prefixSumScratch, blockSize);
        if(prefixSumInput[index]==1) activeCircle[prefixSumOutput[index]] = index;
        if(index == blockSize-1)
        {
            *activeNum = prefixSumOutput[index];
            if(prefixSumInput[index]==1)
            {
                (*activeNum)+=1;
            }
        }
        __syncthreads();
        for(int i=0; i<*activeNum; i++)
        {   
            // compute the bounding box of the circle. The bound is in integer
            // screen coordinates, so it's clamped to the edges of the screen.
            int activeIndex = activeCircle[i];
            short minX = static_cast<short>(imageWidth * (p[activeIndex].x - rad[activeIndex]));
            short maxX = static_cast<short>(imageWidth * (p[activeIndex].x + rad[activeIndex])) + 1;
            short minY = static_cast<short>(imageHeight * (p[activeIndex].y - rad[activeIndex]));
            short maxY = static_cast<short>(imageHeight * (p[activeIndex].y + rad[activeIndex])) + 1;
            if(indexx<minX || indexx>maxX) continue;
            if(indexy<minY || indexy>maxY) continue;
            shadePixel(b+activeIndex, pixelCenterNorm, p[activeIndex], &imgReg);
        }
        __syncthreads();
    }
    (*imgPtr) = imgReg;
}
```

```bash
# ./checker.py

Running scene: rgb...
[rgb] Correctness passed!
[rgb] Student times:  [0.7222, 0.9091, 0.9204]
[rgb] Reference times:  [0.6455, 0.8946, 0.6545]

Running scene: rand10k...
[rand10k] Correctness passed!
[rand10k] Student times:  [4.8459, 5.0858, 4.6444]
[rand10k] Reference times:  [5.9703, 5.8239, 6.0941]

Running scene: rand100k...
[rand100k] Correctness passed!
[rand100k] Student times:  [44.6331, 35.9359, 36.1893]
[rand100k] Reference times:  [49.5452, 43.7377, 44.7678]

Running scene: pattern...
[pattern] Correctness passed!
[pattern] Student times:  [0.8168, 0.871, 0.9131]
[pattern] Reference times:  [1.4148, 1.2123, 1.2404]

Running scene: snowsingle...
[snowsingle] Correctness passed!
[snowsingle] Student times:  [14.0426, 13.7608, 14.0796]
[snowsingle] Reference times:  [36.7026, 36.6314, 36.6902]

Running scene: biglittle...
[biglittle] Correctness passed!
[biglittle] Student times:  [37.1478, 34.7683, 34.4154]
[biglittle] Reference times:  [28.8619, 28.2687, 28.026]

Running scene: rand1M...
[rand1M] Correctness passed!
[rand1M] Student times:  [137.3887, 136.0685, 135.298]
[rand1M] Reference times:  [308.8042, 311.6557, 309.5644]

Running scene: micro2M...
[micro2M] Correctness passed!
[micro2M] Student times:  [217.9831, 219.9023, 214.7851]
[micro2M] Reference times:  [592.31, 596.9622, 602.7113]
------------
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.6455           | 0.7222          | 9               |
| rand10k         | 5.8239           | 4.6444          | 9               |
| rand100k        | 43.7377          | 35.9359         | 9               |
| pattern         | 1.2123           | 0.8168          | 9               |
| snowsingle      | 36.6314          | 13.7608         | 9               |
| biglittle       | 28.026           | 34.4154         | 8               |
| rand1M          | 308.8042         | 135.298         | 9               |
| micro2M         | 592.31           | 214.7851        | 9               |
--------------------------------------------------------------------------
|                                    | Total score:    | 71/72           |
--------------------------------------------------------------------------
```

将部分访问共享内存优化为访问寄存器，性能优化较小，通过所有测试  

```C++
// 并行计算渲染的每个像素点
__global__ void kernelRenderPixels(){
    int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    int indexy = blockIdx.y*blockDim.y + threadIdx.y;

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;

    if(indexx >= imageWidth || indexy >= imageHeight) return;

    int numCircles = cuConstRendererParams.numCircles;
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (indexy * imageWidth + indexx)]);
    // local memory
    float4 imgReg = *imgPtr;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(indexx) + 0.5f), invHeight * (static_cast<float>(indexy) + 0.5f));
    // 归一化边界
    float boxL = (blockIdx.x * blockDim.x) * invWidth;
    float boxR = (min((blockIdx.x + 1) * blockDim.x, (int)imageWidth)) * invWidth;
    float boxB = (blockIdx.y * blockDim.y) * invHeight;
    float boxT = (min((blockIdx.y + 1) * blockDim.y, (int)imageHeight)) * invHeight;

    int blockSize = blockDim.x*blockDim.y;
    int index = threadIdx.y*blockDim.x + threadIdx.x;

    // local memory
    extern __shared__ float array[];
    // 坐标和半径
    float4* p = (float4*)array;
    // 具有当前block上的像素点的圆
    int* activeCircle = (int*)&p[blockSize];
    // 前缀和数组
    uint* prefixSumInput = (uint*)&activeCircle[blockSize];
    uint* prefixSumOutput = (uint*)&prefixSumInput[blockSize];
    uint* prefixSumScratch = (uint*)&prefixSumOutput[blockSize];
    // activeCircle 实际长度
    int* activeNum = (int*)&prefixSumScratch[2*blockSize];

    // 分批次处理图，将位置和半径存到共享内存
    for(int b=0; b<numCircles; b+=blockSize)
    {
        int circleIdx = b + index;
        bool is_active = false;
        if(circleIdx < numCircles)
        {
            int index3 = 3 * (circleIdx);
            // read position and radius
            float3 pos = *(float3*)(&cuConstRendererParams.position[index3]);
            float r = cuConstRendererParams.radius[circleIdx];
            p[index] = make_float4(pos.x, pos.y, pos.z, r);
            if(circleInBoxConservative(pos.x, pos.y, r, boxL, boxR, boxT, boxB)
                &&circleInBox(pos.x, pos.y, r, boxL, boxR, boxT, boxB))
            {
                is_active = true;
            }
        }
        prefixSumInput[index] = is_active ? 1 : 0;
        __syncthreads();
        sharedMemExclusiveScan(index, prefixSumInput, prefixSumOutput, prefixSumScratch, blockSize);
        if(is_active) activeCircle[prefixSumOutput[index]] = index;
        if(index == blockSize-1)
        {
            *activeNum = prefixSumOutput[index];
            if(is_active)
            {
                (*activeNum)+=1;
            }
        }
        __syncthreads();
        int activeNumReg = *activeNum;
        for(int i=0; i<activeNumReg; i++)
        {   
            // compute the bounding box of the circle. The bound is in integer
            // screen coordinates, so it's clamped to the edges of the screen.
            int activeIndex = activeCircle[i];
            short minX = static_cast<short>(imageWidth * (p[activeIndex].x - p[activeIndex].w));
            short maxX = static_cast<short>(imageWidth * (p[activeIndex].x + p[activeIndex].w)) + 1;
            short minY = static_cast<short>(imageHeight * (p[activeIndex].y - p[activeIndex].w));
            short maxY = static_cast<short>(imageHeight * (p[activeIndex].y + p[activeIndex].w)) + 1;
            if(indexx<minX || indexx>maxX) continue;
            if(indexy<minY || indexy>maxY) continue;
            shadePixel(b+activeIndex, pixelCenterNorm, p[activeIndex], &imgReg);
        }
        __syncthreads();
    }
    (*imgPtr) = imgReg;
}
```

```bash
# ./checker.py

Running scene: rgb...
[rgb] Correctness passed!
[rgb] Student times:  [0.6785, 0.7534, 0.7632]
[rgb] Reference times:  [0.6403, 0.6363, 0.6449]

Running scene: rand10k...
[rand10k] Correctness passed!
[rand10k] Student times:  [4.462, 4.6237, 4.4301]
[rand10k] Reference times:  [5.9697, 5.7566, 5.8482]

Running scene: rand100k...
[rand100k] Correctness passed!
[rand100k] Student times:  [41.0869, 37.5473, 37.868]
[rand100k] Reference times:  [52.6447, 43.8476, 44.003]

Running scene: pattern...
[pattern] Correctness passed!
[pattern] Student times:  [1.0924, 0.8026, 0.8519]
[pattern] Reference times:  [0.9731, 0.9835, 0.9784]

Running scene: snowsingle...
[snowsingle] Correctness passed!
[snowsingle] Student times:  [14.4827, 13.2375, 13.5746]
[snowsingle] Reference times:  [36.702, 37.0506, 36.1585]

Running scene: biglittle...
[biglittle] Correctness passed!
[biglittle] Student times:  [36.0557, 32.6609, 32.6775]
[biglittle] Reference times:  [27.8066, 28.0287, 28.2634]

Running scene: rand1M...
[rand1M] Correctness passed!
[rand1M] Student times:  [129.3383, 127.5251, 128.0196]
[rand1M] Reference times:  [305.9576, 308.2851, 308.4245]

Running scene: micro2M...
[micro2M] Correctness passed!
[micro2M] Student times:  [208.5126, 207.6844, 209.1373]
[micro2M] Reference times:  [600.8641, 603.7618, 606.6669]
------------
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.6363           | 0.6785          | 9               |
| rand10k         | 5.7566           | 4.4301          | 9               |
| rand100k        | 43.8476          | 37.5473         | 9               |
| pattern         | 0.9731           | 0.8026          | 9               |
| snowsingle      | 36.1585          | 13.2375         | 9               |
| biglittle       | 27.8066          | 32.6609         | 9               |
| rand1M          | 305.9576         | 127.5251        | 9               |
| micro2M         | 600.8641         | 207.6844        | 9               |
--------------------------------------------------------------------------
|                                    | Total score:    | 72/72           |
--------------------------------------------------------------------------
```

### Grading Guidelines

- The write-up for the assignment is worth 18 points.
- Your parallel prefix implementation is worth 10 points.
- Your render implementation is worth 72 points. These are equally divided into 9 points per scene as follows:
  - 2 correctness points per scene. We will only test your program with image sizes that are multiples of 256.
  - 7 performance points per scene (only obtainable if the solution is correct). Your performance will be graded with respect to the performance of a provided benchmark reference renderer, T<sub>ref</sub>:
    - No performance points will be given for solutions having time (T) 10 times the magnitude(大小) of T<sub>ref</sub>.
    - Full performance points will be given for solutions within 20% of the optimized solution ( T <= 1.20 \* T<sub>ref</sub> )
    - For other values of T (for 1.20 T<sub>ref</sub> < T < 10 _ T<sub>ref</sub>), your performance score on a scale 1 to 7 will be calculated as: `7 _ T_ref / T`.

- Up to five points extra credit (instructor discretion(审慎)) for solutions that achieve significantly greater performance than required. Your write up must clearly explain your approach thoroughly.
- Up to five points extra credit (instructor discretion) for a high-quality parallel CPU-only renderer implementation that achieves good utilization of all cores and SIMD vector units of the cores. Feel free to use any tools at your disposal (e.g., SIMD intrinsics, ISPC, pthreads). To receive credit you should analyze the performance of your GPU and CPU-based solutions and discuss the reasons for differences in implementation choices made.

So the total points for this project is as follows:

- part 1 (5 points)
- part 2 (10 points)
- part 3 write up (13 points)
- part 3 implementation (72 points)
- potential **extra** credit (up to 10 points)

## Assignment Tips and Hints

Below are a set of tips and hints compiled from previous years. Note that there are various ways to implement your renderer and not all hints may apply to your approach.

- There are two potential axes of parallelism in this assignment. One axis is _parallelism across pixels_ another is _parallelism across circles_ (provided the ordering requirement is respected for overlapping circles). Solutions will need to exploit both types of parallelism, potentially at different parts of the computation.
- The circle-intersects-box tests provided to you in `circleBoxTest.cu_inl` are your friend. You are encouraged to use these subroutines.
- The shared-memory prefix-sum operation provided in `exclusiveScan.cu_inl` may be valuable to you on this assignment (not all solutions may choose to use it). See the simple description of a prefix-sum [here](https://docs.nvidia.com/cuda/archive/12.2.1/thrust/index.html#prefix-sums). We
  have provided an implementation of an exclusive prefix-sum on a **power-of-two-sized** arrays in shared memory. **The provided code does not work on non-power-of-two inputs and IT ALSO REQUIRES THAT THE NUMBER OF THREADS IN THE THREAD BLOCK BE THE SIZE OF THE ARRAY. PLEASE READ THE COMMENTS IN THE CODE.**
- Take a look at the `shadePixel` method that is being called. Notice how it is doing many global memory operations to update the color of a pixel. It might be wise to instead use a local accumulator in your `kernelRenderCircles` method. You can then perform the accumulation of a pixel value in a register, and once the final pixel value is accumulated you can then just perform a single write to global memory.
- You are allowed to use the [Thrust library](http://thrust.github.io/) in your implementation if you so choose. Thrust is not necessary to achieve the performance of the optimized CUDA reference implementations. There is one popular way of solving the problem that uses the shared memory prefix-sum implementation that we give you. There another popular way that uses the prefix-sum routines in the Thrust library. Both are valid solution strategies.
- Is there data reuse in the renderer? What can be done to exploit this reuse?
- How will you ensure atomicity of image update since there is no CUDA language primitive that performs the logic of the image update operation atomically? Constructing a lock out of global memory atomic operations is one solution, but keep in mind that even if your image update is atomic, the updates must be performed in the required order. **We suggest that you think about ensuring order in your parallel solution first, and only then consider the atomicity problem (if it still exists at all) in your solution.**
- For the tests which contain a larger number of circles - `rand1M` and `micro2M` - you should be careful about allocating temporary structures in global memory. If you allocate too much global memory, you will have used up all the memory on the device. If you are not checking the `cudaError_t` value that is returned from a call to `cudaMalloc`, then the program will still execute but you will not know that you ran out of device memory. Instead, you will fail the correctness check because you were not able to make your temporary structures. This is why we suggest you to use the CUDA API call wrapper below so you can wrap your `cudaMalloc` calls and produce an error when you run out of device memory.
- If you find yourself with free time, have fun making your own scenes!

### Catching CUDA Errors

By default, if you access an array out of bounds, allocate too much memory, or otherwise cause an error, CUDA won't normally inform you; instead it will just fail silently and return an error code. You can use the following macro (feel free to modify it) to wrap CUDA calls:

```
#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif
```

Note that you can undefine DEBUG to disable error checking once your code is correct for improved performance.

You can then wrap CUDA API calls to process their returned errors as such:

```
cudaCheckError( cudaMalloc(&a, size*sizeof(int)) );
```

Note that you can't wrap kernel launches directly. Instead, their errors will be caught on the next CUDA call you wrap:

```
kernel<<<1,1>>>(a); // suppose kernel causes an error!
cudaCheckError( cudaDeviceSynchronize() ); // error is printed on this line
```

All CUDA API functions, `cudaDeviceSynchronize`, `cudaMemcpy`, `cudaMemset`, and so on can be wrapped.

**IMPORTANT:** if a CUDA function error'd previously, but wasn't caught, that error will show up in the next error check, even if that wraps a different function. For example:

```
...
line 742: cudaMalloc(&a, -1); // executes, then continues
line 743: cudaCheckError(cudaMemcpy(a,b)); // prints "CUDA Error: out of memory at cudaRenderer.cu:743"
...
```

Therefore, while debugging, it's recommended that you wrap **all** CUDA API calls (at least in code that you wrote).

(Credit: adapted from [this Stack Overflow post](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api))

## 3.4 Hand-in Instructions

Please submit your work using Gradescope. If you are working with a partner please remember to tag your partner on gradescope.

1. **Please submit your writeup as the file `writeup.pdf`.**
2. **Please submit run `sh create_submission.sh` to generate a zip to submit to gradescope.** Note that this will run make clean in your code directories so you will have to run make again to run your code. If the script errors saying 'Permission denied', you should run `chmod +x create\_submission.sh` and then try rerunning the script.

   Our grading scripts will rerun the checker code allowing us to verify your score matches what you submitted in the `writeup.pdf`. We might also try to run your code on other datasets to further examine its correctness. 