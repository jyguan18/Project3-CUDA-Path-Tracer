#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void naiveScan(int n, int *odata, const int *idata, int d) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx >= n) return;
            if (idx >= (1 << (d - 1))) {
                odata[idx] = idata[idx - (1 << (d - 1))] + idata[idx];
            }
            else {
                odata[idx] = idata[idx];
            }
            
        }

        __global__ void exclusiveScan(int n, int* odata, const int* idata) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);

            if (idx >= n) return;

            if (idx > 0) {
                odata[idx] = idata[idx - 1];
            }
            else {
                odata[idx] = 0;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* tempIn;
            int* tempOut;
            cudaMalloc((void**)&tempIn, n * sizeof(int));
            cudaMalloc((void**)&tempOut, n * sizeof(int));

            cudaMemcpy(tempIn, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int blockSize = 128;
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            
            for (int d = 1; d <= ilog2ceil(n); ++d) {
                naiveScan << < blocksPerGrid, blockSize >> > (n, tempOut, tempIn, d);
                std::swap(tempOut, tempIn);
            }

            exclusiveScan << < blocksPerGrid, blockSize >> > (n, tempOut, tempIn);

            timer().endGpuTimer();

            cudaMemcpy(odata, tempOut, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(tempIn);
            cudaFree(tempOut);
        }
    }
}
