#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

#define blockSize 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // sweep them up
        __global__ void kernelUpSweep(int n, int* data, int step) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);

            if (idx >= n / step) return;

            idx *= step;
            data[idx + step - 1] += data[idx + (step >> 1) - 1];
        }

        // sweep them down
        __global__ void kernelDownSweep(int n, int* data, int step) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);

            if (idx >= n / step) return;

            idx *= step;
            int temp = data[idx + (step >> 1) - 1];
            data[idx + (step >> 1) - 1] = data[idx + step - 1];
            data[idx + step - 1] += temp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata, bool fromCompact) {
            if (n <= 0) {
                return;
            }

            int logn = ilog2ceil(n);
            int nPadded = 1 << logn;

            int* dev_data;
            cudaMalloc((void**)&dev_data, nPadded * sizeof(int));
            checkCUDAError("scan: cudaMalloc for dev_data failed");

            // copy direction is based on where it's coming from
            cudaMemcpyKind kind = fromCompact ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
            cudaMemcpy(dev_data, idata, n * sizeof(int), kind);
            checkCUDAError("scan: Initial cudaMemcpy failed");

            // make the padded region 0 (if it exists)
            if (nPadded > n) {
                cudaMemset(dev_data + n, 0, (nPadded - n) * sizeof(int));
                checkCUDAError("scan: cudaMemset for padding failed");
            }

            if (!fromCompact) {
                timer().startGpuTimer();
            }

            // Up up sweep
            for (int d = 0; d < logn; ++d) {
                int step = 1 << (d + 1);
                int numThreads = nPadded / step;
                dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);
                kernelUpSweep << <blocksPerGrid, blockSize >> > (nPadded, dev_data, step);
            }

            // exclusive scan, last element is 0
            cudaMemset(dev_data + nPadded - 1, 0, sizeof(int));

            // Down down sweet
            for (int d = logn - 1; d >= 0; --d) {
                int step = 1 << (d + 1);
                int numThreads = nPadded / step;
                dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);
                kernelDownSweep << <blocksPerGrid, blockSize >> > (nPadded, dev_data, step);
            }

            if (!fromCompact) {
                timer().endGpuTimer();
            }

            // Copy result
            cudaMemcpyKind finalKind = fromCompact ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
            cudaMemcpy(odata, dev_data, n * sizeof(int), finalKind);
            checkCUDAError("scan: Final cudaMemcpy failed");

            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         */
        int compact(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return 0;
            }

            int logn = ilog2ceil(n);
            int nPadded = 1 << logn;
            const size_t padded_bytes = nPadded * sizeof(int);

            // buffers setup
            int* dev_idata;
            int* dev_Bools;
            int* dev_odata;
            int* scanData;
            cudaMalloc((void**)&dev_idata, padded_bytes);
            cudaMalloc((void**)&dev_Bools, padded_bytes);
            cudaMalloc((void**)&dev_odata, padded_bytes);
            cudaMalloc((void**)&scanData, padded_bytes);
            checkCUDAError("compact: cudaMalloc failed");

            // Copy host data to device and pad with zeros
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (nPadded > n) {
                cudaMemset(dev_idata + n, 0, (nPadded - n) * sizeof(int));
            }

            timer().startGpuTimer();

            int gridSize = (nPadded + blockSize - 1) / blockSize;

            // Step 1: mark en
            Common::kernMapToBoolean << <gridSize, blockSize >> > (nPadded, dev_Bools, dev_idata);

            // Step 2: scan em
            scan(nPadded, scanData, dev_Bools, true);

            // Step 3: scatter em
            Common::kernScatter << <gridSize, blockSize >> > (nPadded, dev_odata, dev_idata, dev_Bools, scanData);

            timer().endGpuTimer();

            // Get the final count of non-zero elements from the last valid element
            int lastBool = 0, lastScan = 0;
            if (n > 0) {
                cudaMemcpy(&lastBool, dev_Bools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastScan, scanData + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            }
            int count = lastBool + lastScan;

            // Copy the final compacted array back to the host
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);

            // LET THEM BE FREEEE
            cudaFree(dev_Bools);
            cudaFree(scanData);
            cudaFree(dev_idata);
            cudaFree(dev_odata);

            return count;
        }
    }
}