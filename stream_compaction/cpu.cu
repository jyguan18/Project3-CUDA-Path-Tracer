#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                odata[i] = idata[i - 1] + odata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int oIdx = 0;

            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[oIdx] = idata[i];
                    oIdx++;
                }
            }
            timer().endCpuTimer();
            return oIdx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            int* tempData = new int[n];
            int* scanData = new int[n];

            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    tempData[i] = 1;
                }
                else {
                    tempData[i] = 0;
                }
            }

            scanData[0] = 0;
            for (int i = 1; i < n; ++i) {
                scanData[i] = tempData[i - 1] + scanData[i - 1];
            }

            for (int i = 0; i < n; ++i) {
                if (tempData[i] != 0) {
                    odata[scanData[i]] = idata[i];
                }
            }

            timer().endCpuTimer();

            int scanResult = tempData[n - 1] + scanData[n - 1];

            delete[] tempData;
            delete[] scanData;

            return scanResult;
        }


    }
}
