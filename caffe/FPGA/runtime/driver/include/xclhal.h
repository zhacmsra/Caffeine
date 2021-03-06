/**
 * Xilinx SDAccel HAL userspace driver APIs
 * Copyright (C) 2015, Xilinx Inc - All rights reserved
 */

#ifndef _XCL_HAL_H_
#define _XCL_HAL_H_

#ifdef __cplusplus
#include <cstdlib>
#include <cstdint>
#else
#include <stdlib.h>
#include <stdint.h>
#endif

#if defined(_WIN32)
#ifdef XCL_DRIVER_DLL_EXPORT
#define XCL_DRIVER_DLLESPEC __declspec(dllexport)
#else
#define XCL_DRIVER_DLLESPEC __declspec(dllimport)
#endif
#else
#define XCL_DRIVER_DLLESPEC __attribute__((visibility("default")))
#endif


#include "xclperf.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef void * xclDeviceHandle;

    struct xclBin;
    /**
     * Structure used to obtain various bits of information from the device.
     */

    struct xclDeviceInfo {
        unsigned mMagic; // = 0X586C0C6C; XL OpenCL X->58(ASCII), L->6C(ASCII), O->0 C->C L->6C(ASCII);
        char mName[256];
        unsigned short mHALMajorVersion;
        unsigned short mHALMinorVersion;
        unsigned short mVendorId;
        unsigned short mDeviceId;
        unsigned mDeviceVersion;
        unsigned short mSubsystemId;
        unsigned short mSubsystemVendorId;
        size_t mDDRSize;                    // Size of DDR memory
        size_t mDataAlignment;              // Minimum data alignment requirement for host buffers
        size_t mDDRFreeSize;                // Total unused/available DDR memory
        size_t mMinTransferSize;            // Minimum DMA buffer size
        float mTemp;
        float mVoltage;
        float mCurrent;
        unsigned mDDRBankCount;
        unsigned mOCLFrequency;
        unsigned mPCIeLinkWidth;
        unsigned mPCIeLinkSpeed;
        // More properties here
    };

    enum xclMemoryDomains {
        XCL_MEM_HOST_RAM =    0x00000000,
        XCL_MEM_DEVICE_RAM =  0x00000001,
        XCL_MEM_DEVICE_BRAM = 0x00000002,
        XCL_MEM_SVM =         0x00000003,
        XCL_MEM_CMA =         0x00000004
    };

    enum xclDDRFlags {
        XCL_DEVICE_RAM_BANK0 = 0,
        XCL_DEVICE_RAM_BANK1 = 1,
        XCL_DEVICE_RAM_BANK2 = 2,
        XCL_DEVICE_RAM_BANK3 = 3
    };

    enum xclBRAMFlags {
        XCL_DEVICE_BRAM0 = 0,
        XCL_DEVICE_BRAM1 = 1,
        XCL_DEVICE_BRAM2 = 2,
        XCL_DEVICE_BRAM3 = 3,
    };

    /**
     * Define address spaces on the device AXI bus. The enums are used in xclRead() and xclWrite()
     * to pass relative offsets.
     */

    enum xclAddressSpace {
        XCL_ADDR_SPACE_DEVICE_FLAT =    0,  // Absolute address space
        XCL_ADDR_SPACE_DEVICE_RAM =     1,  // Address space for the DDR memory
        XCL_ADDR_KERNEL_CTRL =          2,  // Address space for the OCL Region control port
        XCL_ADDR_SPACE_DEVICE_PERFMON = 3,  // Address space for the Performance monitors
        XCL_ADDR_SPACE_MAX =            8
    };

    /**
     * Defines verbosity levels which are passed to xclOpen during device creation time
     */

    enum xclVerbosityLevel {
        XCL_QUIET = 0,
        XCL_INFO = 1,
        XCL_WARN = 2,
        XCL_ERROR = 3
    };

    enum xclResetKind {
        XCL_RESET_KERNEL,
        XCL_RESET_FULL
    };

    // VERSION 1.0 APIs
    // ----------------

    /**
     * @defgroup devman DEVICE MANAGMENT APIs
     * --------------------------------------
     * APIs to open, close, query and program the device
     * @{
     */

    /**
     * Open a device and obtain its handle.
     * "deviceIndex" is 0 for first device, 1 for the second device and so on
     * "logFileName" is optional and if not NULL should be used to log messages
     * "level" specifies the verbosity level for the messages being logged to logFileName
     */

    XCL_DRIVER_DLLESPEC xclDeviceHandle xclOpen(unsigned deviceIndex, const char *logFileName, xclVerbosityLevel level);

    /**
     * Close an opened device
     */

    XCL_DRIVER_DLLESPEC void xclClose(xclDeviceHandle handle);

    /**
     * Obtain various bits of information from the device
     */

    XCL_DRIVER_DLLESPEC int xclGetDeviceInfo(xclDeviceHandle handle, xclDeviceInfo *info);

    /**
     * Download bitstream to the device. The bitstream is in xclBin format and stored in xclBinFileName.
     * The bitstream may be PR bistream for devices which support PR and full bitstream for devices
     * which require full configuration.
     */

    XCL_DRIVER_DLLESPEC int xclLoadBitstream(xclDeviceHandle handle, const char *xclBinFileName);

    /**
     * Download bitstream to the device. The bitstream is passed in memory in xclBin format. The bitstream
     * may be PR bistream for devices which support PR and full bitstream for devices which require full
     * configuration.
     */

    XCL_DRIVER_DLLESPEC int xclLoadXclBin(xclDeviceHandle handle, const xclBin *buffer);

    /** @} */

    /**
     * @defgroup bufman BUFFER MANAGMENT APIs
     * --------------------------------------
     *
     * Buffer management APIs are used for managing device memory. The board vendors are expected to
     * provide a memory manager with the following 4 APIs. The xclCopyXXX functions will be used by
     * runtime to migrate buffers between host and device memory.
     * @{
     */

    /**
     * Allocate a buffer on the device DDR and return its address
     */

    XCL_DRIVER_DLLESPEC uint64_t xclAllocDeviceBuffer(xclDeviceHandle handle, size_t size);

    /**
     * Allocate a buffer on the device DDR bank and return its address
     */

    XCL_DRIVER_DLLESPEC uint64_t xclAllocDeviceBuffer2(xclDeviceHandle handle, size_t size,
                                                       xclMemoryDomains domain,
                                                       unsigned flags);

    /**
     * Free a previously allocated buffer on the device DDR
     */

    XCL_DRIVER_DLLESPEC void xclFreeDeviceBuffer(xclDeviceHandle handle, uint64_t buf);

    /**
     * Copy host buffer contents to previously allocated device memory. "seek" specifies how many bytes to skip
     * at the beginning of the destination before copying "size" bytes of host buffer.
     */

    XCL_DRIVER_DLLESPEC size_t xclCopyBufferHost2Device(xclDeviceHandle handle, uint64_t dest,
                                                        const void *src, size_t size, size_t seek);

    /**
     * Copy contents of previously allocated device memory to host buffer. "skip" specifies how many bytes to skip
     * from the beginning of the source before copying "size" bytes of device buffer.
     */

    XCL_DRIVER_DLLESPEC size_t xclCopyBufferDevice2Host(xclDeviceHandle handle, void *dest,
                                                        uint64_t src, size_t size, size_t skip);

    /** @} */

    /**
     * @defgroup readwrite DEVICE READ AND WRITE APIs
     * ----------------------------------------------
     *
     * These functions are used to read and write peripherals sitting on the address map. An implementation
     * may use these to implement xclCopyXXX functions. OpenCL runtime will be using the BUFFER MANAGEMNT
     * APIs described above to manage OpenCL buffers. It would use xclRead/xclWrite to program and manage
     * peripherals on the card. For programming the Kernel, OpenCL runtime uses the kernel control register
     * map generated by the OpenCL compiler.
     * Note that the offset is wrt the address space
     * @{
     */

    XCL_DRIVER_DLLESPEC size_t xclWrite(xclDeviceHandle handle, xclAddressSpace space, uint64_t offset,
                                        const void *hostBuf, size_t size);

    XCL_DRIVER_DLLESPEC size_t xclRead(xclDeviceHandle handle, xclAddressSpace space, uint64_t offset,
                                       void *hostbuf, size_t size);

    /** @} */

    // EXTENSIONS FOR PARTIAL RECONFIG FLOW
    // ------------------------------------
    // Update the device PROM with new base bitsream
    XCL_DRIVER_DLLESPEC int xclUpgradeFirmware(xclDeviceHandle handle, const char *fileName);

    // Boot the FPGA with new bitsream in PROM. This will break the PCIe link and render the device
    // unusable till a reboot of the host
    XCL_DRIVER_DLLESPEC int xclBootFPGA(xclDeviceHandle handle);

    // NEW APIs in VERSION 1.1
    // -----------------------

    /**
     * @addtogroup devman
     * @{
     */

    /**
     * Reset the device. All running kernels will be killed and buffers in DDR will be purged.
     * A device would be reset if a user's application dies without waiting for running kernel(s) to finish.
     */

    XCL_DRIVER_DLLESPEC int xclResetDevice(xclDeviceHandle handle, xclResetKind kind);

    /**
     * Set the OCL region frequncy
     */

    XCL_DRIVER_DLLESPEC int xclReClock(xclDeviceHandle handle, unsigned targetFreqMHz);

    /**
     * Return a count of devices found in the system
     */
    XCL_DRIVER_DLLESPEC unsigned xclProbe();

    /**
     * Get exclusive ownership of the device. The lock is necessary before performing buffer
     * migration, register access or bitstream downloads.
     */
    XCL_DRIVER_DLLESPEC int xclLockDevice(xclDeviceHandle handle);

    /** @} */

    /**
     * @defgroup perfmon PERFORMANCE MONITORING OPERATIONS
     * ---------------------------------------------------
     *
     * These functions are used to read and write to the performance monitoring infrastructure.
     * OpenCL runtime will be using the BUFFER MANAGEMNT APIs described above to manage OpenCL buffers.
     * It would use these functions to initialize and sample the performance monitoring on the card.
     * Note that the offset is wrt the address space
     */

    XCL_DRIVER_DLLESPEC size_t xclGetDeviceTimestamp(xclDeviceHandle handle);

    XCL_DRIVER_DLLESPEC double xclGetDeviceClockFreqMHz(xclDeviceHandle handle);

    XCL_DRIVER_DLLESPEC double xclGetReadMaxBandwidthMBps(xclDeviceHandle handle);

    XCL_DRIVER_DLLESPEC double xclGetWriteMaxBandwidthMBps(xclDeviceHandle handle);

    XCL_DRIVER_DLLESPEC void xclSetOclRegionProfilingNumberSlots(xclDeviceHandle handle,
                                                                 uint32_t numSlots);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonClockTraining(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonStartCounters(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonStopCounters(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonReadCounters(xclDeviceHandle handle, xclPerfMonType type,
                                                      xclCounterResults& counterResults);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonStartTrace(xclDeviceHandle handle, xclPerfMonType type,
                                                    uint32_t startTrigger);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonStopTrace(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC uint32_t xclPerfMonGetTraceCount(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonReadTrace(xclDeviceHandle handle, xclPerfMonType type,
                                                   xclTraceResultsVector& traceVector);

    /** @} */

#ifdef __cplusplus
}
#endif

#endif

// XSIP watermark, do not delete 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
