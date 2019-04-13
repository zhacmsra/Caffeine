/**
 * Xilinx SDAccel HAL userspace driver APIs
 * Copyright (C) 2015, Xilinx Inc - All rights reserved
 */

#ifndef _XCL_HAL2_H_
#define _XCL_HAL2_H_

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


#include "xclbin.h"
//#include "xclperf.h"

#define XAPM_MAX_NUMBER_SLOTS     8
#define XAPM_MAX_TRACE_BIT_WIDTH  512
#define XAPM_MAX_NUMBER_FIFOS     (XAPM_MAX_TRACE_BIT_WIDTH/32)
#define MAX_TRACE_NUMBER_SAMPLES  8192

#ifdef __cplusplus
extern "C" {
#endif

    typedef void * xclDeviceHandle;

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
        float mTemp;
        float mVoltage;
        float mCurrent;
        // More properties here
    };

    struct xclDeviceStats {
        uint64_t mBufferObjectWriteSize;
        uint64_t mBufferObjectReadSize;
        unsigned mBufferObjectWriteCount;
        unsigned mBufferObjectReadCount;
        unsigned mBufferObjectSize;
        unsigned mPeakBufferObjectSize;
        unsigned mBufferObjectCount;
    };

    /**
     * Define address spaces on the device AXI bus. The enums are used in xclRead() and xclWrite()
     * to pass relative offsets.
     */

    enum xclAddressSpace {
        XCL_ADDR_SPACE_DEVICE_FLAT = 0,     // Absolute address space
        XCL_ADDR_SPACE_DEVICE_RAM = 1,      // Address space for the DDR memory
        XCL_ADDR_KERNEL_CTRL = 2,           // Address space for the OCL Region control port
        XCL_ADDR_SPACE_DEVICE_PERFMON = 3,  // Address space for the Performance monitors
        XCL_ADDR_SPACE_MAX = 8
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

    enum xclBufferObjectDomain {
        XCL_BO_DEVICE_RAM,
        XCL_BO_DEVICE_BRAM,
        XCL_BO_SHARED_VIRTUAL,
        XCL_BO_SHARED_PHYSICAL
    };

    /**
     * Defines Buffer Object which represents a fragment of device accesible memory and the
     * corresponding backing host memory.

     * 1. Shared virtual memory (SVM) class of systems like CAPI
     *    XCL_BO_SHARED_VIRTUAL
     *    mHostAddr allocated by the HAL driver
     *    mDeviceAddr will point to mHostAddr
     *
     * 2. Shared physical memory class of systems like MPSoc
     *    XCL_BO_SHARED_PHYSICAL
     *    mDeviceAddr allocated by the HAL driver using Linux CMA to point to physcial pages
     *    mHostAddr would point to the virtual address for the corresponding physical pages
     *
     * 3. Dedicated memory class of devices like PCIe card with DDR
     *    XCL_BO_DEVICE_RAM
     *    mDeviceAddr allocated by the HAL driver
     *    mHostAddr will be allocated as backing store to support xclMapBuffer
     *    mHostAddr may also be allocated to support xclWrite[Read]BuffferObject() for platforms
     *    with requirements for DMA buffers like: strict alignment of user pointer, pinned pages
     *    for DMA buffers, etc
     *
     * 4. Dedicated onchip memory class of devices like PCIe card with BRAM
     *    XCL_BO_DEVICE_BRAM
     *    mDeviceAddr allocated by the HAL driver
     *    mHostAddr will be allocated as backing store to support xclMapBuffer
     *    mHostAddr may also be allocated to support xclWrite[Read]BuffferObject() for platforms
     *    with requirements for DMA buffers like: strict alignment of user pointer, pinned pages
     *    for DMA buffers, etc
     */
    struct xclBufferObject {
        uint64_t mDeviceAddr;
        void *mHostAddr;
        size_t mSize;
        xclBufferObjectDomain mDomain;
        unsigned mFlags;
        unsigned mRefCount;
        xclDeviceHandle *mOwner;
    };

    struct xclWaitHandle {
        /* TODO */
        void *foo;
    };
    /**
     * @defgroup devman DEVICE MANAGMENT APIs
     * --------------------------------------
     * APIs to open, close, query and program the device
     * @{
     */

    /**
     * Return a count of devices found in the system
     */
    XCL_DRIVER_DLLESPEC unsigned xclProbe();

    /**
     * Open a device and obtain its handle.
     * "deviceIndex" is 0 for first device, 1 for the second device and so on
     * "logFileName" is optional and if not NULL should be used to log messages
     * "level" specifies the verbosity level for the messages being logged to logFileName
     */

    XCL_DRIVER_DLLESPEC xclDeviceHandle xclOpen(unsigned deviceIndex, const char *logFileName,
                                                xclVerbosityLevel level);

    /**
     * Close an opened device
     */

    XCL_DRIVER_DLLESPEC void xclClose(xclDeviceHandle handle);

    /**
     * Reset the device. All running kernels will be killed and buffers in DDR will be
     * purged. A device would be reset if a user's application dies without waiting for
     * running kernel(s) to finish.
     */

    XCL_DRIVER_DLLESPEC int xclResetDevice(xclDeviceHandle handle, xclResetKind kind);

    /**
     * Obtain various bits of information from the device
     */

    XCL_DRIVER_DLLESPEC int xclGetDeviceInfo(xclDeviceHandle handle, xclDeviceInfo *info);

    /**
     * Download bitstream to the device. The bitstream is encapsulated inside xclBin.
     * The bitstream may be PR bistream for devices which support PR and full bitstream
     * for devices which require full configuration.
     */

    XCL_DRIVER_DLLESPEC int xclLoadXclBin(xclDeviceHandle handle, const xclBin *buffer);

    /** @} */

    /**
     * @defgroup bufman BUFFER MANAGMENT APIs
     * --------------------------------------
     *
     * Buffer management APIs are used for managing device memory. The board vendors are expected to
     * provide a memory manager with the following 5 APIs. The xclWriteBuffer/xclReadBuffer functions
     * will be used by runtime to migrate buffers between host and device memory. These two APIs are
     * supposed to be non blocking. The client can wait for these operations to finish by calling
     * xclWait on the returned handle.
     * @{
     */

    /**
     * Allocate a buffer on the device DDR and return its address
     */

    XCL_DRIVER_DLLESPEC xclBufferObject *xclAllocBufferObject(xclDeviceHandle handle, size_t size);

    /**
     * Free a previously allocated buffer on the device DDR
     */

    XCL_DRIVER_DLLESPEC void xclFreeBufferObject(xclDeviceHandle handle, xclBufferObject *bo);

    /**
     * Copy host buffer contents to previously allocated device memory. "seek" specifies how many bytes
     * to skip at the beginning of the destination before copying "size" bytes of host buffer.
     */

    XCL_DRIVER_DLLESPEC size_t xclWriteBufferObject(xclDeviceHandle handle, xclBufferObject *dest,
                                                    const void *src, size_t size, size_t seek);

    /**
     * Copy contents of previously allocated device memory to host buffer. "skip" specifies how many bytes
     * to skip from the beginning of the source before copying "size" bytes of device buffer.
     */

    XCL_DRIVER_DLLESPEC size_t xclReadBufferObject(xclDeviceHandle handle, void *dest,
                                                   xclBufferObject *src, size_t size, size_t skip);

    /**
     * Map the contents of the buffer object into host memory. "size" specifies how many bytes
     * to map and "offset" specifies how many bytes to skip from the beginning of the device buffer.
     */

    XCL_DRIVER_DLLESPEC size_t xclMapBuffer(xclDeviceHandle handle, xclBufferObject *src, size_t size,
                                            size_t offset);


    XCL_DRIVER_DLLESPEC size_t xclUnMapBuffer(xclDeviceHandle handle, xclBufferObject *src);

    XCL_DRIVER_DLLESPEC void xclWait(xclWaitHandle *handle);
    /** @} */

    /**
     * @defgroup readwrite DEVICE READ AND WRITE APIs
     * ----------------------------------------------
     *
     * These functions are used to read and write peripherals sitting on the address map.  OpenCL runtime will be using the BUFFER MANAGEMNT
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

    XCL_DRIVER_DLLESPEC void xclSetOclRegionProfilingNumberSlots(xclDeviceHandle handle, uint32_t numSlots);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonClockTraining(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonStartCounters(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonStopCounters(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonReadCounters(xclDeviceHandle handle, xclPerfMonType type, xclCounterResults& counterResults);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonStartTrace(xclDeviceHandle handle, xclPerfMonType type, uint32_t startTrigger);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonStopTrace(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC uint32_t xclPerfMonGetTraceCount(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonReadTrace(xclDeviceHandle handle, xclPerfMonType type, xclTraceResultsVector& traceVector);

    /* Performance monitor type or location */
    enum xclPerfMonType {
      XCL_PERF_MON_MEMORY = 0,
      XCL_PERF_MON_HOST_INTERFACE = 1,
      XCL_PERF_MON_OCL_REGION = 2,
      XCL_PERF_MON_TOTAL_PROFILE = 3
    };

    /* Performance monitor counter results */
    /* NOTE: this is 260 bytes */
    struct xclCounterResults {
      float mSampleIntervalUsec;
      uint32_t mWriteBytes[XAPM_MAX_NUMBER_SLOTS];
      uint32_t mWriteTranx[XAPM_MAX_NUMBER_SLOTS];
      uint32_t mWriteLatency[XAPM_MAX_NUMBER_SLOTS];
      uint16_t mWriteMinLatency[XAPM_MAX_NUMBER_SLOTS];
      uint16_t mWriteMaxLatency[XAPM_MAX_NUMBER_SLOTS];
      uint32_t mReadBytes[XAPM_MAX_NUMBER_SLOTS];
      uint32_t mReadTranx[XAPM_MAX_NUMBER_SLOTS];
      uint32_t mReadLatency[XAPM_MAX_NUMBER_SLOTS];
      uint16_t mReadMinLatency[XAPM_MAX_NUMBER_SLOTS];
      uint16_t mReadMaxLatency[XAPM_MAX_NUMBER_SLOTS];
    };

    /* Performance monitor trace results */
    /* NOTE: this is 139 bytes */
    struct xclTraceResults {
      uint8_t  mLogID; /* 0: event flags, 1: host timestamp */
      uint8_t  mOverflow;
      uint8_t  mWriteStartEvent;
      uint8_t  mWriteEndEvent;
      uint8_t  mReadStartEvent;
      uint16_t mTimestamp;
      uint32_t mHostTimestamp;
      uint8_t  mRID[XAPM_MAX_NUMBER_SLOTS];
      uint8_t  mARID[XAPM_MAX_NUMBER_SLOTS];
      uint8_t  mBID[XAPM_MAX_NUMBER_SLOTS];
      uint8_t  mAWID[XAPM_MAX_NUMBER_SLOTS];
      uint8_t  mEventFlags[XAPM_MAX_NUMBER_SLOTS];
      uint8_t  mExtEventFlags[XAPM_MAX_NUMBER_SLOTS];
      uint8_t  mWriteAddrLen[XAPM_MAX_NUMBER_SLOTS];
      uint8_t  mReadAddrLen[XAPM_MAX_NUMBER_SLOTS];
      uint16_t mWriteBytes[XAPM_MAX_NUMBER_SLOTS];
      uint16_t mReadBytes[XAPM_MAX_NUMBER_SLOTS];
      uint16_t mWriteAddrId[XAPM_MAX_NUMBER_SLOTS];
      uint16_t mReadAddrId[XAPM_MAX_NUMBER_SLOTS];
    };

    /* Complete performance monitor trace results */
    /* NOTE: this is 1.1 MB */
    struct xclTraceResultsVector {
      unsigned mLength;
      xclTraceResults mArray[MAX_TRACE_NUMBER_SAMPLES];
    };

    /* Performance monitor trace offload type */
    enum xclPerfMonTraceOffload {
      XCL_TRACE_AXILITE = 0,
      XCL_TRACE_AXIMM = 1
    };

    /* Catch-all for performance monitor configuration */
    /* NOTES: * this is on a per-APM basis
     *        * only Trace and/or Profile modes are supported */
    struct xclPerfMonConfig {
      bool mHasTrace;
      bool mHasCounters;
      bool mShowIDs;
      bool mShowLengths;
      uint8_t mNumberSlots;
      uint8_t mNumberFifos;
      uint32_t mNumberSamples;
      uint32_t mTraceWordWidth;
      uint64_t mBaseAddress;
      xclPerfMonTraceOffload mTraceOffload;
      uint64_t mTraceOffsetAddress[XAPM_MAX_NUMBER_FIFOS];
      std::string mSlotNames[XAPM_MAX_NUMBER_SLOTS];
      uint32_t mSlotDataWidth[XAPM_MAX_NUMBER_SLOTS];
    };

    /** @} */

#ifdef __cplusplus
}
#endif

#endif

// XSIP watermark, do not delete 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
