#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
//#include <CL/opencl.h>
#include <sys/time.h>

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_half.h"

#define EXTN 0

#if EXTN
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// hardware specification
#define UNROLL 16
#define DSIZE 32
#define HWFIn UNROLL
#define HWFOut UNROLL
#define HWFinR 30
#define HWFinC 226
#define HWFR 28
#define HWFC 224
#define HWKsize 3
#define HWKstride 1
#define FC 1
#define CONV 0

#define HWin HWFIn
#define HWout HWFOut
#define BitW DSIZE*UNROLL

//typedef ap_fixed<32, 16> data_t;
//typedef float data_t;

//typedef ap_int<16> data_t;

//typedef ap_fixed<20, 16> intm_t;
//typedef ap_fixed<18, 16> data_t;
//typedef ap_fixed<25, 2> wght_t;
//typedef ap_fixed<32, 16> if_data_t;
//typedef ap_fixed<32, 2> if_wght_t;

typedef float intm_t;
typedef float data_t;
typedef float wght_t;
typedef float if_data_t;
typedef float if_wght_t;

typedef ap_uint<16> idx_t_l;
typedef ap_uint<8>  idx_t_s;

//typedef half intm_t;
//typedef half data_t;
//typedef half wght_t;

typedef ap_uint<BitW> bw_t;
typedef ap_uint<512> ly_t;
//typedef int bw_t;

//typedef int data_t;
#define POOL 1
#define RELU 1
////////////////////////////////////////////////////////////////////////////////
// cnn module specification
#define READ 1
#define WRITE 0

/*
#define DATA_R 224
#define DATA     3
#define CONV11   64
#define CONV12   64
#define POOL12   64
#define CONV21   128
#define CONV22   128
#define POOL22   128
#define CONV31   256
#define CONV32   256
#define CONV33   256
#define POOL33   256
#define CONV41   512
#define CONV42   512
#define CONV43   512
#define POOL43   512
#define CONV51   512
#define CONV52   512
#define CONV53   512
#define POOL53   512
#define PAD      1
#define STRIDE   1
#define KSIZE    3
*/

#if EXTN
} //extern "C"
#endif
