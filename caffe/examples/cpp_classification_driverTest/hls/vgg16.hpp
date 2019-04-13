#include "cnn_cfg.hpp"
#include "falconML.hpp"

#if EXTN
extern "C" {
#endif


void vgg16(ap_uint<BitW> *m_fm, ap_uint<512> *lyinf)
{ 
#pragma HLS interface m_axi offset=slave depth=2000000 port=m_fm bundle=m_1
#pragma HLS interface m_axi offset=slave depth=2000 port=lyinf bundle=m_2
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=m_fm bundle=control
#pragma HLS INTERFACE s_axilite port=lyinf bundle=control

static data_t in[HWFIn][HWFinR][HWFinC];
static data_t Cout[HWFOut][HWFR][HWFC];
static wght_t weight[HWFOut][HWFIn][HWKsize][HWKsize];
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=Cout complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2

static data_t in_1[HWFIn][HWFinR][HWFinC];
static wght_t weight_1[HWFOut][HWFIn][HWKsize][HWKsize];
#pragma HLS ARRAY_PARTITION variable=in_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_1 complete dim=2

int ly[128][16]; 
#pragma HLS ARRAY_PARTITION variable=ly complete dim=2

#pragma HLS RESOURCE variable=in core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=in_1 core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=Cout core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=weight_1 core=RAM_2P_LUTRAM
#pragma HLS RESOURCE variable=weight core=RAM_2P_LUTRAM


ap_uint<512> tmp = lyinf[0]; 
int num_layer = 0; 
int start_num = 0; 

for(int j=0; j<16; j++){
    int data = tmp.range(32*(j+1)-1, 32*j);
    ly[0][j] = (int)data;
}
if(ly[0][0]==1){ //FCN
    num_layer = ly[0][12];
    start_num = ly[0][13];
}
else{   //CONV
    num_layer = ly[0][14];
    start_num = ly[0][15];
}

for(int i=1; i<num_layer + 1; i++){
#pragma HLS pipeline
   tmp = lyinf[i + start_num -1]; 
   for(int j=0; j<16; j++){
       int data = tmp.range(32*(j+1)-1, 32*j);
       ly[i][j] = (int)data;
   }
}


for(int i=1; i<num_layer+1; i++) {

ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, weight_1, ly[i][0], ly[i][1], ly[i][4], ly[i][5], ly[i][2], ly[i][3], ly[i][6], ly[i][7], ly[i][8], (bool)ly[i][9], m_fm, ly[i][10], ly[i][11], ly[i][12], ly[i][15], ly[i][13], ly[i][14]);

//m_fm[ ly[i][12]/UNROLL ] = tt;

}

} // brace of main


#if EXTN
}
#endif

