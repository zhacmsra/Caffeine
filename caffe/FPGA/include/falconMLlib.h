#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <stdexcept>

#include <caffe/caffe.hpp>
#include <CL/opencl.h>

#define SIM 0
#define DOUBLE 0
#define TIME 1
#define TIME_VERBOSE 0

using namespace caffe;
using std::vector;
using std::string;
using std::ofstream;
using std::ifstream;
using std::endl;
using std::cout;

typedef struct fclyr{
int kk;
int k;
} fclayer;

typedef struct lyr
{
    int fin;
    int fout;
    int finrow;
    int fincol;
    int frow;
    int fcol;
    int Ksize;
    int Kstride;
    int pad;
    int mask;
    int addr_in;
    int addr_wght;
    int addr_out;
    int pool_kernel;
    int pool;
    int relu;
} layer;

typedef struct layer_config {
    string type;
    int num_input;
    int num_output;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int kernel_size;
    int kernel_stride;
    int kernel_pad;
    int pool_kernel;
} lycfg;

float rcmp(float a, float b){
    return fabs((a-b)/(a+b));
}
float get_time(timeval start, timeval end){
return ((end.tv_sec-start.tv_sec)*1e+3 + (end.tv_usec-start.tv_usec)*1e-03); 
}

class CNN4FPGA {
    public:
        CNN4FPGA(){
            initialized = false;
        }

        int layerdef_len(){
            return layerdef_length;
        }
        int weight_len(){
            return wght_length ;
        }
        int fm_len(){
            return fm_length;
        }
        int infm_len(){
            return infm_length;
        }
        int lastfm_len(){
            return lastfm_length;
        }
        int fcn_wght_len(){
            return wght_fcn_len;
        }
        int fcn_fm_len(){
            return fm_fcn_len;
        }
        int fcn_input_len(){
            return in_fcn_len; 
        }
        int fcn_output_len(){
            return out_fcn_len;
        }

        shared_ptr< float > weight_ptr(){ 
            return wght_dram;
        }
        shared_ptr< float > weight_ann_ptr(){ 
            return wght_dram_ann;
        }
        shared_ptr< int > layerdef_ptr(){
            return ly_host;
        }

        ~CNN4FPGA() {
            if(initialized == true){
      //    free(wght_dram); 
      //    free(ly_host); 
                wght_dram.reset();
                ly_host.reset();
            }
        }
    public:
        void setCNNModel(shared_ptr< Net< float > > net_);
        void ann(shared_ptr< Net< float > > net_, float* fcin, float* fcout);

        vector< lycfg > origin;
        vector< layer > net4fpga;
        vector< layer > ann_net;
    private:
        vector< lycfg > ParseCNN_Arch(shared_ptr<Net<float> > net_);
        vector< layer > EstabNet4FPGA(vector< lycfg > layer_cfg, 
                                      int* wght_length, int* layerdef_length, int* fm_length, int* infm_length, int* lastfm_length);
        vector< layer > EstabANN4FPGA(shared_ptr< Net< float > > net_, 
                                      int conv_wt_fm_len, int* layerdef_length, int* wght_fcn_length, int* fm_fcn_length, 
                                      int* in_fcn_length, int* out_fcn_length);
        void PrepareLayerdef(vector< layer > b, vector< layer > ann, shared_ptr< int > ly_host);
        void PrepareWght(shared_ptr<Net<float> > net_, vector< lycfg > a, vector< layer > b, shared_ptr< float > wght_dram);
        void PrepareWghtANN(shared_ptr<Net<float> > net_, shared_ptr< float > wght_dram_ann);
        void PrepareWghtANN2( shared_ptr< float > wght_dram_ann);
        void reorder_ann_weight(const float* net_wght, const float* net_bias, int m, int n, float* dram, int addr);
        void print_lycfg(vector< lycfg > origin);    //for debug
        void print_layer(vector< layer > net4fpga); 
        //void verify_reorder(float* wght_dram);
        void verify_reorder(shared_ptr< float > wght_dram);
    private:
        bool initialized;
        int layerdef_length; //layer definition
        int wght_length;     //conv parameters
        int fm_length;
        int infm_length;
        int lastfm_length;
        int wght_fcn_len;    //FCN parameters
        int fm_fcn_len;
        int in_fcn_len; 
        int out_fcn_len;
        shared_ptr< float > wght_dram; //float* wght_dram; 
        shared_ptr< float > wght_dram_ann; //float* wght_dram; 
        shared_ptr<  int  > ly_host; //int* ly_host;
        int DUM;
        vector< fclayer > innerprod;
};


void CNN4FPGA::setCNNModel(shared_ptr<Net<float> > net_) {
    DUM = UNROLL;
    timeval t[5];
    //------- Step 1. Parse Network from Caffe Definition -------//
    gettimeofday(&t[0], NULL);
    origin =  ParseCNN_Arch(net_); 
    gettimeofday(&t[1], NULL);
    //print_lycfg(origin);

    //------- Step 2. Deducting Parameters& Static DRAM assignments from Layer definitions -------//
    net4fpga = EstabNet4FPGA(origin, &wght_length, &layerdef_length, &fm_length, &infm_length, &lastfm_length);
    //print_layer(net4fpga);
    ann_net  = EstabANN4FPGA(net_, (wght_length+fm_length), &layerdef_length, &wght_fcn_len, &fm_fcn_len, &in_fcn_len, &out_fcn_len);
    //print_layer(ann_net);
    //std::cout<< "wght_fcn_len\t" << "fm_fcn_len\t" << "in_fcn_len\t" << "out_fcn_len" <<std::endl;
    //std::cout<< wght_fcn_len << "\t" << fm_fcn_len  << "\t" << in_fcn_len  << "\t" << out_fcn_len <<std::endl;
    gettimeofday(&t[2], NULL);

    //------- Step 3. Prepare for FPGA layer definitions (stored in DRAM, float* ly_host) -------//
    shared_ptr< float > wght_dram_1(new float[wght_length]);
    shared_ptr< float > wght_dram_2(new float[wght_fcn_len]);
    shared_ptr< int > ly_host_1(new int[layerdef_length]);
    wght_dram     = wght_dram_1;
    wght_dram_ann = wght_dram_2;
    ly_host   = ly_host_1;
    PrepareLayerdef(net4fpga, ann_net, ly_host);
    gettimeofday(&t[3], NULL);

    //------- Step 4. Weight Reorderring for FPGA DRAM -------//
    PrepareWght(net_, origin, net4fpga, wght_dram);
    PrepareWghtANN(net_, wght_dram_ann);
    //PrepareWghtANN2(wght_dram_ann);
    gettimeofday(&t[4], NULL);

    //---------Report Profiling Time------------//
#if TIME_VERBOSE
    printf("parse layer def from Caffe: %f ms\n", get_time(t[0], t[1]));
    printf("establish model for FPGA: %f\n", get_time(t[1], t[2]));
    printf("prepare layer definition for FPGA: %f\n", get_time(t[2], t[3]));
    printf("reorder weight for FPGA: %f\n", get_time(t[3], t[4]));
#endif
    //verify_reorder(wght_dram);
    std::cout << "finish pareparing CNN model for FPGA accelerator" << std::endl;
    initialized = true; 
}

vector< lycfg > CNN4FPGA::ParseCNN_Arch(shared_ptr<Net<float> > net_) {
  //--- left for future use: input data shape and batch info from caffe ---//
  //Blob< float >* in_blob = net_->input_blobs()[0];
  //vector< int > shape = in_blob->shape(); // int shape[0] batch, shape[1] channel, shape[2] height

  //Please be noticed that 'layers for weights' in 'net->layers'
  //and 'layers for feature maps' in 'net->blobs' are different.
  //ReLu layer does not use 'net->blobs'
  vector< lycfg > result; //vector of layers parsed
  vector< shared_ptr< Layer< float > > > layers = net_->layers();      //net layers for weight kernel info and values (blob)
  vector< shared_ptr< Blob< float > > > vfmlayer = net_->blobs();      //net blobs for feature map kernel info and values
  vector< string > layer_name = net_->layer_names();
  vector< int > num;
  int kk=0;
  //printf("total layers %d: \n", (int)layer_name.size());
  for(int k=0; k<layer_name.size(); k++){
    num = vfmlayer[kk]->shape();
    LayerParameter lparam = layers[k]->layer_param();
    if( layer_name[k].compare(0, 2, "fc")==0 ) {
        //InnerProductParameter ipparam = lparam.inner_product_param();
        //int ipout = ipparam.num_output();
        //cout << layer_name[k] << " input: " << num[0] << " " << num[1] << " " << num[2] << " " << num[3] << " output: " << ipout << endl;
        fclayer tmpk;
        tmpk.kk = kk;
        tmpk.k  = k;
        innerprod.push_back(tmpk);
        break;
    }
    else if( layer_name[k].compare(0, 4, "conv")==0 ) {
        ConvolutionParameter cparam = lparam.convolution_param(); 
        lycfg t;
        t.type = "conv";
        t.num_input = (int)num[1]; 
        t.num_output = (int)(cparam.num_output());
        t.input_height =  num[2] + 2*(int)(cparam.pad(0));
        t.input_width =  num[3] + 2*(int)(cparam.pad(0));
        t.kernel_size =  (int)cparam.kernel_size(0);
        if (cparam.stride_size()==0){
            cout << "Warning: layer " << layer_name[k] << " \'stride\' field detected a value of 0, set to 1 for default. Please check with your prototxt file" << endl;
            t.kernel_stride =  1;
        }
        else
            t.kernel_stride =  (int)(cparam.stride(0));
        t.kernel_pad =  (int)(cparam.pad(0));
        t.output_height =  (t.input_height-t.kernel_size+t.kernel_stride)/t.kernel_stride;
        t.output_width =  (t.input_width-t.kernel_size+t.kernel_stride)/t.kernel_stride;

        result.push_back(t);
        kk=kk+1;
    }
    else if( layer_name[k].compare(0, 4, "pool")==0 ) {
        PoolingParameter pparam = lparam.pooling_param(); 
        lycfg t;
        t.type = "pool";
        t.num_input = (int)num[1]; 
        t.num_output = (int)num[1]; //(int)(pparam.num_output()); temporary workaround;
        t.input_height =  num[2] + 2*(int)(pparam.pad());
        t.input_width =  num[3] + 2*(int)(pparam.pad());
        t.kernel_size =  (int)pparam.kernel_size();
        t.kernel_stride =  (int)(pparam.stride());
        t.kernel_pad =  (int)(pparam.pad());
        t.output_height =  (t.input_height-t.kernel_size+t.kernel_stride)/t.kernel_stride;
        t.output_width =  (t.input_width-t.kernel_size+t.kernel_stride)/t.kernel_stride;
        result.push_back(t);
        kk=kk+1;
    }
    else if( layer_name[k].compare(0, 4, "relu")==0 ) {
        lycfg t;
        t.type = "relu";
        t.num_input = result[k-1].num_output; 
        t.num_output = result[k-1].num_output;
        t.input_height = result[k-1].output_height;
        t.input_width = result[k-1].output_width;
        t.kernel_size = 1;
        t.kernel_stride = 1;
        t.kernel_pad = 0;
        t.output_height = result[k-1].output_height;
        t.output_width = result[k-1].output_width;
        result.push_back(t);
    }
  }
  printf("finish cnn parsing\n");
  return result;
}

vector< layer > CNN4FPGA::EstabNet4FPGA(vector< lycfg> layer_cfg, int* wght_length, int* layerdef_length, int* fm_length, int* infm_length, int* lastfm_length) {
    //padding
    lycfg intmLayer;
    vector< lycfg > intmRes;
    int orig = layer_cfg.size();

    for(int i=0; i<orig; i++){
       intmLayer.type         = layer_cfg[i].type;
       intmLayer.num_input    = ((layer_cfg[i].num_input+DUM-1)/DUM)*DUM;
       intmLayer.num_output   = ((layer_cfg[i].num_output+DUM-1)/DUM)*DUM; 
       intmLayer.input_height = layer_cfg[i].input_height; 
       intmLayer.input_width  = layer_cfg[i].input_width;
       intmLayer.kernel_size  = layer_cfg[i].kernel_size;
       intmLayer.kernel_stride= layer_cfg[i].kernel_stride;
       intmLayer.kernel_pad   = layer_cfg[i].kernel_pad;
       intmLayer.output_height= layer_cfg[i].output_height;
       intmLayer.output_width = layer_cfg[i].output_width;
       intmRes.push_back(intmLayer);
    }
    int rs = intmRes.size()-1;
    //*lastfm_length = intmRes[rs].num_output*intmRes[rs].output_height*intmRes[rs].output_width;
    *infm_length = intmRes[0].num_input*\
                   (intmRes[0].input_height-2*intmRes[0].kernel_pad)*\
                   (intmRes[0].input_width-2*intmRes[0].kernel_pad);

    //transform to fpga model layer
    vector< layer > result;
    int fm_addr = 0;
    for(int i=0; i<orig; i++){
        layer r = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if(intmRes[i].type.compare("conv")==0){
            r.fin = intmRes[i].num_input; 
            r.fout = intmRes[i].num_output; 
            r.finrow = intmRes[i].input_height; 
            r.fincol = intmRes[i].input_width; 
            r.frow = intmRes[i].output_height; 
            r.fcol = intmRes[i].output_width; 
            r.Ksize = intmRes[i].kernel_size; 
            r.Kstride = intmRes[i].kernel_stride; 
            r.pad = intmRes[i].kernel_pad; 
            r.mask = 0; 
            if(i<orig-1){
                if((intmRes[i+1].type.compare("conv")!=0)) {
                    if(intmRes[i+1].type.compare("relu")==0) r.relu = 1; 
                    else if(intmRes[i+1].type.compare("pool")==0) { 
                            r.pool = 1; 
                            r.pool_kernel = intmRes[i+1].kernel_size;
                         }
                    if(i<orig-2){
                       if((intmRes[i+2].type.compare("conv")!=0)) {
                           if(intmRes[i+2].type.compare("relu")==0) r.relu = 1; 
                           else if(intmRes[i+2].type.compare("pool")==0) {
                               r.pool = 1; 
                               r.pool_kernel = intmRes[i+2].kernel_size;
                           }
                       }
                    }
                }
            }
            if(i==0){
                fm_addr = r.fin*(r.finrow-2*r.pad)*(r.fincol-2*r.pad);
            }
            fm_addr += r.fout* r.frow* r.fcol;
            result.push_back(r);
        }
    }
    *fm_length = fm_addr;
    int i = result.size()-1;
    *lastfm_length = result[i].fout*result[i].frow*result[i].fcol/result[i].pool_kernel/result[i].pool_kernel;
    int wght_addr = 0;
    for(int i=0; i<result.size(); i++) {
        result[i].addr_wght = wght_addr;
        wght_addr += (result[i].fin* result[i].fout* result[i].Ksize* result[i].Ksize + result[i].fout);
    } 
    *wght_length = wght_addr;
    *layerdef_length = (result.size()+1)*16;

    fm_addr = 0;
    result[0].addr_in = fm_addr + wght_addr;
    fm_addr += result[0].fin* result[0].frow* result[0].fcol;
    result[0].addr_out = fm_addr + wght_addr;
    for(int i=1; i<result.size(); i++) {
        result[i].addr_in = result[i-1].addr_out;
        fm_addr += result[i-1].fout* result[i-1].frow* result[i-1].fcol;
        result[i].addr_out = fm_addr + wght_addr;
    } 
    return result;
}

void CNN4FPGA::PrepareWght(shared_ptr<Net<float> > net_, vector< lycfg > a, vector< layer > b, shared_ptr< float > wght_dram){

  vector< lycfg > cast;
  for(int i=0; i<a.size(); i++){
    if(a[i].type.compare("conv")==0) {
        cast.push_back(a[i]);
    }
  }

  vector< std::string > layer_name = net_->layer_names();
  std::cout << layer_name.size() << std::endl;
  int layer_num = layer_name.size();
  int j=0;
  int addr = 0;

  //ofstream curr;
  //curr.open("layer1_curr.txt", ios::app);

  for(int i=0; i<layer_num; i++){
    if(layer_name[i].compare(0, 4, "conv")==0) {
        shared_ptr< Layer< float > > layer = net_->layers()[i];
        vector< shared_ptr< Blob< float > > > blobs = layer->blobs();
        const float * weight_ptr = (const float *) blobs[0]->cpu_data();
        const float * bias_ptr = (const float *) blobs[1]->cpu_data();
        //read weights out and add pading
        float* tmp_dram = new float[b[j].fout*b[j].fin*b[j].Ksize*b[j].Ksize]; 
        for(int u=0; u<b[j].fout; u++){
            for(int k=0; k<b[j].fin; k++){
                for(int h=0; h<b[j].Ksize*b[j].Ksize; h++){
                    if((u<cast[j].num_output)&&(k<cast[j].num_input)) {
                        tmp_dram[u*b[j].fin*b[j].Ksize*b[j].Ksize + k*b[j].Ksize*b[j].Ksize + h] =\
                                    weight_ptr[ u*cast[j].num_input*cast[j].kernel_size*cast[j].kernel_size + \
                                                k*cast[j].kernel_size*cast[j].kernel_size + h]; 

                    }
                    else {
                        tmp_dram[u*b[j].fin*b[j].Ksize*b[j].Ksize + k*b[j].Ksize*b[j].Ksize + h] = 0.0;
                    }
                }
            }
        }
        //reorder this layer's weight
        for(int uu=0; uu<b[j].fout; uu+=HWFOut){
            for(int yy=0; yy<b[j].fin; yy+=HWFIn){
                for(int k=0; k<b[j].Ksize*b[j].Ksize; k++){
                    for(int u=0; u<HWFOut; u++){
                        for(int y=0; y<HWFIn; y++){
                            wght_dram.get()[addr+ uu*b[j].fin*b[j].Ksize*b[j].Ksize + \
                                            yy* b[j].Ksize*b[j].Ksize*HWFIn + k*HWFOut*HWFIn + u*HWFIn + y] = \
                                                tmp_dram[(uu+u)*b[j].fin*b[j].Ksize*b[j].Ksize+ (yy+y)*b[j].Ksize*b[j].Ksize+ k];
                        }
                    }
                }
            }
        }
        for(int uu=0; uu<b[j].fout; uu++){
            if(uu<cast[j].num_output)
                wght_dram.get()[addr+ b[j].fout*b[j].fin*b[j].Ksize*b[j].Ksize+ uu] = bias_ptr[uu]; 
            else
                wght_dram.get()[addr+ b[j].fout*b[j].fin*b[j].Ksize*b[j].Ksize+ uu] = 0.0;
        }
        addr += (b[j].fout*b[j].fin*b[j].Ksize*b[j].Ksize + b[j].fout);
        //free(tmp_dram);
        delete [] tmp_dram;
        j++;
    }
    else if(layer_name[i].compare(0, 2, "fc")==0){
        printf("CNN4FPGA::PrepareWght fc layer\n"); 
    }
  }
  //curr.close(); 
}

vector< layer > CNN4FPGA::EstabANN4FPGA(shared_ptr<Net<float> > net_, int conv_wt_fm_len, int* layerdef_length, int* wght_fcn_length, int* fm_fcn_length, int* in_fcn_length, int* out_fcn_length) {
    printf("Establish ANN for FPGA: %d\n", conv_wt_fm_len);
	int kk = innerprod[0].kk;
	vector< shared_ptr< Layer< float > > > layers = net_->layers();      //net layers for weight kernel info and values (blob)
	vector< shared_ptr< Blob< float > > > vfmlayer = net_->blobs();      //net blobs for feature map kernel info and values
	vector< string > layer_name = net_->layer_names();
	vector< int > num = vfmlayer[kk]->shape();
	int indim = num[1]*num[2]*num[3];
	int outdim = 0;

    kk = 0;
    vector< layer > result;
	for(int k = innerprod[0].k; k<layer_name.size(); k++){
	    LayerParameter lparam = layers[k]->layer_param();
        layer r;
	    if(layer_name[k].compare(0, 2, "fc")==0){
	        shared_ptr< Layer< float > > layer = layers[k];
	        vector< shared_ptr< Blob< float > > > blobs = layer->blobs();
	        InnerProductParameter ipparam = lparam.inner_product_param();
	        outdim = ipparam.num_output();

            r.fin = indim;
            r.fout = UNROLL;
            r.fincol = (outdim<128)?outdim:128;
            r.finrow = (outdim+r.fincol-1)/r.fincol;
            r.frow = r.finrow;
            r.fcol = r.fincol;
            r.Ksize = 1;
            r.Kstride = 1;
            r.pad = 0;
            r.mask = 1;
            r.pool = 0;
            r.relu = 0;

	        indim = outdim;
            result.push_back(r);
	    }
	    else if(layer_name[k].compare(0, 4, "relu")==0){
            result[kk].relu = 1;
	        kk++;
	    }
	    //else if(layer_name[k].compare(0, 4, "prob")==0){
	    //    softmax(in, outdim);    
	    //}
	}  
    int fcn_in = 0;
    int fcn_w = 0;
    int fcn_bias = 0;
    int fcn_addr_i = 0;
    int fcn_addr_w = 0;
    int fcn_addr_b = 0;

    for(int k=0; k<result.size(); k++){
        fcn_addr_w        += (fcn_w + fcn_bias); //address of FCN weight, used as CONV input
        result[k].addr_in  = (fcn_addr_w + conv_wt_fm_len); 
        fcn_w    = result[k].fin * result[k].frow * result[k].fcol;
        fcn_addr_b = fcn_addr_w + fcn_w;        //address of FCN bias, used as CONV bias
        result[k].pool_kernel = (fcn_addr_b + conv_wt_fm_len); 
        fcn_bias = result[k].fout * result[k].frow * result[k].fcol; //pool_kernel is used actually bias
    }
    fcn_addr_i = fcn_addr_b + fcn_bias;         //address of FCN input, used as CONV input & output
    *wght_fcn_length = fcn_addr_i;              //length of FCN weight, equals to address of FCN input
    for(int k=0; k<result.size(); k++){
        result[k].addr_wght = (fcn_addr_i + conv_wt_fm_len); 
                     fcn_in = result[k].fin * result[k].fout * result[k].Ksize * result[k].Ksize;
         result[k].addr_out = (fcn_addr_i + fcn_in + conv_wt_fm_len); 
                 fcn_addr_i = fcn_addr_i + fcn_in;
    }
    layer ls = result[result.size()-1];
    *out_fcn_length  = ls.fout * ls.frow * ls.fcol;  //length of input FCN feature map
    *fm_fcn_length   = fcn_addr_i + *out_fcn_length - *wght_fcn_length; //length of all FCN feature map
    *in_fcn_length   = result[0].fin * result[0].fout * result[0].Ksize * result[0].Ksize; //length of output FCN feature map
    *layerdef_length+= result.size()*16;

    return result;
}
void CNN4FPGA::reorder_ann_weight(const float* net_wght, const float* net_bias, int m, int n, float* dram, int addr){
    int s = 0;
    int new_m = ((m+128-1)/128)*128;
    for(int i=0; i<new_m; i++){
        for(int j=0; j<n/UNROLL; j++){
            for(int k=0; k<UNROLL; k++){
                float data = 0.0;
                if(i<m){
                    data = net_wght[i*n + j*UNROLL + k];
                }
                dram[i*UNROLL+ j*UNROLL*new_m + k + addr] = data;
            }
        }
    }
    for(int j=0; j<new_m; j++){
        for(int i=0; i<UNROLL; i++){
            if(j<m)
	 	//duplicate ANN bias for the purpose of FCN2CONV transform
                dram[addr + new_m*n + j*UNROLL + i] = net_bias[j]; 
            else
                dram[addr + new_m*n + j*UNROLL + i] = 0.0;
        }
    }
}

void CNN4FPGA::PrepareWghtANN(shared_ptr<Net<float> > net_, shared_ptr< float > wght_dram_ann) {
    printf("***** PrepareWghtANN *****\n");
	int kk = innerprod[0].kk;
	vector< shared_ptr< Layer< float > > > layers = net_->layers();      //net layers for weight kernel info and values (blob)
	vector< shared_ptr< Blob< float > > > vfmlayer = net_->blobs();      //net blobs for feature map kernel info and values
	vector< string > layer_name = net_->layer_names();
	vector< int > num = vfmlayer[kk]->shape();
	int indim = num[1]*num[2]*num[3];
	int outdim = 0;
	int addr = 0;
	for(int k = innerprod[0].k; k<layer_name.size(); k++){
	    num = vfmlayer[kk]->shape();
	    LayerParameter lparam = layers[k]->layer_param();
	    //cout << "ann layers: " << layer_name[k] << endl;
	    if(layer_name[k].compare(0, 2, "fc")==0){
	        shared_ptr< Layer< float > > layer = layers[k];
	        vector< shared_ptr< Blob< float > > > blobs = layer->blobs();
	        const float * weight_ptr = (const float *) blobs[0]->cpu_data();
	        const float * bias_ptr = (const float *) blobs[1]->cpu_data();
	
	        InnerProductParameter ipparam = lparam.inner_product_param();
	        outdim = ipparam.num_output();
	        printf("The # of output: %d, # of input: %d\n", outdim, indim);
	        reorder_ann_weight(weight_ptr, bias_ptr, outdim, indim, wght_dram_ann.get(), addr);
	
            outdim = ((outdim+128-1)/128)*128;
	        addr += (outdim*indim + outdim*UNROLL); //size of weight + bias (batch size = UNROLL)
	        indim = outdim;
	        kk++;
	    }
	}
    printf("***** PrepareWghtANN *****\n");
}

//load weight feature maps, re-order under INPUT-weight schedule
void ann_weight(char* filename, int m, int n, float* dram){
    double data;
    FILE* fp = fopen(filename, "r");
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            fscanf(fp, "%lf\n", &data);
            dram[i*n + j] = (float)data;
        }
    }
    fclose(fp);
}
//load weight feature maps, re-order under INPUT-weight schedule
void ann_bias(char* filename, int m, float* dram){
    double data;
    FILE* fp = fopen(filename, "r");
    for(int i=0; i<m; i++){
        fscanf(fp, "%lf\n", &data);
        dram[i] = (float)data;
    }
    fclose(fp);
}
void CNN4FPGA::PrepareWghtANN2( shared_ptr< float > wght_dram_ann){

    float* wght1 = new float[4096*25088];
    float* wght2 = new float[4096*4096];
    float* wght3 = new float[1024*4096];
    float* bias1 = new float[4096];
    float* bias2 = new float[4096];
    float* bias3 = new float[1024];
   
    ann_weight("/curr/chenzhang/tool/caffe_fpga/MMX/weight1.txt", 4096, 25088, wght1);
    ann_weight("/curr/chenzhang/tool/caffe_fpga/MMX/weight2.txt", 4096, 4096, wght2);
    ann_weight("/curr/chenzhang/tool/caffe_fpga/MMX/weight3.txt", 1024, 4096, wght3);
    ann_bias("/curr/chenzhang/tool/caffe_fpga/MMX/bias1.txt", 4096, bias1);
    ann_bias("/curr/chenzhang/tool/caffe_fpga/MMX/bias2.txt", 4096, bias2);
    ann_bias("/curr/chenzhang/tool/caffe_fpga/MMX/bias3.txt", 1024, bias3);

    int addr = 0;
    reorder_ann_weight(wght1, bias1, 4096, 25088, wght_dram_ann.get(), addr);
    addr += 4096*25088+4096*UNROLL;
    reorder_ann_weight(wght2, bias2, 4096, 4096, wght_dram_ann.get(), addr);
    addr += 4096*4096+4096*UNROLL;
    reorder_ann_weight(wght3, bias3, 1024, 4096, wght_dram_ann.get(), addr);

    delete [] wght1;
    delete [] wght2;
    delete [] wght3;
    delete [] bias1;
    delete [] bias2;
    delete [] bias3;
}

void mmult(float* out, float* in, const float* w, const float* bias, int indim, int outdim){
    for(int i=0; i<outdim; i++){
        out[i] = bias[i];
        for(int j=0; j<indim; j++){
            out[i] += w[i*indim + j] * in[j]; 
        }
    }
}

void softmax(float *out, int length) {
    float acc=0;
    for(int i=0; i<length; i++) {
       out[i] = exp(out[i]);
       acc += out[i];
    }
    for(int i=0; i<length; i++) {
       out[i] = out[i]/acc;
    }
}
void relu(float *out, int length){
    for(int i=0; i<length; i++){
        if(out[i]<0)
            out[i] = 0;
    }
}

void CNN4FPGA::ann(shared_ptr<Net<float> > net_, float* fcin, float* fcout) {

int kk = innerprod[0].kk;
vector< shared_ptr< Layer< float > > > layers = net_->layers();      //net layers for weight kernel info and values (blob)
vector< shared_ptr< Blob< float > > > vfmlayer = net_->blobs();      //net blobs for feature map kernel info and values
vector< string > layer_name = net_->layer_names();
vector< int > num = vfmlayer[kk]->shape();
int indim = num[1]*num[2]*num[3];
float *in = new float [indim];
memcpy((void*)in, (const void*)fcin, sizeof(float)*indim);
int outdim = 0;
for(int k = innerprod[0].k; k<layer_name.size(); k++){
    num = vfmlayer[kk]->shape();
    LayerParameter lparam = layers[k]->layer_param();
    //cout << "ann layers: " << layer_name[k] << endl;
    if(layer_name[k].compare(0, 2, "fc")==0){
        shared_ptr< Layer< float > > layer = layers[k];
        vector< shared_ptr< Blob< float > > > blobs = layer->blobs();
        const float * weight_ptr = (const float *) blobs[0]->cpu_data();
        const float * bias_ptr = (const float *) blobs[1]->cpu_data();

        InnerProductParameter ipparam = lparam.inner_product_param();
        outdim = ipparam.num_output();
        float *out = new float [outdim];
        mmult(out, in, weight_ptr, bias_ptr, indim, outdim);
        indim = outdim;
        memcpy((void*)in, (const void*)out, sizeof(float)*outdim);
        delete [] out;
        kk++;
    }
    else if(layer_name[k].compare(0, 4, "relu")==0){
        relu(in, outdim);
    }
    else if(layer_name[k].compare(0, 4, "prob")==0){
        softmax(in, outdim);    
    }
}
memcpy((void*)fcout, (const void*) in, sizeof(float)*outdim);
delete [] in;
}

void CNN4FPGA::PrepareLayerdef(vector< layer > b, vector< layer> ann, shared_ptr< int > ly_host) {
    for(int i=0; i<14; i++){
        ly_host.get()[i] = 0; 
    }
    layer rr;
    ly_host.get()[12] = ann.size(); 
    ly_host.get()[13] = b.size() + 1; 
    ly_host.get()[14] = b.size(); 
    ly_host.get()[15] = 1; 

    std::cout << "fin\tfout\tfinrow\tfincol\tfrow\tfcol\tKsize\tKstride\tpad\tmask\taddr_in\taddr_wght\taddr_out\tpool\trelu\t0" << std::endl;
    for(int i=0; i<b.size()+ann.size(); i++){
        if(i<b.size())
            rr = b[i];
        else
            rr = ann[i-b.size()];
        ly_host.get()[16+ i*16+ 0]  = rr.fin;
        ly_host.get()[16+ i*16+ 1]  = rr.fout;
        ly_host.get()[16+ i*16+ 2]  = rr.finrow;
        ly_host.get()[16+ i*16+ 3]  = rr.fincol;
        ly_host.get()[16+ i*16+ 4]  = rr.frow;
        ly_host.get()[16+ i*16+ 5]  = rr.fcol;
        ly_host.get()[16+ i*16+ 6]  = rr.Ksize;
        ly_host.get()[16+ i*16+ 7]  = rr.Kstride;
        ly_host.get()[16+ i*16+ 8]  = rr.pad;
        ly_host.get()[16+ i*16+ 9]  = rr.mask;
        ly_host.get()[16+ i*16+ 10] = rr.addr_in;
        ly_host.get()[16+ i*16+ 11] = rr.addr_wght;
        ly_host.get()[16+ i*16+ 12] = rr.addr_out;
        ly_host.get()[16+ i*16+ 13] = rr.pool;
        ly_host.get()[16+ i*16+ 14] = rr.relu;
        ly_host.get()[16+ i*16+ 15] = rr.pool_kernel; 
        std::cout <<  rr.fin << "\t" << rr.fout << "\t" << rr.finrow << "\t" << rr.fincol << "\t" << rr.frow << "\t" << 
        rr.fcol << "\t" << rr.Ksize << "\t" << rr.Kstride << "\t" << rr.pad << "\t" << rr.mask << "\t" << rr.addr_in << "\t" << 
        rr.addr_wght << "\t" << rr.addr_out << "\t" << rr.pool << "\t" << rr.relu << "\t" << 0 << "\t" << std::endl; 
    } 
}

void CNN4FPGA::print_layer(vector< layer > net4fpga) {
//add mask    std::cout << "fin\tfout\tfinrow\tfincol\tfrow\tfcol\tKsize\tKstride\tpad\tmask\taddr_in\taddr_wght\taddr_out\tpool\trelu\t0" << std::endl;
    std::cout << "fin\tfout\tfinrow\tfincol\tfrow\tfcol\tKsize\tKstride\tpad\taddr_wght\taddr_in\taddr_out\tpool\trelu\t0" << std::endl;
    for(int i=0; i<net4fpga.size(); i++){
        std::cout \
        << " type: " << "conv" \
        << " in: " << net4fpga[i].fin \
        << " out: " << net4fpga[i].fout \
        << " in_hei: " << net4fpga[i].finrow \
        << " in_wid: " << net4fpga[i].fincol \
        << " out_hei: " << net4fpga[i].frow \
        << " out_wid: " << net4fpga[i].fcol \
        << " kernel: " << net4fpga[i].Ksize \
        << " stride: " << net4fpga[i].Kstride \
        << " pad: " << net4fpga[i].pad \
        << " addr_wght: " << net4fpga[i].addr_wght \
        << " addr_in: " << net4fpga[i].addr_in \
        << " addr_out: " << net4fpga[i].addr_out \
        << " pool: " << net4fpga[i].pool \
        << " relu: " << net4fpga[i].relu \
        << " pool_kernel: " << net4fpga[i].pool_kernel \
        << std::endl;
    } 
}

void CNN4FPGA::print_lycfg(vector< lycfg > origin){
    for(int i=0; i<origin.size(); i++){
       std::cout \
       << " type: " << origin[i].type \
       << " in: " << origin[i].num_input \
       << " out: " << origin[i].num_output \
       << " in_hei: " << origin[i].input_height \
       << " in_wid: " << origin[i].input_width \
       << " out_hei: " << origin[i].output_height \
       << " out_wid: " << origin[i].output_width \
       << " kernel: " << origin[i].kernel_size \
       << " stride: " << origin[i].kernel_stride \
       << " pad: " << origin[i].kernel_pad \
       << std::endl;
    }       
}

void CNN4FPGA::verify_reorder(shared_ptr< float > wght_dram){

    ofstream curr;
    curr.open("curr.txt");
    for(int c=0; c<wght_length; c++){
        curr << wght_dram.get()[c] << endl;
    }
    curr.close();
   int i=0;
   ifstream gd, ths;
   gd.open("goldedn_wght.txt");
   string gdline;
   //ths.open("curr_before.txt");
   //string thsline;
   if (gd.is_open())
   {
     while ( getline(gd, gdline) )
     {
        //getline(ths, thsline);
        //float tmpt = std::strtof(thsline.c_str(), 0); 
         float tmpd = std::strtof(gdline.c_str(), 0); 
         if((rcmp(tmpd, wght_dram.get()[i])>1e-5)&&(i>=0)){
           std::cout << "wrong numbers in weight?" << std::endl; 
           std::cout << "positions: " << i << " orig: " << tmpd << " new: " << wght_dram.get()[i] << std::endl; 
           break;
         }
         i++;
     }
     gd.close();
     //ths.close();
   }
   else cout << "Unable to open file";
}


class OpenCLFPGAModel {

public:
  OpenCLFPGAModel(){
     initialized = false;
  } 
  
  void setFPGAModel(const char* bin_path, CNN4FPGA cnnModel);
  void FPGAinit();    
  vector< float > FPGAexec(float* image);    
  vector< float > FPGAann(float* fcnin);

  ~OpenCLFPGAModel() {
    if (initialized) {

      //free(wght_dram_clhost);
      //free(ly_clhost);
#if SIM
      free(DRAM_sim);
      free(DRAM_LY_sim);
#else
      clReleaseProgram(program);
      clReleaseKernel(kernel);
      clReleaseCommandQueue(commands);
      clReleaseContext(context);
      clReleaseMemObject(DRAM);
      clReleaseMemObject(DRAM_LY);
#endif
    }
  }

#if SIM

#else
  cl_context& getContext() {
    if (initialized) {
      return context;
    }
    else {
      throw std::runtime_error("environment not setup");
    }
  }

  cl_command_queue& getCmdQueue() {
    if (initialized) {
      return commands;
    }
    else {
      throw std::runtime_error("environment not setup");
    }
  }

  cl_kernel& getKernel() {
    if (initialized) {
      return kernel;
    }
    else {
      throw std::runtime_error("environment not setup");
    }
  }
  cl_mem& getArg1() {
    if (initialized) {
      return DRAM_LY;
    }
    else {
      throw std::runtime_error("environment not setup");
    }
  }
  cl_mem& getArg0() {
    if (initialized) {
      return DRAM;
    }
    else {
      throw std::runtime_error("environment not setup");
    }
  }
#endif

private:

  int load_file(
      const char *filename, 
      char **result)
  { 
    int size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL) 
    { 
      *result = NULL;
      return -1; // -1 means file opening fail 
    } 
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *)malloc(size+1);
    if (size != fread(*result, sizeof(char), size, f)) 
    { 
      free(*result);
      return -2; // -2 means file reading fail 
    } 
    fclose(f);
    (*result)[size] = 0;
    return size;
  }
  template<typename dd_type, typename ds_type> 
    void mycpy(dd_type* dstdram, int dst, ds_type* srcdram, int src, int length);
  template<typename d_type> 
    void reorder_output(d_type *m, int num, int row, int col);
  template<typename d_type>
    void prepare_image(d_type *m, int addr_m); 
  template<typename data_type>
    void reorder_ann_input(float* fcnin, int m, int n, data_type* dram);
  template<typename data_type>
    void reorder_ann_output(float* ann_output, int m, int h, data_type* dram);

  void reorder_image_verify(float* ddram);
private:
  bool initialized;
  int ly_vol;
  int wt_vol;          //CONV parameters
  int fm_vol;
  int infm_vol;
  int lastfm_vol;
  int conv_vol;
  int dram_vol;

  int wght_fcn_vol;    //FCN parameters
  int fm_fcn_vol;
  int in_fcn_vol;
  int out_fcn_vol;

  shared_ptr< float > wght_dram_clhost;
  shared_ptr< float > wght_dram_ann_clhost;
  shared_ptr< int > ly_clhost;
  vector< lycfg > orig;
  vector< layer > cnn4fpga;
#if SIM
  bw_t* DRAM_sim;
  ly_t* DRAM_LY_sim;
#else
  cl_context context;                 // compute context
  cl_command_queue commands;          // compute command queue
  cl_program program;                 // compute program
  cl_kernel kernel;                   // compute kernel
  cl_mem DRAM;//[DATA_VOL];              // device memory used for the input/output data
  cl_mem DRAM_LY;//[LayerDef];              // device memory used for layer definition
#endif
#if DOUBLE
  bool double_flag; 
#endif

};

void OpenCLFPGAModel::setFPGAModel( const char* bin_path, CNN4FPGA cnnModel)
{
      // start platform setting up
      int err;
      //CONV parameter and pointer
      ly_vol = cnnModel.layerdef_len();
      wt_vol = cnnModel.weight_len();
      fm_vol = cnnModel.fm_len();
      infm_vol = cnnModel.infm_len();
      lastfm_vol = cnnModel.lastfm_len();

      wght_dram_clhost = cnnModel.weight_ptr();
      ly_clhost = cnnModel.layerdef_ptr();
      orig = cnnModel.origin;
      cnn4fpga = cnnModel.net4fpga;

      //FCN parameter and pointer
	  wght_fcn_vol = cnnModel.fcn_wght_len();
	  fm_fcn_vol   = cnnModel.fcn_fm_len();
	  in_fcn_vol   = cnnModel.fcn_input_len();
	  out_fcn_vol  = cnnModel.fcn_output_len();

      wght_dram_ann_clhost = cnnModel.weight_ann_ptr();

#if DOUBLE
      dram_vol = wt_vol + 2*fm_vol + wght_fcn_vol + fm_fcn_vol;
      conv_vol = wt_vol + 2*fm_vol;
      double_flag = 0;
#else
      dram_vol = wt_vol + fm_vol + wght_fcn_vol + fm_fcn_vol;
      conv_vol = wt_vol + fm_vol;
#endif

    //printf("start FPGA init\n");
    //if_wght_t *fdram = (if_wght_t*)malloc(sizeof(if_wght_t)* wt_vol);
    //if(fdram == NULL)
    //    printf("OpenCLFPGAModel::FPGAinit fdram malloc failed\n");
    //mycpy<if_wght_t, float>(fdram, 0, wght_dram_clhost.get(), 0, wt_vol);
    //printf("finish mycpy FPGA init\n");

#if SIM //SIM
  printf("The total size of is setFPGAModel:: dram_vol = %d\n", dram_vol);
  DRAM_sim = (bw_t*)malloc(sizeof(if_data_t)*(dram_vol));
  //--------- FCN debug: DRAM space --------------
  //DRAM_sim = (bw_t*)malloc(sizeof(float)*(4096*25088+4096*1024+25088*16+4096*16+4096*16+2*1024*16));
  if(DRAM_sim == NULL)
    printf("OpenCLFPGAModel::FPGAinit DRAM_sim malloc failed\n");

  DRAM_LY_sim = (ly_t*)malloc(sizeof(int)*ly_vol);
  if(DRAM_LY_sim == NULL)
    printf("OpenCLFPGAModel::FPGAinit DRAM_LY_sim malloc failed\n");

  //memcpy((void*) DRAM_sim, (const void*)fdram, sizeof(if_wght_t)* wt_vol);
  //memcpy((void*) DRAM_LY_sim, (const void*)ly_clhost.get(), sizeof(int)*ly_vol);

#else //else SIM
      const char* kernel_name = "vgg16" ;
      cl_platform_id platform_id;
      cl_device_id device_id;

      char cl_platform_vendor[1001];
      char cl_platform_name[1001];

      // Connect to first platform
      err = clGetPlatformIDs(1, &platform_id, NULL);

      if (err != CL_SUCCESS) {
          throw std::runtime_error(
              "Failed to find an OpenCL platform!");
      }

      err = clGetPlatformInfo(
          platform_id, 
          CL_PLATFORM_VENDOR, 
          1000, 
          (void *)cl_platform_vendor,NULL);

      if (err != CL_SUCCESS) {
          throw std::runtime_error(
              "clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!");
      }

      err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
      if (err != CL_SUCCESS) {
          throw std::runtime_error("clGetPlatformInfo(CL_PLATFORM_NAME) failed!");
      }

      // Connect to a compute device
      err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, NULL);
      if (err != CL_SUCCESS) {
          throw std::runtime_error("Failed to create a device group!");
      }

      // Create a compute context 
      context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
      if (!context) {
          throw std::runtime_error("Failed to create a compute context!");
      }

      // Create a command commands
      commands = clCreateCommandQueue(context, device_id, 0, &err);
      if (!commands) {
          throw std::runtime_error("Failed to create a command queue context!");
      }

      // Load binary from disk
      unsigned char *kernelbinary;
      int n_i = load_file(bin_path, (char **) &kernelbinary);

      if (n_i < 0) {
          throw std::runtime_error(
              "failed to load kernel from xclbin");
      }
      size_t n_t = n_i;

      int status = 0;

      // Create the compute program from offline
      program = clCreateProgramWithBinary(context, 1, &device_id, &n_t,
              (const unsigned char **) &kernelbinary, &status, &err);
      if ((!program) || (err!=CL_SUCCESS)) {
          throw std::runtime_error(
              "Failed to create compute program from binary");
      }

      // Build the program executable
      err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
      if (err != CL_SUCCESS) {
          throw std::runtime_error("Failed to build program executable!");
      }

      // Create the compute kernel in the program we wish to run
      kernel = clCreateKernel(program, kernel_name, &err);
      if (!kernel || err != CL_SUCCESS) {
          throw std::runtime_error("Failed to create compute kernel!");
      }

      //-------init DRAM-------------//
      DRAM = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(data_t) * dram_vol, NULL, NULL);
      DRAM_LY = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(int) * ly_vol, NULL, NULL);
      if ((!DRAM) || (!DRAM_LY))
      {
        printf("Error: Failed to allocate device memory!\n");
        printf("Test failed\n");
      }

      err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &DRAM);
      err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &DRAM_LY);
      if (err != CL_SUCCESS)
      {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        printf("Test failed\n");
      }

//   cl_event event;
//   struct timeval t0, t2;
//   err = 0;
//   err = clEnqueueWriteBuffer(commands, DRAM, CL_TRUE, 0, sizeof(if_data_t) * wt_vol, fdram, 0, NULL, &event);
//   err = clEnqueueWriteBuffer(commands, DRAM_LY, CL_TRUE, 0, sizeof(int) * ly_vol, ly_clhost.get(), 0, NULL, &event);
//   if (err != CL_SUCCESS)
//   {
//       printf("Error: Failed to write to source array a!\n");
//       printf("Test failed\n");
//   }
//
//   clWaitForEvents(1, &event);
//   gettimeofday(&t0, NULL);
//   //executing vgg16 kernels
//   printf("write weight& configuration to Device!\n");
//   err = clEnqueueTask(commands, kernel, 0, NULL, &event);
//     if (err != CL_SUCCESS)
//   {
//       printf("Error: Failed run kernel!\n");
//       printf("Test failed\n");
//   }
//   clWaitForEvents(1, &event); 
//   gettimeofday(&t2, NULL);
//   float time_kernel = (t2.tv_sec-t0.tv_sec)*1e+3 + (t2.tv_usec-t0.tv_usec)*1e-03 ;
//   printf("Weight Transfer time :%8.6f ms\n ", time_kernel);

#endif //endif SIM
//     if(fdram != NULL)
//       free(fdram);
      initialized = true;
      printf("successfully initialize FPGA bitstream!\n");
}

void OpenCLFPGAModel::FPGAinit(){

    printf("** start FPGA init\n");
    //CONV weight
    if_wght_t *fdram = (if_wght_t*)malloc(sizeof(if_wght_t)* wt_vol);
    if(fdram == NULL)
        printf("OpenCLFPGAModel::FPGAinit fdram malloc failed\n");
    mycpy<if_wght_t, float>(fdram, 0, wght_dram_clhost.get(), 0, wt_vol);
    //FCN weight
    if_wght_t *fdram_ann = (if_wght_t*)malloc(sizeof(if_wght_t)* wght_fcn_vol);
    if(fdram_ann == NULL)
        printf("OpenCLFPGAModel::FPGAinit fdram_ann malloc failed\n");
    mycpy<if_wght_t, float>(fdram_ann, 0, wght_dram_ann_clhost.get(), 0, wght_fcn_vol);

    printf("finish mycpy FPGA init\n");

#if SIM
    //FCN debug
    //transfer weight
    memcpy((void*) DRAM_sim, (const void*)fdram, sizeof(if_wght_t)* wt_vol);
    //printf("+++++++++++++++penCLFPGAModel::FPGAinit, init FPGA with weights: %d\n", conv_vol);
    //transfer FCN weight
    memcpy((void*) (DRAM_sim+(conv_vol*sizeof(if_wght_t)/sizeof(bw_t))), (const void*)fdram_ann, sizeof(if_wght_t)* wght_fcn_vol);
    //transfer layer definition
    memcpy((void*) DRAM_LY_sim, (const void*)ly_clhost.get(), sizeof(int)*ly_vol);

#else
    cl_event event;
    struct timeval t0, t2;
    int err = 0;

    //transfer weight
    err = clEnqueueWriteBuffer(commands, DRAM, CL_TRUE, 0, sizeof(if_data_t) * wt_vol, fdram, 0, NULL, &event);
    //transfer FCN weight
    int OFFSET = sizeof(if_wght_t) * conv_vol;
    err = clEnqueueWriteBuffer(commands, DRAM, CL_TRUE, OFFSET, sizeof(if_wght_t) * wght_fcn_vol, fdram_ann, 0, NULL, &event);
    //transfer layer definition
    err = clEnqueueWriteBuffer(commands, DRAM_LY, CL_TRUE, 0, sizeof(int) * ly_vol, ly_clhost.get(), 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array a!\n");
        printf("Test failed\n");
    }

    clWaitForEvents(1, &event);
    gettimeofday(&t0, NULL);
    //executing vgg16 kernels
    printf("write weight& configuration to Device!\n");
    err = clEnqueueTask(commands, kernel, 0, NULL, &event);
      if (err != CL_SUCCESS)
    {
        printf("Error: Failed run kernel!\n");
        printf("Test failed\n");
    }
    clWaitForEvents(1, &event); 
    gettimeofday(&t2, NULL);
    float time_kernel = (t2.tv_sec-t0.tv_sec)*1e+3 + (t2.tv_usec-t0.tv_usec)*1e-03 ;
    printf("Weight Transfer time :%8.6f ms\n ", time_kernel);

#endif
    if(fdram != NULL)
        free(fdram);
    if(fdram_ann != NULL)
        free(fdram_ann);
}

//load input feature maps, re-order under CONV-weight schedule
void open_input(char* filename, int m, int n, float* dram){
    printf("load input array dram\n");
    double data;
    FILE* fp = fopen(filename, "r");
    for(int i=0; i<m/n; i++){
        for(int t=0; t<n; t++){
            for(int j=0; j<n; j++){
                fscanf(fp, "%lf\n", &data);
                dram[i*n*n + t + j*n] = (float)data;
            }
        }
    }
    fclose(fp);
}
//load weight feature maps, re-order under INPUT-weight schedule
void open_weight(char* filename, int m, int n, float* dram){
    printf("load weight array dram\n");
    double data;
    FILE* fp = fopen(filename, "r");
    for(int i=0; i<m; i++){
        for(int j=0; j<n/UNROLL; j++){
            for(int k=0; k<UNROLL; k++){
                fscanf(fp, "%lf\n", &data);
                dram[i*UNROLL+ j*UNROLL*m+ k] = (float)data;
            }
        }
    }
    fclose(fp);
}
//load output feature map, re-order 
void open_output(char* filename, int m, int h, float* dram){
    printf("load output array dram\n");
    double data;
    FILE* fp = fopen(filename, "r");
    for(int i=0; i<m; i+=UNROLL){
        for(int k=0; k<UNROLL; k++){
            for(int j=0; j<h; j++){
                fscanf(fp, "%lf\n", &data);
                dram[i*h + k + j*UNROLL] = (float)data;
                //dram[(i + k)*h + j] = (float)data;
            }
        }
    }
    fclose(fp);
}
//load bias, re-order 
void open_bias(char* filename, int m, int h, float* dram){
    printf("load bias array dram\n");
    double data;
    FILE* fp = fopen(filename, "r");
    for(int i=0; i<m; i++){
        for(int j=0; j<h; j++){
            fscanf(fp, "%lf\n", &data);
            dram[i*h + j] = (float)data;
        }
    }
    fclose(fp);
}
//compare output
void out_compare(float* output, float* output_gold, int m){
    printf("compare output\n");

    FILE* fp = fopen("/curr/chenzhang/tool/caffe_fpga/output_compare.txt", "w");
    int cnt = 0;
    float flag = 0.0;
    for(int i=0; i<m; i++){
        float res = abs(output[i]+output_gold[i]);
        if(res!=0.0)
            flag = abs(output[i]-output_gold[i])/res;
            if(flag>5e-3){
                cnt++;
                fprintf(fp, "%f,\t%f,\t%f\n", output[i], output_gold[i], flag);
            }
    }
    if(cnt!=0)
        printf("FCN errors! cnt:%d\n", cnt);
    else
        printf("FCN success!\n");
    fclose(fp);
}

vector< float > OpenCLFPGAModel::FPGAexec(float* image) {

#if TIME
  struct timeval t[10];
  //start computation
  gettimeofday(&t[0], NULL);
#endif
  float *ddram = (float*)malloc(sizeof(float)*infm_vol);
  if(ddram == NULL)
    printf("OpenCLFPGAModel::FPGAexec ddram malloc failed\n");

  memcpy((void*)ddram, (const void*)image, sizeof(float)*infm_vol);
#if TIME
  //copy input image
  gettimeofday(&t[1], NULL);
#endif
  prepare_image<float>(ddram, 0);
  //reorder_image_verify(ddram);
  //printf("finish prepare image\n");
  if_data_t *rdram = (if_data_t*)malloc(sizeof(if_data_t)*infm_vol);
  if(rdram == NULL)
    printf("OpenCLFPGAModel::FPGAexec rdram malloc failed\n");

  mycpy<if_data_t, float>(rdram, 0, ddram, 0, infm_vol);
#if TIME
  //reorder input image & data type transformation
  gettimeofday(&t[2], NULL);
#endif

#if SIM

#if DOUBLE 
  //1. host 2 device data transfer flag = 0
  int input_offset = (double_flag==0)?wt_vol:(wt_vol+fm_vol);
  int DATA_OFFSET = (input_offset)*sizeof(if_wght_t)/sizeof(bw_t);
  memcpy((void*)(DRAM_sim+DATA_OFFSET), (const void*)rdram, sizeof(if_data_t)*infm_vol);
  //execution flag = 1
  int pp = (double_flag==1)?(int)0:(int)fm_vol;
  memcpy((void*)DRAM_LY_sim, (const void*)&pp, sizeof(int));

  //2. device 2 host data transfer flag = 0
  if_data_t* feat_dev = (if_data_t*) malloc(sizeof(if_data_t)*(lastfm_vol)); 
  if(feat_dev == NULL)
    printf("openclfpgamodel::fpgaexec feat_dev(sim) malloc failed\n");

  float* fpga_feat = (float*) malloc(sizeof(float)*(lastfm_vol)); 
  if(fpga_feat == NULL)
    printf("openclfpgamodel::fpgaexec fpga_feat(sim) malloc failed\n");
  
  int output_offset = (double_flag==0)?(wt_vol+fm_vol-lastfm_vol):(wt_vol+2*fm_vol-lastfm_vol);
  DATA_OFFSET = sizeof(if_data_t)*(output_offset)/sizeof(bw_t);
  memcpy((void*)feat_dev, (const void*)(DRAM_sim+DATA_OFFSET), sizeof(if_data_t)*lastfm_vol);
  const int num = orig[orig.size()-1].num_output;
  const int row = orig[orig.size()-1].output_height;
  const int col = orig[orig.size()-1].output_width;
  mycpy<float, if_data_t>(fpga_feat, 0, feat_dev, 0, num*row*col);
  reorder_output<float>(fpga_feat, num, row, col);

  vector< float > result;

  for(int i=0; i<lastfm_vol; i++){
    result.push_back(fpga_feat[i]); 
  }

  //3. CNN simulation kernel here
  printf("start CNN computation\n");
  vgg16(DRAM_sim, DRAM_LY_sim);
  printf("finish CNN computation\n");

  double_flag = 1 - double_flag;

  free(feat_dev);
  free(fpga_feat);
  free(ddram);
  free(rdram);
  return result;

#else       //else DOUBLE

    //int DATA_OFFSET = (cnn4fpga[0].addr_in)*sizeof(if_wght_t)/sizeof(bw_t);
    int DATA_OFFSET = (wt_vol)*sizeof(if_wght_t)/sizeof(bw_t);
    memcpy((void*)(DRAM_sim+DATA_OFFSET), (const void*)rdram, sizeof(if_data_t)*infm_vol);
    
    printf("start CNN computation\n");
    //CNN simulation kernel here
    vgg16(DRAM_sim, DRAM_LY_sim);
    
    printf("finish CNN computation\n");
    
    if_data_t* feat_dev = (if_data_t*) malloc(sizeof(if_data_t)*(lastfm_vol)); 
    if(feat_dev == NULL)
      printf("openclfpgamodel::fpgaexec feat_dev(sim) malloc failed\n");
    
    float* fpga_feat = (float*) malloc(sizeof(float)*(lastfm_vol)); 
    if(fpga_feat == NULL)
      printf("openclfpgamodel::fpgaexec fpga_feat(sim) malloc failed\n");
    
    //std::cout<< "last feature map vollum: "<< lastfm_vol <<std::endl;
    
    DATA_OFFSET = sizeof(if_data_t)*(cnn4fpga[cnn4fpga.size()-1].addr_out)/sizeof(bw_t);
    memcpy((void*)feat_dev, (const void*)(DRAM_sim+DATA_OFFSET), sizeof(if_data_t)*lastfm_vol);
    mycpy<float, if_data_t>(fpga_feat, 0, feat_dev, 0, lastfm_vol);
    const int num = orig[orig.size()-1].num_output;
    const int row = orig[orig.size()-1].output_height;
    const int col = orig[orig.size()-1].output_width;
    reorder_output<float>(fpga_feat, num, row, col);
    
    vector< float > result;
    
    for(int i=0; i<lastfm_vol; i++){
      result.push_back(fpga_feat[i]); 
    }
    free(feat_dev);
    free(fpga_feat);
    free(ddram);
    free(rdram);
    
    return result; 
/*
    //++++++++Start FCN debug testing+++++++//
    int OFFSET;
    printf("++++++++++++ innerproduct simulation ++++++++++++\n");   
//transformed (weight-major) parameter
//in,    out, inrow, incol, row, col, Ksize, stride, pad, mask, addr_i(weight), addr_w(input),        addr_o,          pool, relu, pool_kernel  
//25088, 16,  64,    64,    64,  64,  1,     1,      0,   1,    0,              102760448(4096*25088),103227392(+all), 0,    0,    0
    int addr_input_1  = (0);
    int addr_input_2  = (4096*25088);
    int addr_weight_1 = (4096*25088+4096*1024);
    int addr_bias_1   = (4096*25088+4096*1024+25088*16);
    int addr_output_1 = (4096*25088+4096*1024+25088*16+4096*16);
    int addr_weight_2 = (4096*25088+4096*1024+25088*16+4096*16);
    int addr_bias_2   = (4096*25088+4096*1024+25088*16+4096*16+4096*16);
    int addr_output_2 = (4096*25088+4096*1024+25088*16+4096*16+4096*16+1024*16);

    int fc_def[48] = {
    0,     0,    0,  0,  0,  0, 0, 0, 0, 0,          0,           0,           0, 0, 2, 2,
    25088, 16,  64, 64, 64, 64, 1, 1, 0, 1, addr_input_1, addr_weight_1, addr_output_1, 0, 0, addr_bias_1,
     4096, 16,  32, 32, 32, 32, 1, 1, 0, 1, addr_input_2, addr_weight_2, addr_output_2, 0, 0, addr_bias_2};
    memcpy((void*)(DRAM_LY_sim), (const void*)fc_def, sizeof(float)*48);
    memset((void*)(DRAM_sim), 0, sizeof(float)*(4096*25088+4096*1024+25088*16+4096*16+4096*16+2*1024*16));

    float*  input = (float*)malloc(25088*16*sizeof(float));
    float* weight = (float*)malloc(4096*25088*sizeof(float));
    float*   bias = (float*)malloc(4096*16*sizeof(float));
    float* output_gold = (float*)malloc(4096*16*sizeof(float));

    //1. --- used as CONV-input 
    open_weight("/curr/chenzhang/tool/caffe_fpga/MMX/weight1.txt", 4096, 25088, weight);
    OFFSET = addr_input_1*sizeof(float)/sizeof(bw_t);
    memcpy((void*)(DRAM_sim+OFFSET), (const void*)weight, sizeof(float)*4096*25088);
    //1. --- used as CONV-weight 
    open_input("/curr/chenzhang/tool/caffe_fpga/MMX/input.txt", 25088, 16, input);
    OFFSET = addr_weight_1*sizeof(float)/sizeof(bw_t);
    memcpy((void*)(DRAM_sim+OFFSET), (const void*)input, sizeof(float)*25088*16);
    //1. --- used as CONV-bias 
    open_bias("/curr/chenzhang/tool/caffe_fpga/MMX/bias1.txt", 4096, 16, bias);
    OFFSET = addr_bias_1*sizeof(float)/sizeof(bw_t);
    memcpy((void*)(DRAM_sim+OFFSET), (const void*)bias, sizeof(float)*4096*16);
//
//  //execute first layer
//  vgg16(DRAM_sim, DRAM_LY_sim);
//  //re-order first layer output
//  float* output2 = (float*)malloc(4096*16*sizeof(float));
//  float* otmp   = (float*)malloc(16*16*sizeof(float));
//  OFFSET = (addr_output_1)*sizeof(float)/sizeof(bw_t);
//  memcpy((void*)output2, (const void*)(DRAM_sim+OFFSET), sizeof(float)*4096*16);
//  for(int ii=0; ii<4096; ii+=16){
//      memcpy((void*)otmp, (const void*)(output2+ii*16), sizeof(float)*16*16);
//      for(int jj=0; jj<16; jj++){
//          for(int kk=0; kk<16; kk++){
//              output2[ii*16 + jj*16 + kk] = otmp[kk*16 + jj]; 
//          }
//      } 
//  }
//  memcpy((void*)(DRAM_sim+OFFSET), (const void*)output2, sizeof(float)*4096*16);
//  free(otmp);
//  free(output2);
//  int fc_def_2[48] = {
//  0,     0,    0,  0,  0,  0, 0, 0, 0, 0,          0,           0,           0, 0, 1, 1,
//  //25088, 16,  64, 64, 64, 64, 1, 1, 0, 1, addr_input_1, addr_weight_1, addr_output_1, 0, 0, 0,
//   4096, 16,  32, 32, 32, 32, 1, 1, 0, 0, addr_input_2, addr_weight_2, addr_output_2, 0, 0, 0};
//  memcpy((void*)(DRAM_LY_sim), (const void*)fc_def_2, sizeof(float)*32);
//
//  //2. --- used as CONV-weight 
//  open_input("/curr/chenzhang/tool/caffe_fpga/MMX/output1.txt", 4096, 16, input);
//  OFFSET = addr_weight_2*sizeof(float)/sizeof(bw_t);
//  memcpy((void*)(DRAM_sim+OFFSET), (const void*)input, sizeof(float)*4096*16);
//
    //2. --- used as CONV-input 
    open_weight("/curr/chenzhang/tool/caffe_fpga/MMX/weight2.txt", 1024, 4096, weight);
    OFFSET = addr_input_2*sizeof(float)/sizeof(bw_t);
    memcpy((void*)(DRAM_sim+OFFSET), (const void*)weight, sizeof(float)*1024*4096);
    //2. --- used as CONV-bias 
    open_bias("/curr/chenzhang/tool/caffe_fpga/MMX/bias2.txt", 1024, 16, bias);
    OFFSET = addr_bias_2*sizeof(float)/sizeof(bw_t);
    memcpy((void*)(DRAM_sim+OFFSET), (const void*)bias, sizeof(float)*1024*16);

    free(bias);
    free(input);
    free(weight);
    
    //vgg16(DRAM_sim, DRAM_LY_sim);

    float* output = (float*)malloc(4096*16*sizeof(float));
    OFFSET = (addr_output_2)*sizeof(float)/sizeof(bw_t);
    memcpy((void*)output, (const void*)(DRAM_sim+OFFSET), sizeof(float)*1024*16);

    open_output("/curr/chenzhang/tool/caffe_fpga/MMX/output2.txt", 1024, 16, output_gold);
    //out_compare(output, output_gold, 1024*16);

    vector< float > result;
    for(int i=0; i<1024; i++){
      result.push_back(output[i]); 
    }
    free(output);
    free(output_gold);
    return result;
    //++++++++ End FCN debug testing +++++++//
*/
#endif  //end DOUBLE

#else //else: no_SIM

#if DOUBLE //non_SIM branck DOUBLE
//TODO: write ping-pong hardware deriver
  int err=0;
#if TIME
  gettimeofday(&t[3], NULL);
#endif

  cl_event event;
  //3. start kernel execution
  //---3. write layer definition
  int pp = (double_flag==1)?(int)0:(int)fm_vol;
  err = clEnqueueWriteBuffer(commands, DRAM_LY, CL_TRUE, 0, sizeof(int) * 1, &pp, 0, NULL, &event);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write new layer definition array to device! %d\n", err);
    printf("Test failed\n");
  }
  printf("start kernel!\n");
  clWaitForEvents(1, &event);

  //1. start write buffer
  int input_offset = (double_flag==0)?wt_vol:(wt_vol+fm_vol);
  int DATA_OFFSET = input_offset*sizeof(if_wght_t);
  err = clEnqueueWriteBuffer(commands, DRAM, CL_FALSE, DATA_OFFSET, sizeof(data_t)*infm_vol, rdram, 0, NULL, &event);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to source array a!\n");
      printf("Test failed\n");
  }

#if TIME
  gettimeofday(&t[4], NULL);
#endif
  //2. reads buffer out
  float* fpga_feat = (float*) malloc(sizeof(float)*(lastfm_vol)); 
  if(fpga_feat == NULL)
    printf("openclfpgamodel::fpgaexec fpga_feat(exec) malloc failed\n");

  if_data_t* feat_dev = (if_data_t*) malloc(sizeof(if_data_t)*(lastfm_vol)); 
  if(feat_dev == NULL)
    printf("openclfpgamodel::fpgaexec feat_dev(exec) malloc failed\n");

  int output_offset = (double_flag==0)?(wt_vol+fm_vol-lastfm_vol):(wt_vol+2*fm_vol-lastfm_vol);
  DATA_OFFSET = sizeof(if_data_t)*(output_offset);
  err = clEnqueueReadBuffer( commands, DRAM, CL_FALSE, DATA_OFFSET, sizeof(if_data_t)*lastfm_vol, feat_dev, 0, NULL, &event );
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
  }
  //clWaitForEvents(1, &event);

#if TIME
  gettimeofday(&t[5], NULL);
#endif
  err = clEnqueueTask(commands, kernel, 0, NULL, &event);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed run kernel!\n");
      printf("Test failed\n");
  }
  clWaitForEvents(1, &event);
  //4. flip pingpong flag
  double_flag = 1 - double_flag;
#if TIME
  gettimeofday(&t[6], NULL);
#endif
  mycpy<float, if_data_t>(fpga_feat, 0, feat_dev, 0, lastfm_vol);
  const int num = orig[orig.size()-1].num_output;
  const int row = orig[orig.size()-1].output_height;
  const int col = orig[orig.size()-1].output_width;
  reorder_output<float>(fpga_feat, num, row, col);
#if TIME
  gettimeofday(&t[7], NULL);
  printf("1. Copy input image time : %8.6f ms\n ", get_time(t[0], t[1]));
  printf("2. reorder inpt img time : %8.6f ms\n ", get_time(t[1], t[2]));
  printf("3. Write inp Buffer time : %8.6f ms\n ", get_time(t[3], t[4]));
  printf("4. Exec Enqueue Task time: %8.6f ms\n ", get_time(t[5], t[6]));
  printf("5. Read outp Buffer time : %8.6f ms\n ", get_time(t[4], t[5]));
  printf("6. Reorder output time   : %8.6f ms\n ", get_time(t[6], t[7]));
#endif

  vector< float > result;

  for(int i=0; i<lastfm_vol; i++){
    result.push_back(fpga_feat[i]); 
  }
  free(feat_dev);
  free(fpga_feat);

  return result;

#else // non_SIM DOUBLE else (non_DOUBLE)

  int err=0;
#if TIME
  //start kernel
  gettimeofday(&t[3], NULL);
#endif
  cl_event event;
  int DATA_OFFSET = cnn4fpga[0].addr_in*sizeof(if_wght_t);
  err = clEnqueueWriteBuffer(commands, DRAM, CL_TRUE, DATA_OFFSET, sizeof(data_t)*infm_vol, rdram, 0, NULL, &event);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to source array a!\n");
      printf("Test failed\n");
  }

  clWaitForEvents(1, &event);
#if TIME
  //executing vgg16 kernels
  gettimeofday(&t[4], NULL);
#endif
  printf("start kernel!\n");
  err = clEnqueueTask(commands, kernel, 0, NULL, &event);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed run kernel!\n");
      printf("Test failed\n");
  }
  clWaitForEvents(1, &event);
  printf("Kernel has returned !\n");
#if TIME
  gettimeofday(&t[5], NULL);
#endif
  //reads buffer out
  float* fpga_feat = (float*) malloc(sizeof(float)*(lastfm_vol)); 
  if(fpga_feat == NULL)
    printf("openclfpgamodel::fpgaexec fpga_feat(exec) malloc failed\n");

  if_data_t* feat_dev = (if_data_t*) malloc(sizeof(if_data_t)*(lastfm_vol)); 
  if(feat_dev == NULL)
    printf("openclfpgamodel::fpgaexec feat_dev(exec) malloc failed\n");

  DATA_OFFSET = sizeof(if_data_t)*(cnn4fpga[cnn4fpga.size()-1].addr_out);
  err = clEnqueueReadBuffer( commands, DRAM, CL_TRUE, DATA_OFFSET, sizeof(if_data_t)*lastfm_vol, feat_dev, 0, NULL, &event );
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
  }
  clWaitForEvents(1, &event);
  printf("Results has been read out!\n");
#if TIME
  //t4: reads buffer out
  gettimeofday(&t[6], NULL);
#endif
  mycpy<float, if_data_t>(fpga_feat, 0, feat_dev, 0, lastfm_vol);
  const int num = orig[orig.size()-1].num_output;
  const int row = orig[orig.size()-1].output_height;
  const int col = orig[orig.size()-1].output_width;
  reorder_output<float>(fpga_feat, num, row, col);
#if TIME
  //t5: reads buffer out
  gettimeofday(&t[7], NULL);
#if TIME_VERBOSE
  printf("1. Copy input image time : %8.6f ms\n ", get_time(t[0], t[1]));
  printf("2. reorder inpt img time : %8.6f ms\n ", get_time(t[1], t[2]));
  printf("3. Write inp Buffer time : %8.6f ms\n ", get_time(t[3], t[4]));
  printf("4. Exec Enqueue Task time: %8.6f ms\n ", get_time(t[4], t[5]));
  printf("5. Read outp Buffer time : %8.6f ms\n ", get_time(t[5], t[6]));
  printf("6. Reorder output time   : %8.6f ms\n ", get_time(t[6], t[7]));
#else
  printf("A. Exec Kernel Time: %8.6f ms\n ", get_time(t[4], t[5]));
  printf("B. Exec Overal Time: %8.6f ms\n ", get_time(t[1], t[2])+\
                                             get_time(t[3], t[4])+\
                                             get_time(t[4], t[5])+\
                                             get_time(t[5], t[6]) +\
                                             get_time(t[6], t[7]));

#endif
#endif

  vector< float > result;

  for(int i=0; i<lastfm_vol; i++){
    result.push_back(fpga_feat[i]); 
  }
  free(feat_dev);
  free(fpga_feat);

  return result;
#endif //endif non_SIM DOUBLE

#endif //endif SIM
// if_data_t* layer1_fm = (if_data_t*)malloc(sizeof(float)*FM11);
} //end of FPGAexec()

template<typename data_type>
void OpenCLFPGAModel::reorder_ann_input(float* fcnin, int m, int n, data_type* dram){
//assuming n = UNROLL = FCN batch size
//assuming m = input size
    int s = 0;
    for(int i=0; i<m; i++){
        for(int j=0; j<n/m; j++){
            for(int t=0; t<m; t++){
                dram[i*m + j*m*m + t] = (data_type)fcnin[s];
                s++;
            }
        }
    }
}
template<typename data_type>
void OpenCLFPGAModel::reorder_ann_output(float* ann_output, int m, int h, data_type* dram){
    int s = 0;
    for(int j=0; j<h/UNROLL; j++){
        for(int i=0; i<m; i++){
            for(int k=0; k<UNROLL; k++){
                //dram[i*h + k + j*16] = (float)data;
                ann_output[j*UNROLL + i*h + k] = (float)dram[s] ;
                s++;
            }
        }
    }
}
//load input feature maps, re-order under CONV-weight schedule
void ann_input(char* filename, int m, int n, float* dram){
    double data;
    FILE* fp = fopen(filename, "r");
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            fscanf(fp, "%lf\n", &data);
            dram[i*n + j] = (float)data;
        }
    }
    fclose(fp);
}
//compare output
void ann_compare(char* filename, float* output, int m){

    FILE* fp = fopen(filename, "r");
    FILE* cp = fopen("out_compare.txt", "w");
    int cnt = 0;
    float flag = 0.0;
    float data = 0.0;
    for(int i=0; i<m; i++){
        fscanf(fp, "%f\n", &data);
        float res = abs(output[i] + data);
        if(res!=0.0){
            flag = abs(output[i] - data)/res;
            if(flag>5e-3){
               cnt++;
               fprintf(cp, "%f, %f, %f\n", output[i], data, flag);
            }
        }
    }
    if(cnt!=0)
        printf("OpenCLFPGAModel::FCN errors! cnt:%d\n", cnt);
    else
        printf("OpenCLFPGAModel::FCN success!\n");
    fclose(cp);
    fclose(fp);
}

vector< float > OpenCLFPGAModel::FPGAann(float* fcnin) {
#if SIM
    //float* fcnin = new float[25088*16];
    //ann_input("/curr/chenzhang/tool/caffe_fpga/MMX/input.txt", 16, 25088, fcnin);
    //prepare input
    if_data_t* dram_ann_input = (if_data_t*)malloc(sizeof(if_data_t)*in_fcn_vol);
    reorder_ann_input<if_data_t>(fcnin, UNROLL, in_fcn_vol/UNROLL, dram_ann_input);

    int OFFSET = (conv_vol+wght_fcn_vol)*sizeof(if_data_t)/sizeof(bw_t);
    memcpy((void*)(DRAM_sim+OFFSET), (const void*)dram_ann_input, sizeof(if_data_t)*in_fcn_vol);
    free(dram_ann_input);

    //prepare layer definition
    int fcn_flag = 1;
    memcpy((void*)DRAM_LY_sim, (const void*)(&fcn_flag), sizeof(int));

    //start kernel
    vgg16(DRAM_sim, DRAM_LY_sim);

    //fetch output 
    if_data_t* dram_ann_output = (if_data_t*)malloc(sizeof(if_data_t)*out_fcn_vol);
    float*     ann_output = new float[out_fcn_vol];
    OFFSET = (dram_vol-out_fcn_vol)*sizeof(if_data_t)/sizeof(bw_t);
    memcpy((void*)dram_ann_output, (const void*)(DRAM_sim+OFFSET), sizeof(if_data_t)*out_fcn_vol);
    reorder_ann_output<if_data_t>(ann_output, UNROLL, out_fcn_vol/UNROLL, dram_ann_output);

    //ann_compare("/curr/chenzhang/tool/caffe_fpga/MMX/output3.txt", ann_output, 1024*16);
    //delete [] fcnin;

    vector< float > result;
    for(int i=0; i<out_fcn_vol; i++){
        result.push_back(ann_output[i]); 
    } 
    delete [] ann_output;
    free(dram_ann_output);
    return result;
#else
    printf("\n******\n executing FPGAann and OpenCL \n");
    cl_event event;
    if_data_t* dram_ann_input = (if_data_t*)malloc(sizeof(if_data_t)*in_fcn_vol);
    reorder_ann_input<if_data_t>(fcnin, UNROLL, in_fcn_vol/UNROLL, dram_ann_input);

    int OFFSET = (conv_vol+wght_fcn_vol)*sizeof(if_data_t);
    int err = clEnqueueWriteBuffer(commands, DRAM, CL_TRUE, OFFSET, sizeof(data_t)*in_fcn_vol, dram_ann_input, 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array a!\n");
        printf("Test failed\n");
    }
    clWaitForEvents(1, &event);
    free(dram_ann_input);

    //prepare layer definition
    int fcn_flag = 1;
    err = clEnqueueWriteBuffer(commands, DRAM_LY, CL_TRUE, 0, sizeof(int) * 1, &fcn_flag, 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write new layer definition array to device! %d\n", err);
        printf("Test failed\n");
    }
    clWaitForEvents(1, &event);

    //start kernel
    err = clEnqueueTask(commands, kernel, 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed run kernel!\n");
        printf("Test failed\n");
    }
    clWaitForEvents(1, &event);

    //fetch output 
    if_data_t* dram_ann_output = (if_data_t*)malloc(sizeof(if_data_t)*out_fcn_vol);
    float*     ann_output = new float[out_fcn_vol];
    OFFSET = (dram_vol-out_fcn_vol)*sizeof(if_data_t);
    err = clEnqueueReadBuffer( commands, DRAM, CL_TRUE, OFFSET, sizeof(if_data_t)*out_fcn_vol, dram_ann_output, 0, NULL, &event );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        printf("Test failed\n");
    }
    reorder_ann_output<if_data_t>(ann_output, UNROLL, out_fcn_vol/UNROLL, dram_ann_output);

    vector< float > result;
    for(int i=0; i<out_fcn_vol; i++){
        result.push_back(ann_output[i]); 
    } 
    delete [] ann_output;
    free(dram_ann_output);
    return result;

#endif
}

template<typename d_type>
void OpenCLFPGAModel::prepare_image(d_type *m, int addr_m) {
    int imgrow= orig[0].input_height - 2*orig[0].kernel_pad;
    int imgcol= orig[0].input_width  - 2*orig[0].kernel_pad;
    d_type* adata = (d_type*)malloc(UNROLL*imgrow*imgcol*sizeof(d_type));
    if(adata == NULL)
        printf("ERROR: OpenCLFPGAModel::prepare_image malloc\n");
	for(int i=0; i<UNROLL; i++) {
	    for(int j=0; j<imgrow; j++) {
	        for(int k=0; k<imgcol; k++){
                if(i<3)
	                adata[i*imgrow*imgcol+ j*imgcol + k] = m[i*imgrow*imgcol + j*imgcol + k + addr_m];
                else
	                adata[i*imgrow*imgcol+ j*imgcol + k] = (d_type)0;
	        }
	    }
	}
	for(int j=0; j<imgrow; j++) {
	    for(int k=0; k<imgcol; k++){
	        for(int i=0; i<UNROLL; i++) {
	            m[(j*imgcol+k)*UNROLL + i + addr_m] = adata[i*imgrow*imgcol + j*imgcol + k];
	        }
	    }
	}
    if(adata != NULL)
        free(adata);
}


template<typename dd_type, typename ds_type>
void OpenCLFPGAModel::mycpy(dd_type* dstdram, int dst, ds_type* srcdram, int src, int length){
    int nancnt = 0;
    for(int i=0; i<length; i++) {
        if(isnan(srcdram[i+ src])) {
    //        dstdram[i+ dst] = (dd_type)4096;
            nancnt++;
        }
        else
            dstdram[i+ dst] = (dd_type)(srcdram[i+ src]);
    }
    //printf("nancnt: %d, \n", nancnt);
}

template<typename d_type>
void OpenCLFPGAModel::reorder_output(d_type *m, int num, int row, int col) {
   //static d_type data[num][row][col];
   d_type* data = (d_type*)malloc(sizeof(d_type)*num*row*col);
   if(data == NULL)
    printf("OpenCLFPGAModel::reorder_output data malloc failed\n");
   for(int ii=0; ii<num; ii+=UNROLL) {
       for(int r=0; r<row; r++) {
           for(int c=0; c<col; c++) {
               for(int i=0; i<UNROLL; i++) {
                   data[(ii+i)*row*col + r*col + c] = m[(r*col+c)*UNROLL + (ii)*row*col + i];
               }
           }
       }
   }
   for(int i=0; i<num; i++) {
       for(int j=0; j<row; j++) {
           for(int k=0; k<col; k++) {
               m[i*row*col + j*col + k] = data[i*col*row + j*col + k];
           }
       }
   }
   if(data != NULL)
    free(data);
}

void OpenCLFPGAModel::reorder_image_verify(float* ddram){

  ofstream curr;
  curr.open("input_image.txt", ios::app);
  for(int i=0; i<infm_vol; i++){
    curr << ddram[i] << std::endl; 
  } 
  curr.close();

  ifstream icur;
  icur.open("input_image_gold.txt");
  string line;
  int j=0;
   if (icur.is_open())
   {
     while ( getline(icur, line) )
     {
         float tmpd = std::strtof(line.c_str(), 0); 
         if((rcmp(tmpd, ddram[j])>1e-5)){
           std::cout << "wrong numbers in input images?" << std::endl; 
           std::cout << "positions: " << j << " orig: " << tmpd << " new: " << ddram[j] << std::endl; 
           break;
         }
         j++;
     }
   }
   else cout << "Unable to open file";
}
