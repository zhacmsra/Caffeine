#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <queue>
#include <pthread.h>

#include <CL/opencl.h>
#include <sys/time.h>
//#include "/curr/chenzhang/tool/caffe/FPGA/include/kernel-cl-batch.hpp"
//#include "/curr/chenzhang/tool/caffe_fpga/FPGA/include/OpenCLEnv.h"

#include <sstream>

#include "./hls/cnn_cfg.hpp"
//if SIM
//#include "./hls/vgg16.hpp"
#include "../../FPGA/include/falconMLlib.h"

#ifdef USE_OPENCV
// NOLINT(build/namespaces)
using namespace caffe;  
using std::string;
using std::queue;
using std::vector;

#define USE_FPGA 1
#define USE_PTHD 1
#define FPGA_Verify 0
#define IMSHOW 0


/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier{
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file,
             const string& bitstream
             );

  //std::vector<Prediction> Classify(const cv::Mat& img, int N = 5, bw_t* DRAM = 0, ly_t* DRAM_LY = 0 );
  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

#if USE_FPGA 
 public:
  CNN4FPGA cnn_model;
  OpenCLFPGAModel fpga;
#endif
#if USE_PTHD
 public:
  pthread_t th1, th2;
  int param_cnn_img, param_cnn_feat, param_ann_feat, param_ann_result;
  bool param_flag;
  float* img1, *img2, *feat1, *feat2, *fpga_result1, *fpga_result2;

  int cnn_img(){
    return param_cnn_img;
  }
  int cnn_feat(){
    return param_cnn_feat;
  }
  int ann_feat(){
    return param_ann_feat;
  }
  int ann_result(){
    return param_ann_result;
  }
  float* img_ptr(int a){
    if(a==1) return img1; 
    else if(a==2) return img2;
    else return NULL;
  }
  float* feat_ptr(int a){
    if(a==1) return feat1; 
    else if(a==2) return feat2;
    else return NULL;
  }
  float* fpga_result_ptr(int a){
    if(a==1) return fpga_result1; 
    else if(a==2) return fpga_result2;
    else return NULL;
  }
  ~Classifier(){
    //delete [] img1; 
    //delete [] img2;
    delete [] feat1;
    delete [] feat2;
    delete [] fpga_result1;
    delete [] fpga_result2;
  }
#endif

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);
#if USE_FPGA
  std::vector<float> FPGA_Predict(const cv::Mat& img);
#endif

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
 public:
  shared_ptr<Net<float> > net_;
};

#if USE_PTHD //two pthreads
static void* cnnThd(void* arg){
  Classifier *p = (Classifier *)arg;

  float* img = p->img_ptr(p->cnn_img());
  float* feat = p->feat_ptr(p->cnn_feat());
  vector<float> result = p->fpga.FPGAexec(img);
  for(int i=0; i<result.size(); i++){
      feat[i] = result[i];
  }
//  std::cout << "Cnn read: img " << p->cnn_img() << " Cnn write: feat " << p->cnn_feat() << std::endl;
}

static void* annThd(void* arg){
  Classifier *p = (Classifier *)arg;

  float* feat = p->feat_ptr(p->ann_feat());
  float* fpga_result = p->fpga_result_ptr(p->ann_result());
  p->cnn_model.ann(p->net_, feat, fpga_result);
//  std::cout << "Ann read: feat " << p->ann_feat() << " Ann write: result " << p->ann_result() << std::endl;
}
#endif

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file,
                       const string& bitstream
                       ) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

#if USE_FPGA
  cnn_model.setCNNModel(net_);
  fpga.setFPGAModel(bitstream.c_str(), cnn_model);
  fpga.FPGAinit();
  //printf("finish FPGA initilization\n");
#endif
#if USE_PTHD
  param_flag = 0;
  //img1 = new float[cnn_model.infm_len()];
  //img2 = new float[cnn_model.infm_len()];
  feat1 = new float[cnn_model.lastfm_len()];
  feat2 = new float[cnn_model.lastfm_len()];
  fpga_result1 = new float[1000];
  fpga_result2 = new float[1000];
#endif

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N){
#if USE_FPGA
  std::vector<float> output = FPGA_Predict(img);
#else
  std::vector<float> output = Predict(img);
#endif

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  timeval startSw, endSw;
  gettimeofday(&startSw, NULL);
  /*library implementation*/
  net_->ForwardPrefilled(); 
  /*library implementation*/
  gettimeofday(&endSw, NULL);
  printf("library time :%8.6f ms\n ", (endSw.tv_sec-startSw.tv_sec)*1e+3 + (endSw.tv_usec-startSw.tv_usec)*1e-03 );

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}
#if USE_FPGA
std::vector<float> Classifier::FPGA_Predict(const cv::Mat& img){
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);

  // Forward dimension change to all layers.
  net_->Reshape();
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  Preprocess(img, &input_channels);

  //FPGA Acceleration Test
  printf("FPGA Prediction Start\n");

#if USE_PTHD

    const float* fdata = net_->input_blobs()[0]->cpu_data();
    img1 = new float[cnn_model.infm_len()];
    for(int i=0; i<cnn_model.infm_len(); i++) {
      img1[i] = (float)fdata[i];
    }

    param_cnn_img = 1; 
    param_ann_result = 1;
    if(param_flag == 0){
        param_cnn_feat = 1; 
        param_ann_feat = 2;
        param_flag = 1;
    }
    else{
        param_cnn_feat = 2; 
        param_ann_feat = 1;   
        param_flag = 0;
    }
    int ret_thrd1 = pthread_create(&th1, NULL, cnnThd, (void*)this);
    int ret_thrd2 = pthread_create(&th2, NULL, annThd, (void*)this);
    if (ret_thrd1 != 0){ 
            printf("falied to create thd1\n"); }
    if (ret_thrd2 != 0) {
            printf("falied to create thd2\n"); }

    void* retval;
    int tmp1 = pthread_join(th1, &retval);
    if (tmp1 != 0) {
            printf("cannot join with Cnn thread\n"); }
    int tmp2 = pthread_join(th2, &retval);
    if (tmp2 != 0) {
            printf("cannot join with Ann thread\n"); }

    
    Blob<float>* output_layer = net_->output_blobs()[0];
    
    std::vector<float> output;
    for(int k =0; k<output_layer->channels(); k++) {
      output.push_back(fpga_result1[k]); 
    }
    delete [] img1;
    return output;
#else
  float fpga_result[1000];

  float *ddram = (float*)malloc(sizeof(float)*INFM);
  const float* fdata = net_->input_blobs()[0]->cpu_data();
  for(int i=0; i<INFM; i++) {
    ddram[i] = (float)fdata[i];
  }

  vector< float > result = fpga.FPGAexec(ddram);
  float* fpga_feat = (float*)malloc(sizeof(float)*result.size()); 
  for(int i=0; i<result.size(); i++){
    fpga_feat[i] = result[i]; 
  }
  cnn_model.ann(net_, fpga_feat, fpga_result);
  //ann(fpga_feat, fpga_result);

  Blob<float>* output_layer = net_->output_blobs()[0];

  std::vector<float> output;
  for(int k =0; k<output_layer->channels(); k++) {
    output.push_back(fpga_result[k]); 
  }

  free(ddram);
  //free(dddram);
  //free(feat_dev);
  free(fpga_feat);
  printf("FPGA Prediction finish\n");
  return output;
#endif
}
#endif

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

std::string ConvertPrecision (float number) {
    std::ostringstream buff;
    buff << std::setw(6) << std::setfill(' ') << std::fixed << std::setprecision(2) << (number * 100);
    return buff.str();
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg"
              << " bitstream.xclbin" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  string bitstream   = argv[6];
  Classifier classifier(model_file, trained_file, mean_file, label_file, bitstream);
  std::cout << "Initializing FPGA with " << bitstream << std::endl;

  std::ifstream out;
  string str = argv[5];
  out.open(str.c_str(), ios::in);

  string file ;
  string last_file;
  char c;
  timeval start_t, end_t;
  timeval synt;

  queue< string > fifo;
  int num = 0;
  while(std::getline(out, file)){
    num+=1; 
  }
  out.close();
  std::cout << "the number of images: "<< num << std::endl;
  out.open(str.c_str(), ios::in);

  int phase = 0;
#if USE_PTHD
    phase += 1;
#endif
#if DOUBLE
    phase += 2;
#endif

  string strID;
  size_t pos;
  CvFont font;
  cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 0.8f, 0.8f, 1.0f, 2, CV_AA);
 
  for(int i=0; i<num+phase; i++){
    
    last_file = file;

    if(i<num) {
        std::getline(out, file);
    };
    
    cv::Mat img = cv::imread(file, -1);

    CHECK(!img.empty()) << "Unable to decode image " << file;
    fifo.push(file);

    //TODO: show image that has been read
    IplImage *pImgC;
    if (i>=phase) {
        pImgC = cvLoadImage(last_file.c_str());
        cvNamedWindow("Image Under Test");
        cvMoveWindow("Image Under Test", 960 - (pImgC->width / 2), 384 - (pImgC->height / 2));
        //cvNamedWindow("Image Under Test", CV_WINDOW_AUTOSIZE);
        gettimeofday(&synt, NULL);
        while(synt.tv_sec % 8 != 0) {
            gettimeofday(&synt, NULL);
        }
        cvShowImage("Image Under Test", pImgC);
        cvWaitKey(1000);
        cvDestroyWindow("Image Under Test");
    }
    //insert your code here

    std::vector<Prediction> predictions;
    predictions = classifier.Classify(img, 5);

    if(i>=phase){
        std::cout << "---------- Prediction for " << fifo.front() << " ----------" << std::endl;
        fifo.pop();
        for (size_t i = 0; i < predictions.size(); ++i) {
          Prediction p = predictions[i];
          std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
        }
    }

    //TODO: show preduction result on the shown image
    
    IplImage *pImgG;
    if (i>=phase) {
        //pImgG = cvLoadImage(last_file.c_str(), 0);
        pImgG = cvCreateImage(cvGetSize(pImgC), IPL_DEPTH_8U, 1);       
        cvCvtColor(pImgC, pImgG, CV_RGB2GRAY);
        cvCvtColor(pImgG, pImgC, CV_GRAY2RGB);
        CvPoint P;
        P.x = 10;
        P.y = 25;
    
        for (size_t i = 0; i < predictions.size(); ++i) {
            Prediction p = predictions[i];
            pos = p.first.find(",");
            strID = ConvertPrecision(p.second) + "% - " + p.first.substr(10, pos-10);
            cvPutText(pImgC, strID.c_str(), cvPoint(P.x, P.y+i*25), &font, CV_RGB(255,0,0));
        }

        cvNamedWindow("Image Recognition Result on FPGA");
        cvMoveWindow("Image Recognition Result on FPGA", 960 - (pImgG->width / 2), 384 - (pImgG->height / 2));
        cvShowImage("Image Recognition Result on FPGA", pImgC);
        if ( i % 5 == 0) {
            c = cvWaitKey(0);
            gettimeofday(&synt, NULL);
            synt.tv_sec = (synt.tv_sec >> 3) * 8 + 4;
            settimeofday(&synt, NULL);
            switch (c) {
                case 31:
                    break;
                case 'e':
                    return 0;
            }
        }
        else {
            cvWaitKey(4000);
        }
        cvDestroyWindow("Image Recognition Result on FPGA");
        cvReleaseImage(&pImgG);
        cvReleaseImage(&pImgC);
    }
    //insert your code here
  }
  out.close();
  printf("List Prediction finish\n");
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV


