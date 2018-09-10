//
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "seg_test.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;

//构造函数
Classifier::Classifier(const string& model_file,
                       const string& trained_file,string mode="cpu") {

  if(mode=="gpu")
  {Caffe::set_mode(Caffe::GPU);}
  else
  {Caffe::set_mode(Caffe::CPU);}
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  input_geometry_ =  Size(input_layer->width(), input_layer->height());
}

//预测功能函数
  Mat Classifier::Predict(const  Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  vector< Mat> input_channels;
  Init_img(img, &input_channels);
  //计时
  struct timeval time;
  gettimeofday(&time, NULL); // Start Time
  totalTime = (time.tv_sec * 1000) + (time.tv_usec / 1000);
  vector<Blob<float>*> input_vec;
  net_->Forward(input_vec);//need parameters
  gettimeofday(&time, NULL);  //END-TIME
  totalTime = (((time.tv_sec * 1000) + (time.tv_usec / 1000)) - totalTime);
  cout << "Processing time:" << totalTime << " ms" <<  endl;
  /* Copy the output layer to a  vector */
  Blob<float>* output_layer = net_->output_blobs()[0];

  int width = output_layer->width();
  int height = output_layer->height(); 
  int channels = output_layer->channels();
  int num = output_layer->num();
  // compute argmax
  Mat class_each_row (channels, width*height, CV_32FC1, const_cast<float *>(output_layer->cpu_data()));
  class_each_row = class_each_row.t(); // transpose to make each row with all probabilities
  Point maxId;    // point [x,y] values for index of max
  double maxValue;    // the holy max value itself
  Mat prediction_map(height, width, CV_8UC1);
  for (int i=0;i<class_each_row.rows;i++){
      minMaxLoc(class_each_row.row(i),0,&maxValue,0,&maxId); // 
      prediction_map.at<uchar>(i) = maxId.x;     
  }
  cout<<"开始显示"<< endl;
  Mat map;
  map=prediction_map.clone();
  Visualization(prediction_map);
  return map;
}
//可以放在构造函数中直接初始化
void Classifier::Init_img(const  Mat& img,vector< Mat>* input_channels) {
  //Convert the input image to the input image format of the network.
  Blob<float>* input_layer = net_->input_blobs()[0];
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();//cpu数据地址
  for (int i = 0; i < input_layer->channels(); ++i) {
    Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
  Mat img_resized;
  if (img.size() != input_geometry_)
     resize(img, img_resized, input_geometry_);
  else
    img_resized = img;
  Mat img_float;
  img_resized.convertTo(img_float, CV_32FC3);
  Mat img_norm;
  subtract(img_float,  Scalar(89.661, 86.524, 90.404), img_norm);
  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the  Mat
   * objects in input_channels. */
  split(img_norm, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

void Classifier::Visualization(Mat pred_map) {
  //找出轮廓点
  Mat src_gray=pred_map;
  Mat canny_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  Canny( src_gray*50, canny_output, 50, 100, 3 );
  findContours( canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0) );
  contour=make_pair(contours,hierarchy);
}

