//This script visualize the semantic segmentation for your input image.
#ifndef SEG_TEST_HPP
#define SEG_TEST_HPP
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

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,string mode);//初始化网络设置


  Mat Predict(const  Mat& img);//预测主函数
  double proc_time()
   {return totalTime;}
  pair<vector<vector<Point> >,vector<Vec4i> > contour_point()
   {return contour;}
 private:

  void Init_img(const  Mat& img,vector< Mat>* input_channels);//图像初始化辅助函数

  void Visualization( Mat pred_map);//opencv处理图像显示

 private:
  shared_ptr<Net<float> > net_;
  Size input_geometry_;
  int num_channels_;
  double totalTime;
  pair<vector<vector<Point> >,vector<Vec4i> > contour;
};
#endif
