#include "seg_test.cpp"
#include <vector>
#include <string>
using namespace std;
using namespace cv;

//添加绿色mask函数
Mat apply_mask(Mat src,Mat pred_map,float alpha=0.5)
{
  Scalar color= Scalar(34 ,139 ,34);
  vector< Scalar> colors;
  colors.push_back(color);//可添加多种色彩
  Mat out_image;
  out_image.create(pred_map.size(),src.type());  
  for(int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
          if(int(pred_map.at<uchar>(i,j))==0)
            {  out_image.at< Vec3b>(i,j)[0]=src.at< Vec3b>(i,j)[0]*(1-alpha)+colors[0][0]*alpha;   
               out_image.at< Vec3b>(i,j)[1]=src.at< Vec3b>(i,j)[1]*(1-alpha)+colors[0][1]*alpha;  
               out_image.at< Vec3b>(i,j)[2]=src.at< Vec3b>(i,j)[2]*(1-alpha)+colors[0][2]*alpha;
            } 
          else
            {
               out_image.at< Vec3b>(i,j)[0]=src.at< Vec3b>(i,j)[0];   
               out_image.at< Vec3b>(i,j)[1]=src.at< Vec3b>(i,j)[1];  
               out_image.at< Vec3b>(i,j)[2]=src.at< Vec3b>(i,j)[2];
            }   
        }
    }
  return out_image;
}
//视频处理函数
int video_process(string in_file,string out_file,Classifier classifier)
{
  VideoCapture capture(in_file);  
  VideoWriter writer(out_file, CV_FOURCC('D', 'I', 'V', 'X'), 20.0, Size(640, 360));
  Mat frame, output;
  if(!capture.isOpened())
    return -1;
  
  while(true)
  {
    capture>>frame;
    cout<<"高:"<<frame.rows<<"宽:"<<frame.cols<<endl;
    Mat image_roi;
    //处理输入尺寸不同的情况(1280*1080,1280*720,1280*960)
    if(frame.rows==1080)
    {
      Rect rect(0, 360, 1280, 720);
      image_roi = frame(rect);
      cout<<"Size:"<<image_roi.rows<<","<<image_roi.cols<<endl;
    }    
    else if(frame.rows==960)
    {
      Rect rect(0, 240, 1280, 720);
      image_roi = frame(rect);
      cout<<"Size:"<<image_roi.rows<<","<<image_roi.cols<<endl;
    } 
    else if(frame.rows==720)
    {
      image_roi = frame;
      cout<<"Size:"<<image_roi.rows<<","<<image_roi.cols<<endl;
    }
    else
    {cout<<"尺寸错误"<<endl;
     break;
    }
    Mat pred_map=classifier.Predict(image_roi);
    resize(image_roi, image_roi, Size(640,360));//进行图像resize
    //添加绿色mask
    Mat result=apply_mask(image_roi,pred_map,0.5);
    //变量point为轮廓点序列 使用drawContours( out_image, contours, i, color, 1, 8, hierarchy, 0, Point() );
    pair<vector<vector<Point> >,vector<Vec4i> > point_pair=classifier.contour_point();
    vector<vector<Point> > point(point_pair.first.begin(),point_pair.first.end());
    vector<Vec4i> hierarchy(point_pair.second.begin(),point_pair.second.end());
    //使用point轮廓点画轮廓     
       for( int i = 0; i< point.size(); i++ )
     {
       Scalar color = Scalar(0, 0, 255);//轮廓颜色
       drawContours( result, point, i, color, 1, 8);
     } //画轮廓
    char buffer[20];
    double time=classifier.proc_time();
    sprintf(buffer,"FPS is:%.4f",1000/time);
    string  fps(buffer);
    putText(result, fps, Point(40, 40), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 20,255), 2);
    writer<<result;
    imshow("frame",result);
    if((char)waitKey(1) == 'q')
      {break ;}
  }
  destroyAllWindows();
  return 0;
}


int main(int argc, char** argv) {
 
  ::google::InitGoogleLogging(argv[0]);
  string model_file   = "/home/chao/ICNet-master/evaluation/prototxt/sub124_640_test.prototxt";
  string trained_file = "/home/chao/ICNet-master/evaluation/snapshot640/snapshot_iter_40000.caffemodel"; //for visualization
  string mode;
  if(argc==2)
    {
      mode=argv[1];
    }
  else
    mode="cpu";
  Classifier classifier(model_file, trained_file,mode);
  string file ;
  cout<<"请输入文件:";
  while(cin>>file)//获取图片路径
  { 
     if(file.find(".jpg")<file.length() || file.find(".png")<file.length() || file.find(".jpeg")<file.length())
     {
       Mat img =  imread(file, 1);//opencv读取图片
       Mat image_roi;
       cout<<"高:"<<img.rows<<"宽:"<<img.cols<<endl;
       if(img.rows==1080)
    	{
      	  Rect rect(0, 360, 1280, 720);
     	  image_roi = img(rect);
    	}    
       else if(img.rows==960)
    	{
      	  Rect rect(0, 240, 1280, 720);
     	  image_roi = img(rect);
    	} 
    	else if(img.rows==720)
    	{
    	  image_roi = img;
    	}
    	else
       {cout<<"尺寸错误"<<endl;
        return 1;
       }
       Mat pred_map=classifier.Predict(image_roi);
       resize(image_roi, image_roi, Size(640,360));//进行图像resize
       Mat result=apply_mask(image_roi,pred_map,0.5);
       //变量point为轮廓点序列 使用drawContours( out_image, contours, i, color, 1, 8, hierarchy, 0, Point() );
       pair<vector<vector<Point> >,vector<Vec4i> > point_pair=classifier.contour_point();
       vector<vector<Point> > point(point_pair.first.begin(),point_pair.first.end());
       vector<Vec4i> hierarchy(point_pair.second.begin(),point_pair.second.end());
       vector<Point> p(point[0].begin(),point[0].end());//轮廓点数组
       //使用point轮廓点画轮廓     
       for( int i = 0; i< point.size(); i++ )
     {
       Scalar color = Scalar(0, 0, 255);//轮廓颜色
       drawContours( result, point, i, color, 1, 8);
     } //画轮廓

       imshow("frame",result); //显示      
       while(true)
         if((char)waitKey(1) == 'q')
             {destroyAllWindows();
               break ;}
     }
     //视频显示
     else if(file.find(".mp4")<file.length() || file.find(".avi")<file.length() || file.find(".mkv")<file.length())
     {
       string f=file.substr(file.find_last_of('/')+1,file.size()-file.find_last_of('/')-1);
       string out_file="../"+f.replace(f.find_last_of('.'),4,".avi");
       video_process(file,out_file,classifier);
     }
     cout<<"请输入文件路径(.jpg/.png/.mp4):";
  }
}
