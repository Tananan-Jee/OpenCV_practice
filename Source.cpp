#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


///////////////////  Images  //////////////////////

//int main(){
//
//    string path = "pics/semmon_illus.jpg";
//    Mat img = imread(path);
//    imshow("Image", img);
//    waitKey(0);
//}


///////////////////  video  //////////////////////
////
//int main(){
//
//    string path = "pics/test_video.mp4";
//
//    VideoCapture cap(path);
//    Mat img;
//    int i = 1;
//
//    while (true) {
//        cap.read(img);
//        if (img.empty()) break;
//        imshow("image.png", img);
//        waitKey(1);  //wait for 1 millisecond
//    }
//}

///////////////// capture  video  //////////////////////

//int main(){
//
//    string path = "pics/test_video.mp4";
//    
//    VideoCapture cap(path);
//    Mat img;
//    int i = 0;
//    printf("%f",cap.get(CAP_PROP_FPS));
//    while (true) {
//        cap.read(img);
//        string name = "pics/vdo/stop_video" + to_string(i) + ".png";
//        if(img.empty()) break;
//        if(i%30 ==0){
//            imwrite(name, img);
//        }
//        i++;
//    }
//
//}


///////////////////  change RGB  to BGR  //////////////////////

//int main(){
//
//    string path = "pics/semmon_illus.jpg";
//    Mat img = imread(path);
//    Mat sw_img;
//    cvtColor(img, sw_img, COLOR_RGB2BGR);
//    imshow("Original Image", img);
//    imshow("BGR Image", sw_img);
//    waitKey(0);
//}


///////////////////  rotate 45 degree  //////////////////////

//int main(){
//
//    string path = "pics/semmon_illus.jpg";
//    Mat img = imread(path);
//    Point center = Point( img.cols/2, img.rows/2 ); //center of rotation
//    double angle = 45.0; // + is counter clockwise
//    double scale = 0.5;
//    Mat rot_mat = getRotationMatrix2D( center, angle, scale );
//    Mat rotate;
//    warpAffine( img, rotate, rot_mat, rotate.size() );
//    imshow("Original Image", img);
//    imshow("Rotated Image", rotate);
//    waitKey(0);
//}

///////////////////  change RGB to grayscale  //////////////////////

//int main(){
//
//    string path = "pics/semmon_illus.jpg";
//    Mat img = imread(path);
//    Mat gray_img;
//    cvtColor(img, gray_img, COLOR_RGB2GRAY);
//    Mat gray_img_2 = imread(path, IMREAD_GRAYSCALE);
//    imshow("Original Image", img);
//    imshow("Grayscale Image (BGR2GRAY)", gray_img);
//    imshow("Grayscale Image (RGB2GRAY)", gray_img_2);
//    //When using IMREAD_GRAYSCALE, the codec's internal grayscale conversion will be used, if available. 
//    //Results may differ to the output of cvtColor(). 
//    //The reason is that there are multiple implementations of the grayscale conversion in play. 
//    //cvtColor() is THE opencv implementation and will be consistent across platforms. 
//    //When you use imread() to convert to grayscale, you are at the mercy of the platform-specific implementation of imread(). 
//    //so imread() returns slightly different grayscale values on each platform.
//    waitKey(0);
//}



/////////////////////  Detect edges in a grayscale image  //////////////////////

//int main(){
//
//    string path = "pics/semmon_illus.jpg";
//    Mat img = imread(path);
//    Mat gray_img;
//    cvtColor(img, gray_img, COLOR_RGB2GRAY);
//
//    ////////////////  Sobel  //////////////////////
//    Mat sobel_x;
//    Mat sobel_y;
//    Mat abs_sobel_x;
//    Mat abs_sobel_y;
//    Mat sobel;
//    Sobel(gray_img, sobel_x, CV_16S, 1, 0);
//    Sobel(gray_img, sobel_y, CV_16S, 0, 1);
//    // converting back to CV_8U
//    convertScaleAbs(sobel_x, abs_sobel_x);
//    convertScaleAbs(sobel_y, abs_sobel_y);
//    addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0, sobel);
//    imshow("Original Image", img);
//    imshow("Sobel", sobel);
//
//    ////////////////  Laplacian  //////////////////////
//    Mat lap;
//    Laplacian(gray_img, lap, CV_8U);
//    imshow("Laplacian", lap);
//
//    ////////////////  canny  //////////////////////
//    Mat can;
//    Mat detected_edges;
//    Mat dst;
//    int threshold=100;
//    int ratio=3;
//    blur(gray_img, detected_edges, Size(3,3) );
//    Canny(detected_edges, can, threshold, threshold*ratio);
//    dst = Scalar::all(0);
//    gray_img.copyTo( dst, can);
//    imshow("Canny", dst);
//    waitKey(0);
//}


/////////////////////  add noise + denoise  //////////////////////

//int main() {
//
//    string path = "pics/semmon_illus.jpg";
//    Mat img = imread(path);
//    Mat gray_img;
//    cvtColor(img, gray_img, COLOR_RGB2GRAY);
//    imshow("Grayscale Image", gray_img);
//    /////////////////////  add noise //////////////////////
//    //uniform random noise
//    Mat uniform_noise = Mat::zeros(gray_img.rows, gray_img.cols, CV_8UC1);
//    randu(uniform_noise, 0, 255);
//    //imshow("Uniform random noise", uniform_noise );
//    Mat noisy_image = gray_img.clone();
//    noisy_image = gray_img + uniform_noise * 0.2;
//    imshow("Noisy_image - Uniform noise", noisy_image);
//    /////////////////////  denoise  //////////////////////
//    Mat blur_img;
//    blur(noisy_image, blur_img, Size(4, 4));
//    imshow("Blur_image", blur_img);
//    waitKey(0);
//}
//add