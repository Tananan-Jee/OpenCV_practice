#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
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
//    cvtColor(img, gray_img, COLOR_BGR2GRAY);
//    Mat gray_img_2 = imread(path, IMREAD_GRAYSCALE);
//    imshow("Original Image", img);
//    imshow("Grayscale Image (BGR2GRAY)", gray_img);
//    imshow("Grayscale Image (imread)", gray_img_2);
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
//    cvtColor(img, gray_img, COLOR_BGR2GRAY);
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
//    cvtColor(img, gray_img, COLOR_BGR2GRAY);
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
//    Mat gaussian_blur_img;
//    blur(noisy_image, blur_img, Size(4, 4));
//    GaussianBlur(noisy_image, gaussian_blur_img, Size(7, 7), 0); //kernel should be positive and odd
//    imshow("Blur image", blur_img);
//    imshow("Gaussian Blur image", gaussian_blur_img);
//    waitKey(0);
//}



/////////////////////  Binarize a grayscale image  //////////////////////

//int main(){
//  
//    Mat bi_gray;
//    int threshold_bi;
//
//    string path = "pics/semmon_illus.jpg";
//    Mat img = imread(path);
//
//    ///  grayscale image  ///
//    Mat gray_img;
//    cvtColor(img, gray_img, COLOR_BGR2GRAY);
//    imshow("Grayscale Image", gray_img);
//
//    ///  Binarize a grayscale image  ///
//    namedWindow("Threshold", WINDOW_AUTOSIZE); //create trackbar window
//    createTrackbar("value", "Threshold", &threshold_bi, 255); 
//    while(true) 
//    {
//        threshold(gray_img, bi_gray, threshold_bi, 255, THRESH_BINARY);
//        imshow("Binarize grayscale image", bi_gray);
//
//        int iKey = waitKey(50); // Wait until user press some key for 50ms
//        if (iKey == 27)  //if user press 'ESC' key -> close window
//        {
//            break;
//        }
//    }
//    
//    waitKey(0);
//}


/////////////////////  Apply labeling operation to a binarized image  //////////////////////

//int main(){
//  
//    string path = "pics/semmon_illus.jpg";
//    Mat img = imread(path);
//
//    ///  grayscale image  ///
//    Mat gray_img;
//    cvtColor(img, gray_img, COLOR_BGR2GRAY);
//    imshow("Grayscale Image", gray_img);
//
//    ///  Binarize a grayscale image  ///
//    int threshold_bi = 127;
//    Mat bi_gray;
//    threshold(gray_img, bi_gray, threshold_bi, 255, THRESH_BINARY);
//    imshow("Binarize grayscale image", bi_gray);
//    Mat labelimage(bi_gray.size(), CV_32S);
//    int nLabels = connectedComponents(bi_gray, labelimage, 8);
//    vector<Vec3b> colors(nLabels);
//    colors[0] = Vec3b(0, 0, 0);//background
//    for (int label = 1; label < nLabels; ++label) {
//        colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
//    }
//    Mat dst(img.size(), CV_8UC3);
//    for (int r = 0; r < dst.rows; ++r) {
//        for (int c = 0; c < dst.cols; ++c) {
//            int label = labelimage.at<int>(r, c);
//            Vec3b& pixel = dst.at<Vec3b>(r, c);
//            pixel = colors[label];
//        }
//    }
//    imshow("Connected Components", dst);
//    
//    waitKey(0);
//}



/////////////////////  Do single-camera calibration and get intrinsic & extrinsic parameters  //////////////////////

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{ 7,7 };

int main()
{
    // Creating vector to store vectors of 3D points for each checkerboard image
    vector<vector<Point3f> > objpoints;

    // Creating vector to store vectors of 2D points for each checkerboard image
    vector<vector<Point2f> > imgpoints;

    // Defining the world coordinates for 3D points
    vector<Point3f> objp;
    for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
    {
        for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
            objp.push_back(Point3f(j*2.1, i*2.1, 0));
    }


    // Extracting path of individual image stored in a given directory
    vector<String> images;
    // Path of the folder containing checkerboard images
    string path = "pics/chess/*.jpg";

    glob(path, images);

    Mat frame, gray;
    // vector to store the pixel coordinates of detected checker board corners 
    vector<Point2f> corner_pts;
    bool success;

    // Looping over all the images in the directory
    for (int i{ 0 }; i < images.size(); i++)
    {
        frame = imread(images[i]);
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Finding checker board corners
        // If desired number of corners are found in the image then success = true  
        success = findChessboardCorners(gray, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        
        /*
         * If desired number of corner are detected,
         * we refine the pixel coordinates and display
         * them on the images of checker board
        */
        if (success)
        {
            TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);

            // refining pixel coordinates for given 2d points.
            cornerSubPix(gray, corner_pts, Size(11, 11), Size(-1, -1), criteria);

            // Displaying the detected corner points on the checker board
            drawChessboardCorners(frame, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }

        imshow("Image", frame);
        waitKey(0);
    }

    destroyAllWindows();

    Mat cameraMatrix, distCoeffs, R, T, inpara, expara, perview;

    /*
     * Performing camera calibration by
     * passing the value of known 3D points (objpoints)
     * and corresponding pixel coordinates of the
     * detected corners (imgpoints)
    */
    calibrateCamera(objpoints, imgpoints, Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T, inpara, expara, perview);

    cout << "cameraMatrix : " << cameraMatrix << endl;
    cout << "distCoeffs : " << distCoeffs << endl;
    cout << "Rotation vector : " << R << endl;
    cout << "Translation vector : " << T << endl;
    cout << "Intrinsics parameters : " << expara << endl;
    cout << "Extrinsics parameters : " << distCoeffs << endl;
    cout << "perViewErrors : " << perview << endl;


    /////////////////////  Display the reprojected points on the captured image based on the estimated parameters in single-camera calibration  //////////////////////

    Mat objectPoints(4.2, 4.2, 0);
    vector<int> points2d(2);
    int projectx;
    int projecty;
    Mat rvec;
    Mat tvec;

    // Looping over all the images in the directory
    for (int i{ 0 }; i < images.size(); i++)
    {
        Mat bg = imread(images[i], CV_8UC3);
        rvec = R.row(i).clone();
        tvec = T.row(i).clone();
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, points2d);
        projectx = points2d.at(0);
        projecty = points2d.at(1);
        circle(bg, Point(projectx,projecty), 3, Scalar(255,0,0));
        imshow("Projected Point", bg);

        waitKey(0);
    }
    


    return 0;
}

