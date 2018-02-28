#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


int main()
{
    
    std::cout << "Hello world!" << std::endl;

    /*
     * Read image from file
     */
    cv::Mat img;
    //~ img = cv::imread("../data/skysat_lasvegas_rectified/rectified_equalized_resized_frames_step18/000.tif", CV_LOAD_IMAGE_UNCHANGED);
    //~ img = cv::imread("../data/mansion_image/mansion_image_0000.jpg", CV_LOAD_IMAGE_UNCHANGED);
    //~ img = cv::imread("../data/CCITT_1.TIF", CV_LOAD_IMAGE_COLOR);
    img = cv::imread("../data/000.tif", CV_LOAD_IMAGE_UNCHANGED);
    
    /*
     * Display image
     */
    if (img.empty())
    {
        std::cout<< "Image not loaded" << std::endl;
    }
    else
    {
        cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
        cv::imshow("Image", img);
        cv::waitKey();
    }   
    
    return 0;
}
