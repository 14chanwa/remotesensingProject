#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rslf_plot.hpp>
#include <rslf_utils.hpp>
#include <string>
#include <iostream>


void rslf::plot_mat
(
    cv::Mat img, 
    std::string window_name,
    int fill_row_red,
    int fill_col_red
)
{
    // Find min and max values
    double min, max;
    cv::minMaxLoc(img, &min, &max);
    
    // Copy and scale to greyscale
    cv::Mat tmp;
    img.convertTo(tmp, CV_8U, 255.0/(max-min));
    
    if (tmp.channels() == 1)
        // Convert to RGB
        cv::cvtColor(tmp, tmp, CV_GRAY2RGB);
    
    
    // Red color
    cv::Vec3b red;
    red.val[2] = 255;
    
    // Fill row?
    if (fill_row_red > -1) 
    {
        for (int j=0; j<tmp.cols; j++)
            tmp.at<cv::Vec3b>(fill_row_red, j) = red;
    }
    
    // Fill col?
    if (fill_col_red > -1) 
    {
        for (int i=0; i<tmp.rows; i++)
            tmp.at<cv::Vec3b>(i, fill_col_red) = red;
    }
    
    // Plot in window
    cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
    cv::imshow(window_name, tmp);
}
