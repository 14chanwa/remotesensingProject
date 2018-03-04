#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rslf_plot.hpp>
#include <rslf_utils.hpp>
#include <string>
#include <iostream>

#define _RSLF_MAX_WINDOW_DIMENSION 800

void rslf::plot_mat
(
    cv::Mat img, 
    std::string window_name
)
{
    // Copy and scale values
    cv::Mat tmp = rslf::copy_and_scale_uchar(img);
    
    // Plot in window
    // Ratio
    float ratio = (tmp.cols + 0.0) / tmp.rows;
    int window_cols, window_rows;
    if (tmp.cols >= tmp.rows)
    {
        window_cols = std::min(_RSLF_MAX_WINDOW_DIMENSION, tmp.cols);
        window_rows = (int)std::ceil(window_cols / ratio);
    }
    else
    {
        window_rows = std::min(_RSLF_MAX_WINDOW_DIMENSION, tmp.rows);
        window_cols = (int)std::ceil(window_rows * ratio);
    }
    cv::namedWindow(window_name, CV_WINDOW_NORMAL);
    cv::resizeWindow(window_name, window_cols, window_rows);
    cv::imshow(window_name, tmp);
}

cv::Mat rslf::copy_and_scale_uchar
(
    cv::Mat img
)
{
    cv::Mat res;
    img.copyTo(res);
    // If the dtype is not a multiple of uchar
    if (img.type() % 8 != 0) {
        // Copy and scale values
        // Find min and max values
        double min, max;
        cv::minMaxLoc(img, &min, &max);
        res -= min;
        // Copy and scale values to uchar
        res.convertTo(res, CV_8U, 255.0/(max-min));
    } 
    //~ else 
    //~ {
        //~ // Only copy values
        //~ img.copyTo(res);
    //~ }
    return res;
}

cv::Mat rslf::draw_red_lines
(
    cv::Mat img,
    int fill_row_red,
    int max_height,
    int fill_col_red,
    int max_width
)
{
    // Copy and scale values
    cv::Mat res = rslf::copy_and_scale_uchar(img);
    
    if (fill_row_red < 0 && fill_col_red < 0)
        // Nothing to do
        return res;
    
    if (img.channels() == 1)
        // Convert to RGB
        cv::cvtColor(res, res, CV_GRAY2RGB);
    
    // Red color
    cv::Vec3b red;
    red.val[2] = 255;
    
    // Fill row?
    if (fill_row_red > -1) 
    {
        for (int j=0; j<res.cols; j++)
            res.at<cv::Vec3b>(fill_row_red, j) = red;
    }
    
    // Fill col?
    if (fill_col_red > -1) 
    {
        for (int i=0; i<res.rows; i++)
            res.at<cv::Vec3b>(i, fill_col_red) = red;
    }
    
    // Truncate the image along rows
    if (fill_row_red > -1 && max_height > 0)
    {
        int first_row, last_row;
        // Get the first and last rows to be copied
        // First row
        if (fill_row_red - max_height < 0)
            first_row = 0;
        else
            first_row = fill_row_red - max_height / 2;
        // Last row
        if (first_row + max_height < res.rows)
            last_row = first_row + max_height;
        else
            last_row = res.rows - 1;
        
        // Get a copy of the image between first and last row
        cv::Mat tmp;
        cv::Range ranges[2];
        ranges[0] = cv::Range(first_row, last_row); 
        ranges[1] = cv::Range::all();
        res(ranges).copyTo(tmp);
        res = tmp;
    }
    
    // Truncate the image along cols
    if (fill_col_red > -1 && max_width > 0)
    {
        int first_col, last_col;
        // Get the first and last cols to be copied
        // First col
        if (fill_col_red - max_width < 0)
            first_col = 0;
        else
            first_col = fill_col_red - max_width / 2;
        // Last row
        if (first_col + max_width < res.cols)
            last_col = first_col + max_width;
        else
            last_col = res.cols - 1;
        
        // Get a copy of the image between first and last row
        cv::Mat tmp;
        cv::Range ranges[2];
        ranges[0] = cv::Range::all();
        ranges[0] = cv::Range(first_col, last_col); 
        res(ranges).copyTo(tmp);
        res = tmp;
    }
    
    return res;
}
