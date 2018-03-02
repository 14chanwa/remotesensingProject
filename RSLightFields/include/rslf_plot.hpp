#ifndef _RSLF_PLOT
#define _RSLF_PLOT

#include <string>
#include <opencv2/core/core.hpp>


namespace rslf
{
    
    /**
     * Plots the provided cv::Mat in a cv::namedWindow.
     * Optional: overlay with a vertical or horizontal red line.
     * 
     * @param img The matrix to be plotted.
     * @param window_name Name of the created window.
     * @param fill_row Fill a row in blank? -1 for no, else row index
     * @param fill_col Fill a col in blank? -1 for no, else col index
     */
    void plot_mat
    (
        cv::Mat img, 
        std::string window_name = "Image",
        int fill_row_red = -1,
        int fill_col_red = -1
    );
    
    /**
     * Builds a copy of the matrix scaled to uchar (0..255).
     * 
     * @param img The matrix to be scaled.
     * @return A copy of the matrix, scaled to uchar.
     */
    cv::Mat copy_and_scale_uchar
    (
        cv::Mat img
    );
    
    /**
     * Scale values to plottable (uchar 0..255) and overlays the given 
     * matrix with red lines.
     * 
     * @param img The matrix to be plotted.
     * @param fill_row Fill a row in blank? -1 for no, else row index
     * @param max_height Truncate the image to this height.
     * @param fill_col Fill a col in blank? -1 for no, else col index
     * @param max_width Truncate the image to this width.
     */
    cv::Mat draw_red_lines
    (
        cv::Mat img,
        int fill_row_red = -1,
        int max_height = -1,
        int fill_col_red = -1,
        int max_width = -1
    );
    
}


#endif
