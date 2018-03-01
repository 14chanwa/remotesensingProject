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
     * @param img The 1-channel matrix to be plotted.
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
    
}


#endif
