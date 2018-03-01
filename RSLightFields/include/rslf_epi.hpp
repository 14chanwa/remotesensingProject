#ifndef _RSLF_EPI
#define _RSLF_EPI

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>


namespace rslf 
{

    /**
     * Assume we have a set of rectified images such that the epipolar
     * planes are on the lines. Builds the EPI along the given row.
     * 
     * @param imgs The set of images from which to build the EPI.
     * @param row The row along which the EPI will be built.
     * @return The requested EPI.
     */
    cv::Mat build_row_epi_from_imgs
    (
        std::vector<cv::Mat> imgs,
        int row
    );

}

#endif
