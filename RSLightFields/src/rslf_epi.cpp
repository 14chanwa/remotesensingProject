#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <rslf_epi.hpp>
#include <rslf_utils.hpp>
#include <vector>


//~ #define _RSLF_EPI_VERBOSE
//~ #define _RSLF_EPI_DEBUG


cv::Mat rslf::build_row_epi_from_imgs
(
    std::vector<cv::Mat> imgs,
    int row
)
{
    cv::Mat img0 = imgs[0];
    cv::Mat epi
    (
        imgs.size(), // rows
        img0.cols, // cols
        img0.type() // dtype
    );
#ifdef _RSLF_EPI_DEBUG
    std::cout << "Created cv::Mat of size (" << imgs.size() << "x" << 
        img0.cols << "), dtype=" << rslf::type2str(img0.type()) << std::endl;
#endif

    // Builds rows by copy
    for (int i = 0; i < imgs.size(); i++)
    {
        imgs[i].row(row).copyTo(epi.row(i));
    }
    
    return epi;
}
