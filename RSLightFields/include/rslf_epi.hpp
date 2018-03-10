#ifndef _RSLF_EPI
#define _RSLF_EPI

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <rslf_types.hpp>


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
    Mat build_row_epi_from_imgs
    (
        Vec<Mat> imgs,
        int row
    );
    
    Vec<Mat> build_epis_from_imgs
    (
        Vec<Mat> imgs
    );
    
    /**
     * Assume we have a set of rectified images such that the epipolar
     * planes are on the lines in a folder. Builds the EPI along the given row.
     * Images are supposed to be ordered in alphabetical order.
     * 
     * @param path_to_folder The path containing the images.
     * @param extension The extension of the images to be read
     * @param row The row along which the EPI will be built.
     * @param cv_read_mode The mode with which OpenCV will read the files.
     * @return The requested EPI.
     */
    Mat build_row_epi_from_path
    (
        std::string path_to_folder,
        std::string extension,
        int row,
        int cv_read_mode = CV_LOAD_IMAGE_GRAYSCALE
    );

}

#endif
