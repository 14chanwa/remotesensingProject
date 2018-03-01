#ifndef _RSLF_IO
#define _RSLF_IO

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>


namespace rslf 
{
    
    /**
     * Reads an image from the given path, name and extension.
     * 
     * @param path_to_folder Path to the folder containing the file.
     * @param name_we Name of the file without path or extension.
     * @param extension The extension of the file.
     * @param cv_read_mode The mode with which OpenCV will read the file.
     * @return The requested image in cv::Mat format.
     */
    cv::Mat read_img_from_file
    (
        std::string path_to_folder, 
        std::string name_we,
        std::string extension,
        int cv_read_mode = CV_LOAD_IMAGE_GRAYSCALE
    );
    
    /**
     * Reads all images with requested extension in requested folder.
     * 
     * @param path_to_folder Path to the folder containing the files.
     * @param extension The extension of the files.
     * @param cv_read_mode The mode with which OpenCV will read the files.
     * @return The requested images in a vector of cv::Mat.
     */
    std::vector<cv::Mat> read_imgs_from_folder
    (
        std::string path_to_folder,
        std::string extension,
        int cv_read_mode = CV_LOAD_IMAGE_GRAYSCALE 
    );

}

#endif
