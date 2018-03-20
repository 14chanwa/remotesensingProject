#ifndef _RSLF_IO
#define _RSLF_IO

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

#include <rslf_types.hpp>

namespace rslf 
{
    
/**
 * Reads an image from the given path, name and extension.
 * 
 * @param path_to_folder Path to the folder containing the file.
 * @param name_we Name of the file without path or extension.
 * @param extension The extension of the file.
 * @param cv_read_mode The mode with which OpenCV will read the file.
 * @return The requested image in Mat format.
 */
Mat read_img_from_file
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
 * @return The requested images in a vector of Mat.
 */
std::vector<Mat> read_imgs_from_folder
(
    std::string path_to_folder,
    std::string extension,
    int cv_read_mode = CV_LOAD_IMAGE_GRAYSCALE 
);

/**
 * Writes the given Mat to a .yml file in the given folder.
 * 
 * @param img The Mat to save.
 * @param path_to_folder Path to the folder where to save the file.
 * @param name_we Name of the file without path or extension.
 * @param extension The extension of the file.
 */
void write_mat_to_yml
(
    Mat img,
    std::string path_to_folder, 
    std::string name_we,
    std::string extension = "yml"
);

/**
 * Writes the given Mat to an image in the given folder.
 * 
 * @param img The Mat to save.
 * @param path_to_folder Path to the folder where to save the file.
 * @param name_we Name of the file without path or extension.
 * @param extension The extension of the file.
 * @param compression_params Compression parameters.
 */
void write_mat_to_imgfile
(
    Mat img,
    std::string path_to_folder, 
    std::string name_we,
    std::string extension = "png",
    std::vector<int> compression_params = std::vector<int>()
);

/**
 * Reads the given .yml file in the given folder to a Mat.
 * 
 * @param path_to_folder Path to the folder where to read the file.
 * @param name_we Name of the file without path or extension.
 * @param extension The extension of the file.
 * @return A Mat containing the image
 */
Mat read_mat_from_yml
(
    std::string path_to_folder, 
    std::string name_we,
    std::string extension = "yml"
);

}

#endif
