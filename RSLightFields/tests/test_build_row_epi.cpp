#include <opencv2/core/core.hpp>
#include <string>
#include <iostream>
#include <rslf.hpp>

/*
 * test_build_row_epi.cpp
 * 
 * Given a folder containing the SkySat images, build an EPI corresponding
 * to a given row and plot it.
 * Provide a desired row in argument (otherwise default).
 */

int main(int argc, char* argv[])
{
    int inspected_row = 380;
    
    if (argc > 1)
    {
        std::cout << "Selected line: " << argv[1] << std::endl;
        inspected_row = std::stoi(argv[1]);
    }
    
    // Load all images in folder
    std::vector<cv::Mat> list_mat = rslf::read_imgs_from_folder("../data/skysat_lasvegas_rectified/rectified_equalized_resized_frames_step18/", "tif", CV_LOAD_IMAGE_UNCHANGED);
    
    cv::Mat epi = rslf::build_row_epi_from_imgs(list_mat, inspected_row);
    
    // Plot the first image overlayed with a red line corresponding
    // to the epipolar plane
    rslf::plot_mat(list_mat[0], "000.tif", inspected_row);
    rslf::plot_mat(epi, "EPI");
    
    cv::waitKey();
    
    return 0;
}
