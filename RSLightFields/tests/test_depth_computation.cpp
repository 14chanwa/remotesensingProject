#include <opencv2/core/core.hpp>
#include <string>
#include <iostream>
#include <rslf.hpp>
#include <rslf_depth_computation.hpp>
#include <chrono>

/*
 * test_depth_computation.cpp
 * 
 * Compute 1D depth (on center line of an epi).
 */
 
int read_skysat_lasvegas_rectified(int inspected_row);
int read_mansion_image(int inspected_row);

int main(int argc, char* argv[])
{

    // Select row
    int inspected_row = 380;
    if (argc > 1)
    {
        std::cout << "Selected line: " << argv[1] << std::endl;
        inspected_row = std::stoi(argv[1]);
    }
    
    // Load all images in folder
    std::vector<cv::Mat> list_mat = rslf::read_imgs_from_folder("../data/skysat_lasvegas_rectified/rectified_equalized_resized_frames_step18/", "tif", CV_LOAD_IMAGE_UNCHANGED);
    
    std::cout << list_mat.size() << " images read" << std::endl;
    
    cv::Mat epi = rslf::build_row_epi_from_imgs(list_mat, inspected_row);
    
    std::vector<float> d_list;
    for (int i=0; i<41; i++)
        d_list.push_back(-2 + 0.1 * i);
    
    std::cout << d_list.size() << " d values requested" << std::endl;
    
    rslf::DepthComputer1D<float> depth_computer_1d(epi, d_list);
    depth_computer_1d.run();
    
    return 0;
}

