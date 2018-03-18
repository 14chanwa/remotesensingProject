#include <opencv2/core/core.hpp>
#include <string>
#include <iostream>
#include <iomanip>
#include <rslf.hpp>
#include <rslf_fine_to_coarse.hpp>
#include <chrono>

/*
 * test_fine_to_coarse.cpp
 * 
 * 
 */
 

int main(int argc, char* argv[])
{
    
    std::cout << "Started fine to coarse computation..." << std::endl;
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    
    // Load all images in folder
    //~ std::vector<cv::Mat> list_mat = rslf::read_imgs_from_folder("../data/skysat_lasvegas_rectified/rectified_equalized_resized_frames_step18/", "tif", CV_LOAD_IMAGE_UNCHANGED);
    std::vector<cv::Mat> list_mat = rslf::read_imgs_from_folder("../data/mansion_image_resized/", "jpg", CV_LOAD_IMAGE_UNCHANGED);
    
    std::cout << list_mat.size() << " images read" << std::endl;
    
    std::vector<cv::Mat> epis = rslf::build_epis_from_imgs(list_mat);
    
    //~ rslf::plot_mat(epis[0], "EPI 0");
    //~ rslf::plot_mat(epis[500], "EPI 500");
    //~ cv::waitKey();
    
    std::vector<float> d_list;
    float d_min = -2.0;
    float d_max = 4.0;
    float interval = 0.05;
    for (int i=0; i<(d_max-d_min)/interval; i++)
        d_list.push_back(d_min + interval * i);
    
    std::cout << d_list.size() << " d values requested" << std::endl;
    
    //~ rslf::FineToCoarse<float> fine_to_coarse(epis, d_list);
    rslf::FineToCoarse<cv::Vec3f> fine_to_coarse(epis, d_list);
    fine_to_coarse.run();
    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();
    std::cout << "Time elapsed: " << duration << " seconds" << std::endl;
    
    std::vector<cv::Mat> out_map_s_v_u_;
    std::vector<cv::Mat> out_validity_s_v_u_;
    fine_to_coarse.get_results(out_map_s_v_u_, out_validity_s_v_u_);
    
    // Save imgs 
    // Base name
    // https://stackoverflow.com/questions/31255486/c-how-do-i-convert-a-stdchronotime-point-to-long-and-back
    auto now = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now).time_since_epoch().count();
    
    std::string base_filename = std::to_string(ms);
    std::cout << "Base name: " << base_filename << std::endl;
    
    std::cout << "Plotting" << std::endl;
    
    int cv_colormap = cv::COLORMAP_JET;
    
    for (int s=0; s<epis[0].rows; s++)
    {
        
        cv::Mat disparity_map = rslf::copy_and_scale_uchar(out_map_s_v_u_[s]);
        cv::applyColorMap(disparity_map, disparity_map, cv_colormap);
        
        int m_dim_v_ = disparity_map.rows;
        int m_dim_u_ = disparity_map.cols;
        
        // Threshold scores
        cv::Mat disparity_map_with_scores = cv::Mat::zeros(m_dim_v_, m_dim_u_, disparity_map.type());
        
        cv::add(disparity_map, disparity_map_with_scores, disparity_map_with_scores, out_validity_s_v_u_[s]);
        
        std::stringstream ss;
        ss << std::setw(3) << std::setfill('0') << s;
        std::string s_str = ss.str();
        
        rslf::write_mat_to_imgfile
        (
            disparity_map_with_scores,
            "../output/", 
            base_filename + "_dmap_" + s_str,
            "png"
        );
    }
    
    //~ cv::Mat coloured_epi = depth_computer_2d.get_coloured_epi();
    //~ rslf::write_mat_to_imgfile
    //~ (
        //~ coloured_epi,
        //~ "../output/", 
        //~ base_filename + "_epi_colored",
        //~ "png"
    //~ );
    
    //~ rslf::plot_mat(coloured_epi, "EPI + depth");
    
    //~ cv::waitKey();
    
    
    
    return 0;
}

