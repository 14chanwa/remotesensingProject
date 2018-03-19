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
    std::vector<cv::Mat> list_mat = rslf::read_imgs_from_folder("../data/skysat_lasvegas_rectified/rectified_equalized_resized_frames_step18/", "tif", CV_LOAD_IMAGE_UNCHANGED);
    //~ std::vector<cv::Mat> list_mat = rslf::read_imgs_from_folder("../data/mansion_image_resized/", "jpg", CV_LOAD_IMAGE_UNCHANGED);
    
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
    
    rslf::FineToCoarse<float> fine_to_coarse(epis, d_list);
    //~ rslf::FineToCoarse<cv::Vec3f> fine_to_coarse(epis, d_list);
    fine_to_coarse.run();
    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();
    std::cout << "Time elapsed: " << duration << " seconds" << std::endl;
    
    std::vector<cv::Mat> out_plot_depth_s_v_u_;
    fine_to_coarse.get_coloured_depth_maps(out_plot_depth_s_v_u_, cv::COLORMAP_HOT);
    std::vector<cv::Mat> out_plot_epi_pyr_p_s_u_;
    fine_to_coarse.get_coloured_epi_pyr(out_plot_epi_pyr_p_s_u_, -1, cv::COLORMAP_HOT);
    std::vector<cv::Mat> out_plot_depth_pyr_p_v_u_;
    fine_to_coarse.get_coloured_depth_pyr(out_plot_depth_pyr_p_v_u_, -1, cv::COLORMAP_HOT);
    
    // Save imgs 
    // Base name
    // https://stackoverflow.com/questions/31255486/c-how-do-i-convert-a-stdchronotime-point-to-long-and-back
    auto now = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now).time_since_epoch().count();
    
    std::string base_filename = std::to_string(ms);
    std::cout << "Base name: " << base_filename << std::endl;
    
    std::cout << "Plotting" << std::endl;
    
    for (int s=0; s<out_plot_depth_s_v_u_.size(); s++)
    {
        std::cout << "plotting s=" << s << std::endl;
        std::stringstream ss;
        ss << std::setw(3) << std::setfill('0') << s;
        std::string s_str = ss.str();
        
        rslf::write_mat_to_imgfile
        (
            out_plot_depth_s_v_u_[s],
            "../output/", 
            base_filename + "_dmap_" + s_str,
            "png"
        );
    }
    
    for (int p=0; p<out_plot_depth_pyr_p_v_u_.size(); p++)
    {
        std::stringstream ss;
        ss << std::setw(3) << std::setfill('0') << p;
        std::string s_str = ss.str();
    
        rslf::write_mat_to_imgfile
        (
            out_plot_depth_pyr_p_v_u_[p],
            "../output/", 
            base_filename + "_pyr_depth_" + s_str,
            "png"
        );
        rslf::write_mat_to_imgfile
        (
            out_plot_epi_pyr_p_s_u_[p],
            "../output/", 
            base_filename + "_pyr_epi_" + s_str,
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
