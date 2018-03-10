#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <rslf_epi.hpp>
#include <rslf_io.hpp>
#include <experimental/filesystem>
#include <vector>


//~ #define _RSLF_EPI_VERBOSE
//~ #define _RSLF_EPI_DEBUG


rslf::Mat rslf::build_row_epi_from_imgs
(
    Vec<Mat> imgs,
    int row
)
{
    Mat img0 = imgs[0];
    Mat epi
    (
        imgs.size(), // rows
        img0.cols, // cols
        img0.type() // dtype
    );
#ifdef _RSLF_EPI_DEBUG
    std::cout << "Created Mat of size (" << imgs.size() << "x" << 
        img0.cols << "), dtype=" << rslf::type2str(img0.type()) << std::endl;
#endif

    // Builds rows by copy
#pragma omp parallel for
    for (int i = 0; i < imgs.size(); i++)
    {
        imgs[i].row(row).copyTo(epi.row(i));
    }
    
    return epi;
}

rslf::Vec<rslf::Mat> rslf::build_epis_from_imgs
(
    Vec<Mat> imgs
)
{
    Vec<Mat> epis;//(imgs[0].rows, cv::Mat::zeros(imgs.size(), imgs[0].cols, imgs[0].type()));
    //~ #pragma omp parallel for
    for (int v=0; v<imgs[0].rows; v++)
    {   
        Mat epi(imgs.size(), imgs[0].cols, imgs[0].type());
        // Builds rows by copy
        for (int s=0; s<imgs.size(); s++)
        {
            imgs[s].row(v).copyTo(epi.row(s));
        }
        epis.push_back(epi);
        
    }
    //~ for (int v=0; v<imgs[0].rows; v++)
    //~ {   
        //~ std::cout << epis[v].at<float>(0, 0) << std::endl;
    //~ }
    
    return epis;
}

rslf::Mat rslf::build_row_epi_from_path
(
    std::string path_to_folder,
    std::string extension,
    int row,
    int cv_read_mode
)
{
    // Get all valid item names in directory
    Vec<std::string> list_files;
    for (auto & p : std::experimental::filesystem::directory_iterator(path_to_folder)) 
    {
        std::string file_name = p.path().string();
        std::string current_extension = file_name.substr(file_name.find_last_of(".") + 1);
        std::string tmp = file_name.substr(file_name.find_last_of("/") + 1);
        std::string current_name = tmp.substr(0, tmp.find_last_of("."));
        if (current_extension == extension) 
        {
            list_files.push_back(current_name);
        } 
    }
    
    std::sort(list_files.begin(), list_files.end());
    
    // Sample image
    Mat tmp = read_img_from_file
        (
            path_to_folder, 
            list_files[0], 
            extension, 
            cv_read_mode
        );
    
    // Read items
    Mat epi
    (
        list_files.size(), // rows
        tmp.cols, // cols
        tmp.type() // dtype
    );
    
    // For each file, read the corresponding row
#pragma omp parallel for
    for (int i=0; i<list_files.size(); i++)
    {
#ifdef _RSLF_EPI_DEBUG
        std::cout << "Read " << current_name << "." << extension << std::endl;
#endif
        Mat img = read_img_from_file(path_to_folder, list_files[i], extension, cv_read_mode);
        img.row(row).copyTo(epi.row(i));
    }
    
    return epi;
}
