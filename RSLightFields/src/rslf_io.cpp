#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>
#include <rslf_io.hpp>
#include <experimental/filesystem>
#include <vector>


//#define _RSLF_IO_VERBOSE
//#define _RSLF_IO_DEBUG

cv::Mat rslf::read_img_from_file
(
    std::string path_to_folder, 
    std::string name_we,
    std::string extension,
    int cv_read_mode
) 
{
    cv::Mat img = cv::imread(path_to_folder + name_we + "." + extension, cv_read_mode);
    if (img.empty())
        std::cout << ">>> rslf::read_img_from_file -- WARNING: Empty image: " <<
            path_to_folder + name_we + "." + extension +
            " -- cv_read_mode=" << std::to_string(cv_read_mode) << std::endl;
    else
#ifdef _RSLF_IO_DEBUG
        std::cout << img.at<float>(0, 0) << std::endl;
#endif
    return img;
}

std::vector<cv::Mat> rslf::read_imgs_from_folder
(
    std::string path_to_folder,
    std::string extension,
    int cv_read_mode
) 
{
    // Get all valid item names in directory
    std::vector<std::string> list_files;
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
    
    // Read items
    std::vector<cv::Mat> imgs;
    for (std::string current_name : list_files)
    {
#ifdef _RSLF_IO_VERBOSE
        std::cout << "Read " << current_name << "." << extension << std::endl;
#endif
        cv::Mat img = read_img_from_file(path_to_folder, current_name, extension, cv_read_mode);
        imgs.push_back(img);
    }
    
    return imgs;
}


