#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>
#include <rslf_io.hpp>
#include <experimental/filesystem>
#include <vector>


#define _RSLF_IO_VERBOSE
//~ #define _RSLF_IO_DEBUG


rslf::Mat rslf::read_img_from_file
(
    std::string path_to_folder, 
    std::string name_we,
    std::string extension,
    int cv_read_mode,
    bool transpose
) 
{
    Mat img = cv::imread(path_to_folder + name_we + "." + extension, cv_read_mode);
    if (img.empty())
        std::cout << ">>> rslf::read_img_from_file -- WARNING: Empty image: " <<
            path_to_folder + name_we + "." + extension +
            " -- cv_read_mode=" << std::to_string(cv_read_mode) << std::endl;
    else
#ifdef _RSLF_IO_DEBUG
        std::cout << img.at<float>(0, 0) << std::endl;
#endif
    
    if (transpose)
        cv::transpose(img, img);
    
    return img;
}

rslf::Vec<rslf::Mat> rslf::read_imgs_from_folder
(
    std::string path_to_folder,
    std::string extension,
    int cv_read_mode,
    bool transpose
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
    
    // Read items
    Vec<Mat> imgs;
    for (std::string current_name : list_files)
    {
#ifdef _RSLF_IO_VERBOSE
        std::cout << "Read " << current_name << "." << extension << std::endl;
#endif
        Mat img = read_img_from_file(path_to_folder, current_name, extension, cv_read_mode);
        
        if (transpose)
            cv::transpose(img, img);
        
        imgs.push_back(img);
    }
    
    return imgs;
}

void rslf::write_mat_to_yml
(
    Mat img,
    std::string path_to_folder, 
    std::string name_we,
    std::string extension
)
{
#ifdef _RSLF_IO_VERBOSE
    std::cout << "Write " << path_to_folder + name_we + "." + extension << std::endl;
#endif
    cv::FileStorage storage(path_to_folder + name_we + "." + extension, cv::FileStorage::WRITE);
    storage << "img" << img;
    storage.release();  
}

void rslf::write_mat_to_imgfile
(
    Mat img,
    std::string path_to_folder, 
    std::string name_we,
    std::string extension,
    Vec<int> compression_params
)
{
#ifdef _RSLF_IO_VERBOSE
    std::cout << "Write " << path_to_folder + name_we + "." + extension << std::endl;
#endif
    cv::imwrite(path_to_folder + name_we + "." + extension, img);
}

rslf::Mat rslf::read_mat_from_yml
(
    std::string path_to_folder, 
    std::string name_we,
    std::string extension
)
{
#ifdef _RSLF_IO_VERBOSE
    std::cout << "Read " << path_to_folder + name_we + "." + extension << std::endl;
#endif
    Mat img;
    cv::FileStorage storage(path_to_folder + name_we + "." + extension, cv::FileStorage::READ);
    storage.getFirstTopLevelNode() >> img;
    storage.release();
    return img;
}
