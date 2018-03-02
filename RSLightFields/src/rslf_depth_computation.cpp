#include <string>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <rslf_depth_computation.hpp>


float rslf::interpolation_1d_nearest_neighbour
(
    cv::Mat line_matrix, 
    float index
) 
{
    int rounded_index = (int)std::round(index);
    if (rounded_index < 0 || rounded_index > line_matrix.cols - 1)
        return std::nan("");
    
    return line_matrix.at<float>(0, rounded_index);
}

rslf::DepthComputer1D::DepthComputer1D
(
    cv::Mat epi, 
    std::vector<float> d_list,
    Interpolation1D interpolation_function
) 
{
    
    this->epi_ = epi;
    this->d_list_ = d_list;
    
    this->interpolation_function_ = interpolation_function;
    
    // Build radiance matrix 
    this->dim_s_ = this->epi_.rows;
    this->dim_d_ = this->d_list_.size();
    this->dim_u_ = this->epi_.cols;
    
    int dims[3] = {this->dim_s_, this->dim_d_, this->dim_u_};
    this->radiances_s_d_u_ = cv::Mat(3, dims, CV_32FC1);
    
}

void rslf::DepthComputer1D::run() 
{
    // Iterate over all columns of the EPI
    for (int u=0; u<this->epi_.cols; u++)
    {
        // s_hat is the center horizontal line index of the epi
        int s_hat = (int) std::floor((0.0 + this->dim_s_) / 2);
        
        // Row matrix
        cv::Mat D = cv::Mat(1, this->dim_d_, CV_32FC1);
        for (int d=0; d<this->dim_d_; d++)
            D.at<float>(0, d) = d_list_[d];
        
        // Col matrix
        cv::Mat S = cv::Mat(this->dim_s_, 1, CV_32FC1);
        for (int s=0; s<this->dim_s_; s++)
            S.at<float>(s, 0) = s_hat - s;
        
        // Index matrix
        cv::Mat I = S * D;
        I += u;
        
        // Fill radiances
#pragma omp parallel for
        for (int s=0; s<I.rows; s++)
        {
#pragma omp parallel for
            for (int d=0; d<I.cols; d++)
            {
                this->radiances_s_d_u_.at<float>(s, d, u) =
                    this->interpolation_function_
                    (
                        this->epi_.row(s), 
                        I.at<float>(s, d)
                    );
            }
        }
    }
    
    // DEBUG
    // Try to get a 2D slice
    cv::Range range[3];
    range[0] = cv::Range::all();
    range[1] = cv::Range::all();
    range[2] = cv::Range(0,1);
    
    cv::Mat slice = radiances_s_d_u_(range);
    
    cv::Mat mat2D;
    mat2D.create(2, &(this->radiances_s_d_u_.size[0]), this->radiances_s_d_u_.type());
    slice.copySize(mat2D);
    
    //~ std::cout << slice << std::endl;
    
    
    
}
