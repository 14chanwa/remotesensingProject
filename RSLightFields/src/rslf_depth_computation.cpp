#include <string>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <rslf_depth_computation.hpp>


/*
 * Template specializations
 */


template<>
float rslf::interpolation_1d_nearest_neighbour<float>
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

template<>
rslf::BandwidthKernel<float>::BandwidthKernel(float h) : h_(h) {
    
}

template<>
float rslf::BandwidthKernel<float>::evaluate(float x) {
    // If none, return 0
    if (x != x)
        return 0;
    // Else return 1 - (x/h)^2 if (x/h)^2 < 1, else 0
    float tmp = std::pow(x / this->h_, 2);
    return (tmp > 1 ? 0 : 1 - tmp);
}
