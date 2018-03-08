#include <string>
#include <vector>
#include <iostream>
#include <limits>
#include <opencv2/core/core.hpp>
#include <rslf_depth_computation.hpp>


/*
 * Template specializations
 */
 
template<>
float rslf::nan_type<float>() {
    return std::numeric_limits<float>::quiet_NaN();
}

template<>
cv::Vec3f rslf::nan_type<cv::Vec3f>() {
    return cv::Vec3f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
}

template<>
bool rslf::is_nan_type<float>(float x) {
    return x != x;
}

template<>
bool rslf::is_nan_type<cv::Vec3f>(cv::Vec3f x) {
    return x[0] != x[0] || x[1] != x[1] ||x[2] != x[2];
}

template<>
float rslf::BandwidthKernel<float>::evaluate(float x) {
    // If none, return 0
    if (rslf::is_nan_type<float>(x))
        return 0;
    // Else return 1 - (x/h)^2 if (x/h)^2 < 1, else 0
    float tmp = std::pow(x / m_h_, 2);
    return (tmp > 1 ? 0 : 1 - tmp);
}

template<>
void rslf::BandwidthKernel<float>::evaluate_mat(const Mat& src, Mat& dst) {
    // (x/h)^2
    cv::multiply(src, src, dst, inv_m_h_sq);
    // 1 - (x/h)^2
    cv::subtract(1.0, dst, dst);
    // max( 1 - (x/h)^2, 0 ) ; this operation removes nan's
    cv::max(dst, 0.0, dst);
}

template<>
float rslf::BandwidthKernel<cv::Vec3f>::evaluate(cv::Vec3f x) {
    // If none, return 0
    if (rslf::is_nan_type<cv::Vec3f>(x))
        return 0;
    // Else return 1 - (x/h)^2 if (x/h)^2 < 1, else 0
    float tmp = std::pow(cv::norm(x) / m_h_, 2);
    return (tmp > 1 ? 0 : 1 - tmp);
}

template<>
void rslf::BandwidthKernel<cv::Vec3f>::evaluate_mat(const Mat& src, Mat& dst) {
    // (x/h)^2
    cv::multiply(src, src, dst, inv_m_h_sq);
    // sum over channels
    int rows = src.rows;
    int cols = src.cols;
    dst = dst.reshape(1, rows * cols);
    cv::reduce(dst, dst, 1, cv::REDUCE_SUM);
    dst = dst.reshape(1, rows);
    // 1 - (x/h)^2
    cv::subtract(1.0, dst, dst);
    // max( 1 - (x/h)^2, 0 ) ; this operation removes nan's
    cv::max(dst, 0.0, dst);
}
