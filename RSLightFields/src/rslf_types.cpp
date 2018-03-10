#include <limits>

#include <rslf_types.hpp>


/*
 * NaN
 */
template<>
float rslf::nan_type<float>() 
{
    return std::numeric_limits<float>::quiet_NaN();
}

template<>
cv::Vec3f rslf::nan_type<cv::Vec3f>() 
{
    return cv::Vec3f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
}

template<>
bool rslf::is_nan_type<float>(float x) 
{
    return x != x;
}

template<>
bool rslf::is_nan_type<cv::Vec3f>(cv::Vec3f x) 
{
    return x[0] != x[0] || x[1] != x[1] ||x[2] != x[2];
}

/*
 * OpenCV type explanation
 */
// Courtesy of Octopus https://stackoverflow.com/a/17820615
std::string rslf::type2str(int type) 
{
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) 
    {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

/*
 * Template norm
 */
template<>
float rslf::norm<float>(float x) 
{
    return std::abs(x);
}

template<>
float rslf::norm<cv::Vec3f>(cv::Vec3f x) 
{
    return cv::norm(x);
}
