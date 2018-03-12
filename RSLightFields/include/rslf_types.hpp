#ifndef _RSLF_TYPES
#define _RSLF_TYPES    

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>


namespace rslf
{
    /*
     * Aliases
     */
    using Mat = cv::Mat;
    
    template<typename T>
    using Vec = std::vector<T>;
    
    /**
     * Gets an explicit form of an OpenCV type.
     * 
     * @param type The type to disambiguate.
     * @return A string describing the type.
     */
    std::string type2str(int type);
    
    /*
     * NaN
     */
    template<typename DataType>
    DataType nan_type();
    
    template<typename DataType>
    bool is_nan_type(DataType x);
    
    template<>
    bool is_nan_type<float>(float x);
    
    template<>
    bool is_nan_type<cv::Vec3f>(cv::Vec3f x);
    
    /*
     * Zero
     */
    template<typename DataType>
    cv::Scalar zero_scalar();
    
    template<>
    cv::Scalar zero_scalar<float>();
    
    template<>
    cv::Scalar zero_scalar<cv::Vec3f>();
    
    /*
     * Template norm
     */
    template<typename DataType>
    float norm(DataType x);
    
}

#endif
