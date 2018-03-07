#ifndef _RSLF_TYPES
#define _RSLF_TYPES    

#include <vector>
#include <opencv2/core/core.hpp>

namespace rslf
{
    /*
     * Aliases
     */
    using Mat = cv::Mat;
    
    template<typename T>
    using Vec = std::vector<T>;
}

#endif
