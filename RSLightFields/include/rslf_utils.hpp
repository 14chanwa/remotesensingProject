#ifndef _RSLF_UTILS
#define _RSLF_UTILS

#include <string>

namespace rslf
{
    
    /**
     * Gets an explicit form of an OpenCV type.
     * 
     * @param type The type to disambiguate.
     * @return A string describing the type.
     */
    std::string type2str(int type);
    
}


#endif
