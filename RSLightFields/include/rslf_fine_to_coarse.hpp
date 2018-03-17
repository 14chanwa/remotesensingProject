#ifndef _RSLF_FINE_TO_COARSE
#define _RSLF_FINE_TO_COARSE

#include <rslf_depth_computation_core.hpp>


namespace rslf
{

    
    /**
     * Downsamples given EPIs by a factor 2 in the spatial dimensions
     * 
     * @param in_epis Input EPIs
     * @param out_epis Output EPIs
     */
    void downsample_EPIs(const Vec<Mat>& in_epis, Vec<Mat>& out_epis);
    
    
    /**
     * Fuse a pyramid of depths into one depth map
     * 
     * @param disp_pyr_p_s_v_u_ A pyramid of disparity maps (finest first)
     * @param validity_indicators_p_s_v_u_ A pyramid of validity indicators for the respective disparity maps
     * @param out_map_s_v_u_ The output fused map
     * @param out_validity_s_v_u_ The output disp validity mask
     */
    void fuse_disp_maps(const Vec<Vec<Mat >>& disp_pyr_p_s_v_u_, const Vec<Vec<Mat> >& validity_indicators_p_s_v_u_, Vec<Mat>& out_map_s_v_u_, Vec<Mat>& out_validity_s_v_u_);
    
}

#endif
