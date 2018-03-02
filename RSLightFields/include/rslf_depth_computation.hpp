#ifndef _RSLF_DEPTH_COMPUTATION
#define _RSLF_DEPTH_COMPUTATION

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>


namespace rslf
{
    
    typedef float (* Interpolation1D)
    (
        cv::Mat line_matrix, 
        float index
    );

    float interpolation_1d_nearest_neighbour
    (
        cv::Mat line_matrix, 
        float index
    );


    class DepthComputer1D
    {
    public:
        DepthComputer1D
        (
            cv::Mat epi, 
            std::vector<float> d_list,
            Interpolation1D interpolation_function = (Interpolation1D)interpolation_1d_nearest_neighbour
        );
        void run();
    
    private:
        cv::Mat epi_;
        std::vector<float> d_list_;
        cv::Mat radiances_s_d_u_;

        Interpolation1D interpolation_function_;
        
        /**
         * Dimension along s axis
         */
        int dim_s_;
        /**
         * Dimension along d axis
         */
        int dim_d_;
        /**
         * Dimension along u axis
         */
        int dim_u_;
    };

}

#endif
