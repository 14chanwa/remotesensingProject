#ifndef _RSLF_FINE_TO_COARSE
#define _RSLF_FINE_TO_COARSE

#include <rslf_depth_computation_core.hpp>
#include <rslf_depth_computation.hpp>


#define _MIN_SPATIAL_DIM 10


namespace rslf
{

    
    /**
     * Downsamples given EPIs by a factor 2 in the spatial dimensions
     * 
     * @param in_epis Input EPIs
     * @param out_epis Output EPIs
     */
    void downsample_EPIs
    (
        const Vec<Mat>& in_epis, 
        Vec<Mat>& out_epis
    );
    
    
    /**
     * Fuse a pyramid of depths into one depth map
     * 
     * @param disp_pyr_p_s_v_u_ A pyramid of disparity maps (finest first)
     * @param validity_indicators_p_s_v_u_ A pyramid of validity indicators for the respective disparity maps
     * @param out_map_s_v_u_ The output fused map
     * @param out_validity_s_v_u_ The output disp validity mask
     */
    void fuse_disp_maps
    (
        const Vec<Vec<Mat >>& disp_pyr_p_s_v_u_, 
        const Vec<Vec<Mat> >& validity_indicators_p_s_v_u_, 
        Vec<Mat>& out_map_s_v_u_, 
        Vec<Mat>& out_validity_s_v_u_
    );
    

    /*
     * *****************************************************************
     * FineToCoarse
     * *****************************************************************
     */

    template<typename DataType>
    class FineToCoarse
    {
    public:
        FineToCoarse
        (
            const Vec<Mat>& epis, 
            const Vec<float>& d_list,
            float epi_scale_factor = -1,
            const Depth1DParameters<DataType>& parameters = Depth1DParameters<DataType>::get_default()
        );
        ~FineToCoarse();
        
        void run();
        void get_results(Vec<Mat>& out_map_s_v_u_, Vec<Mat>& out_validity_s_v_u_);
        
        //~ Mat get_coloured_epi(int v = -1, int cv_colormap = cv::COLORMAP_JET);
        //~ Mat get_disparity_map(int s = -1, int cv_colormap = cv::COLORMAP_JET);
    
    private:
        Vec<Depth2DComputer<DataType>*> m_computers_;
    };
    
    
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

    template<typename DataType>
    FineToCoarse<DataType>::FineToCoarse
    (
        const Vec<Mat>& epis, 
        const Vec<float>& d_list,
        float epi_scale_factor,
        const Depth1DParameters<DataType>& parameters
    )
    {
        Vec<Mat> tmp_epis = epis;
        
        int m_dim_v_ = tmp_epis.size();
        int m_dim_u_ = tmp_epis[0].cols;
        
        while (m_dim_v_ > _MIN_SPATIAL_DIM && m_dim_u_ > _MIN_SPATIAL_DIM)
        {
            
            std::cout << "Creating Depth2DComputer with sizes (" << m_dim_v_ << ", " << m_dim_u_ << ")" << std::endl;
            
            // Create a new Depth2DComputer
            Depth2DComputer<DataType>* computer = new Depth2DComputer<DataType>(tmp_epis, d_list, epi_scale_factor, parameters);
            
            // Downsample
            Vec<Mat> downsampled_epis;
            downsample_EPIs(tmp_epis, downsampled_epis);
            tmp_epis = downsampled_epis;
            
            m_dim_v_ = tmp_epis.size();
            m_dim_u_ = tmp_epis[0].cols;
            
            m_computers_.push_back(computer);
        }
    }
    
    template<typename DataType>
    FineToCoarse<DataType>::~FineToCoarse()
    {
        for (int p=0; p<m_computers_.size(); p++)
        {
            delete m_computers_[p];
        }
    }
    
    template<typename DataType>
    void FineToCoarse<DataType>::run()
    {
        for (int p=0; p<m_computers_.size(); p++)
        {
            m_computers_[p]->run();
        }
    }
    
    template<typename DataType>
    void FineToCoarse<DataType>::get_results(Vec<Mat>& out_map_s_v_u_, Vec<Mat>& out_validity_s_v_u_)
    {
        std::cout << "Getting results..." << std::endl;
        
        Vec<Vec<Mat >> disp_pyr_p_s_v_u_;
        Vec<Vec<Mat >> validity_indicators_p_s_v_u_;
        
        for (int p=0; p<m_computers_.size(); p++)
        {
            disp_pyr_p_s_v_u_.push_back(m_computers_[p]->get_depths_s_v_u_());
            validity_indicators_p_s_v_u_.push_back(m_computers_[p]->get_valid_depths_mask_s_v_u_());
        }
        
        fuse_disp_maps
        (
            disp_pyr_p_s_v_u_, 
            validity_indicators_p_s_v_u_, 
            out_map_s_v_u_, 
            out_validity_s_v_u_
        );
    }
    

}

#endif
