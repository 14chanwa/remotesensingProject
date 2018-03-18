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
     * Fuse a pyramid of depths into one depth map and apply a median filter on top of it
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
        void plot_results(Vec<Mat>& out_plot_depth_s_v_u_, int cv_colormap);
        
        //~ Mat get_coloured_epi(int v = -1, int cv_colormap = cv::COLORMAP_JET);
        //~ Mat get_disparity_map(int s = -1, int cv_colormap = cv::COLORMAP_JET);
    
    private:
        Vec<Depth2DComputer<DataType>* > m_computers_;
        Vec<Depth1DParameters<DataType>* > m_parameter_instances_;
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
        
        int m_start_dim_v_ = tmp_epis.size();
        int m_start_dim_u_ = tmp_epis[0].cols;
        
        int m_dim_v_ = tmp_epis.size();
        int m_dim_u_ = tmp_epis[0].cols;
        
        while (m_dim_v_ > _MIN_SPATIAL_DIM && m_dim_u_ > _MIN_SPATIAL_DIM)
        {
            
            std::cout << "Creating Depth2DComputer with sizes (" << m_dim_v_ << ", " << m_dim_u_ << ")" << std::endl;
            
            Depth1DParameters<DataType>* new_parameters = new Depth1DParameters<DataType>();
            *new_parameters = parameters;
            // Compute scale factor
            new_parameters->m_slope_factor = (0.0 + m_dim_u_) / m_start_dim_u_;
            
            // Create a new Depth2DComputer
            Depth2DComputer<DataType>* computer = new Depth2DComputer<DataType>(tmp_epis, d_list, epi_scale_factor, *new_parameters);
            
            // Downsample
            Vec<Mat> downsampled_epis;
            downsample_EPIs(tmp_epis, downsampled_epis);
            tmp_epis = downsampled_epis;
            
            m_dim_v_ = tmp_epis.size();
            m_dim_u_ = tmp_epis[0].cols;
            
            m_computers_.push_back(computer);
            m_parameter_instances_.push_back(new_parameters);
        }
    }
    
    template<typename DataType>
    FineToCoarse<DataType>::~FineToCoarse()
    {
        for (int p=0; p<m_computers_.size(); p++)
        {
            std::cout << "delete " << p << std::endl;
            delete m_computers_[p];
            std::cout << "delete computer ok" << std::endl;
            delete m_parameter_instances_[p];
            std::cout << "delete parameter ok" << std::endl;
            std::cout << "ok" << std::endl;
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
    
    template<typename DataType>
    void FineToCoarse<DataType>::plot_results(Vec<Mat>& out_plot_depth_s_v_u_, int cv_colormap)
    {
        std::cout << "Plot depth results..." << std::endl;
        
        Vec<Mat> out_map_s_v_u_;
        Vec<Mat> out_validity_s_v_u_;
        
        get_results(out_map_s_v_u_, out_validity_s_v_u_);
    
        for (int s=0; s<out_map_s_v_u_.size(); s++)
        {
            
            cv::Mat disparity_map = rslf::copy_and_scale_uchar(out_map_s_v_u_[s]);
            cv::applyColorMap(disparity_map, disparity_map, cv_colormap);
            
            int m_dim_v_ = disparity_map.rows;
            int m_dim_u_ = disparity_map.cols;
            
            // Threshold scores
            cv::Mat disparity_map_with_scores = cv::Mat::zeros(m_dim_v_, m_dim_u_, disparity_map.type());
            
            cv::add(disparity_map, disparity_map_with_scores, disparity_map_with_scores, out_validity_s_v_u_[s]);
            
            out_plot_depth_s_v_u_.push_back(disparity_map_with_scores);
        }
    
    }
}

#endif
