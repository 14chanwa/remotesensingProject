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
            float d_min,
            float d_max,
            int dim_d,
            float epi_scale_factor = -1,
            const Depth1DParameters<DataType>& parameters = Depth1DParameters<DataType>::get_default()
        );
        ~FineToCoarse();
        
        void run();
        
        void get_results(Vec<Mat>& out_map_s_v_u_, Vec<Mat>& out_validity_s_v_u_);
        
        void get_coloured_depth_maps(Vec<Mat>& out_plot_depth_s_v_u_, int cv_colormap = cv::COLORMAP_JET);
        void get_coloured_epi_pyr(Vec<Mat>& out_plot_epi_pyr_p_s_u_, int v = -1, int cv_colormap = cv::COLORMAP_JET);
        void get_coloured_depth_pyr(Vec<Mat>& out_plot_depth_pyr_p_v_u_, int s = -1, int cv_colormap = cv::COLORMAP_JET);
        
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
        float d_min,
        float d_max,
        int dim_d,
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
            
            Depth1DParameters<DataType>* new_parameters = new Depth1DParameters<DataType>(parameters);
            
            // Compute scale factor
            new_parameters->m_slope_factor = (0.0 + m_dim_u_) / m_start_dim_u_;
            
            // Create a new Depth2DComputer
            Depth2DComputer<DataType>* computer = new Depth2DComputer<DataType>(tmp_epis, d_min, d_max, dim_d, epi_scale_factor, *new_parameters);
            
            // Downsample
            Vec<Mat> downsampled_epis;
            downsample_EPIs(tmp_epis, downsampled_epis);
            tmp_epis = downsampled_epis;
            
            m_dim_v_ = tmp_epis.size();
            m_dim_u_ = tmp_epis[0].cols;
            
            m_computers_.push_back(computer);
            m_parameter_instances_.push_back(new_parameters);
        }
        
        // The last level should accept all disparity measures
        m_computers_.back()->set_accept_all(true);
    }
    
    template<typename DataType>
    FineToCoarse<DataType>::~FineToCoarse()
    {
        for (int p=0; p<m_computers_.size(); p++)
        {
            delete m_computers_[p];
            delete m_parameter_instances_[p];
        }
    }
    
    template<typename DataType>
    void FineToCoarse<DataType>::run()
    {
        for (int p=0; p<m_computers_.size(); p++)
        {
            m_computers_[p]->run();
            if (p < m_computers_.size() - 1)
            {
                std::cout << "Setting new depth bounds..." << std::endl;
                
                // Depth map
                Vec<Mat> depth_map_up = m_computers_[p]->get_depths_s_v_u_();
                Vec<Mat> depth_map_down = m_computers_[p+1]->get_depths_s_v_u_();
                // Validity mask
                Vec<Mat> mask_up = m_computers_[p]->get_valid_depths_mask_s_v_u_();
                Vec<Mat> mask_down = m_computers_[p+1]->get_valid_depths_mask_s_v_u_();
                
                // dmin map
                Vec<Mat> dmin_map = m_computers_[p+1]->edit_dmin();
                // dmax map
                Vec<Mat> dmax_map = m_computers_[p+1]->edit_dmax();
                
                int m_dim_s_ = depth_map_up.size();
                
                int m_dim_v_up = depth_map_up[0].rows;
                int m_dim_u_up = depth_map_up[0].cols;
                
                int m_dim_v_down = depth_map_down[0].rows;
                int m_dim_u_down = depth_map_down[0].cols;
                
                // Edit the min/max d's
#pragma omp parallel for
                for (int s=0; s<m_dim_s_; s++)
                {
                    for (int v=0; v<m_dim_v_down; v++)
                    {
                        for (int u=0; u<m_dim_u_down; u++)
                        {
                            Vec<float> candidate_ds;
                            // 1st line
                            // Get the upscaled u, v
                            int v_up = std::min(2 * v, m_dim_v_up - 1);
                            int u_up = std::min(2 * u, m_dim_u_up - 1);
                            // Get the point at the left
                            int u_left = u_up;
                            float d_left = nan_type<float>();
                            while (u_left > 1)
                            {
                                u_left -= 1;
                                if (mask_up[s].at<uchar>(v_up, u_left) > 0)
                                {
                                    d_left = depth_map_up[s].at<float>(v_up, u_left);
                                    break;
                                }
                            }
                            int u_right = u_up;
                            float d_right = nan_type<float>();
                            while (u_right < m_dim_u_up-1)
                            {
                                u_right += 1;
                                if (mask_up[s].at<uchar>(v_up, u_right) > 0)
                                {
                                    d_right = depth_map_up[s].at<float>(v_up, u_right);
                                    break;
                                }
                            }
                            //~ if (!is_nan_type<float>(d_left) && !is_nan_type<float>(d_right))
                            //~ {
                                //~ dmin_map[s].at<float>(v, u) = std::min(d_left, d_right);
                                //~ dmax_map[s].at<float>(v, u) = std::max(d_left, d_right);//candidate_ds[mid_idx+1];
                            //~ }
                            if (!is_nan_type<float>(d_left) && !is_nan_type<float>(d_right))
                            {
                                candidate_ds.push_back(d_left);
                                candidate_ds.push_back(d_right);
                            }
                            
                            // 2nd line
                            if (v_up+1 < m_dim_v_up)
                            {
                                // Get the upscaled u, v
                                v_up += 1;
                                int u_up = std::min(2 * u, m_dim_u_up - 1);
                                // Get the point at the left
                                int u_left = u_up;
                                float d_left = nan_type<float>();
                                while (u_left > 1)
                                {
                                    u_left -= 1;
                                    if (mask_up[s].at<uchar>(v_up, u_left) > 0)
                                    {
                                        d_left = depth_map_up[s].at<float>(v_up, u_left);
                                        break;
                                    }
                                }
                                int u_right = u_up;
                                float d_right = nan_type<float>();
                                while (u_right < m_dim_u_up-1)
                                {
                                    u_right += 1;
                                    if (mask_up[s].at<uchar>(v_up, u_right) > 0)
                                    {
                                        d_right = depth_map_up[s].at<float>(v_up, u_right);
                                        break;
                                    }
                                }
                                if (!is_nan_type<float>(d_left) && !is_nan_type<float>(d_right))
                                {
                                    candidate_ds.push_back(d_left);
                                    candidate_ds.push_back(d_right);
                                }
                            }
                            if (candidate_ds.size() > 1)
                            {
                                std::sort(candidate_ds.begin(), candidate_ds.end());
                                int nb_candidates = candidate_ds.size();
                                //int mid_idx = (int)std::floor((nb_candidates-1.0)/2);
                                // Modify dmin and dmax
                                dmin_map[s].at<float>(v, u) = candidate_ds[0];//candidate_ds[mid_idx];
                                dmax_map[s].at<float>(v, u) = candidate_ds.back();//candidate_ds[mid_idx+1];
                            }
                        }
                    }
                }
                
                //~ std::cout << dmax_map[50] << std::endl;
            }
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
    void FineToCoarse<DataType>::get_coloured_depth_maps(Vec<Mat>& out_plot_depth_s_v_u_, int cv_colormap)
    {
        std::cout << "Plot depth results..." << std::endl;
        
        Vec<Mat> out_map_s_v_u_;
        Vec<Mat> out_validity_s_v_u_;
        
        get_results
        (
            out_map_s_v_u_, 
            out_validity_s_v_u_
        );
        
        int m_dim_s_ = out_map_s_v_u_.size();
        int m_dim_v_ = out_map_s_v_u_[0].rows;
        int m_dim_u_ = out_map_s_v_u_[0].cols;
        
        ImageConverter_uchar image_converter;
        image_converter.fit(out_map_s_v_u_[(int)std::round(m_dim_s_/2.0)]);
    
        for (int s=0; s<m_dim_s_; s++)
        {
            
            Mat disparity_map;
            image_converter.copy_and_scale(out_map_s_v_u_[s], disparity_map);
            cv::applyColorMap(disparity_map, disparity_map, cv_colormap);
            
            int m_dim_v_ = disparity_map.rows;
            int m_dim_u_ = disparity_map.cols;
            
            // Threshold scores
            disparity_map.setTo(0.0, out_validity_s_v_u_[s] == 0);
            
            // Get image norm view
            Mat im_norm = Mat(m_dim_v_, m_dim_u_, CV_32FC1);
            for (int v=0; v<m_dim_v_; v++)
            {
                for (int u=0; u<m_dim_u_; u++)
                {
                    Mat tmp = m_computers_[0]->get_epis()[v];
                    im_norm.at<float>(v, u) = norm<DataType>(tmp.at<DataType>(s, u));
                }
            }
            disparity_map.setTo(0.0, im_norm < _SHADOW_NORMALIZED_LEVEL);
            
            out_plot_depth_s_v_u_.push_back(disparity_map);
        }
    
    }
    
    template<typename DataType>
    void FineToCoarse<DataType>::get_coloured_epi_pyr(Vec<Mat>& out_plot_epi_pyr_p_s_u_, int v, int cv_colormap) {
        
        int m_pyr_size_ = m_computers_.size();
        int m_dim_v_orig_ = m_computers_[0]->get_depths_s_v_u_()[0].rows;
        
        if (v == -1)
            v = (int)std::round(m_dim_v_orig_/2.0);
        
        ImageConverter_uchar image_converter;
        
        out_plot_epi_pyr_p_s_u_.clear();
        for (int p=0; p<m_pyr_size_; p++)
        {
            int m_dim_s_ = m_computers_[p]->get_depths_s_v_u_().size();
            int m_dim_v_ = m_computers_[p]->get_depths_s_v_u_()[0].rows;
            int m_dim_u_ = m_computers_[p]->get_depths_s_v_u_()[0].cols;
            
            Mat tmp(m_dim_s_, m_dim_u_, CV_32FC1);
            
            int v_scaled = (int)std::round(1.0 * v * m_dim_v_ / m_dim_v_orig_);
            
            Vec<Mat> masks = m_computers_[p]->get_valid_depths_mask_s_v_u_();
            
            for (int s=0; s<m_dim_s_; s++)
            {
                m_computers_[p]->get_depths_s_v_u_()[s].row(v_scaled).copyTo(tmp.row(s));
                tmp.row(s).setTo(0.0, masks[s].row(v_scaled) == 0);
            }
            
            if (p == 0)
                image_converter.fit(tmp);
            
            image_converter.copy_and_scale(tmp, tmp);
            cv::applyColorMap(tmp, tmp, cv_colormap);
            
            // Get epi norm view
            Mat epi_norm = Mat(m_dim_s_, m_dim_u_, CV_32FC1);
            for (int s=0; s<m_dim_s_; s++)
            {
                for (int u=0; u<m_dim_u_; u++)
                {
                    Mat tmp = m_computers_[0]->get_epis()[v];
                    epi_norm.at<float>(s, u) = norm<DataType>(tmp.at<DataType>(s, u));
                }
            }
            tmp.setTo(0.0, epi_norm < _SHADOW_NORMALIZED_LEVEL);
            
            out_plot_epi_pyr_p_s_u_.push_back(tmp);
        }
        
    }
    
    
    template<typename DataType>
    void FineToCoarse<DataType>::get_coloured_depth_pyr(Vec<Mat>& out_plot_depth_pyr_p_v_u_, int s, int cv_colormap) {
        
        int m_pyr_size_ = m_computers_.size();
        int m_dim_s_ = m_computers_[0]->get_depths_s_v_u_().size();
        
        if (s == -1)
            s = (int)std::round(m_dim_s_/2.0);
        
        ImageConverter_uchar image_converter;
        
        out_plot_depth_pyr_p_v_u_.clear();
        for (int p=0; p<m_pyr_size_; p++)
        {
            Mat tmp = m_computers_[p]->get_depths_s_v_u_()[s].clone();
            
            if (p == 0)
                image_converter.fit(tmp);
            
            image_converter.copy_and_scale(tmp, tmp);
            cv::applyColorMap(tmp, tmp, cv_colormap);
            
            Vec<Mat> masks = m_computers_[p]->get_valid_depths_mask_s_v_u_();
            
            tmp.setTo(0.0, masks[s] == 0);
            
            out_plot_depth_pyr_p_v_u_.push_back(tmp);
        }
    }
}

#endif
