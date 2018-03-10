#ifndef _RSLF_DEPTH_COMPUTATION
#define _RSLF_DEPTH_COMPUTATION

#include <rslf_plot.hpp>
#include <rslf_depth_computation_core.hpp>


namespace rslf
{
    /*
     * *****************************************************************
     * Depth1DComputer
     * *****************************************************************
     */
    
    /**
     * Template class with depth computation using 1d slices of the EPI.
     */
    template<typename DataType>
    class Depth1DComputer
    {
    public:
        Depth1DComputer
        (
            const Mat& epi, 
            const Vec<float>& d_list,
            int s_hat = -1, // default s_hat will be s_max / 2,
            float epi_scale_factor = -1,
            const Depth1DParameters<DataType>& parameters = Depth1DParameters<DataType>::get_default()
        );
        
        void run();
        Mat get_coloured_epi(int cv_colormap = cv::COLORMAP_JET);
    
    private:
        Mat m_epi_;
        const Vec<float>& m_d_list_;
        
        Mat m_edge_confidence_u_;
        Mat m_disp_confidence_u_;
        
        Mat m_scores_u_d_;
        
        Mat m_best_depth_u_;
        Mat m_score_depth_u_;
        
        /**
         * Line on which to compute the depth
         */
        int m_s_hat_;
        
        const Depth1DParameters<DataType>& m_parameters_;
    };
    
    
    /*
     * *****************************************************************
     * Depth1DComputer_pile
     * *****************************************************************
     */

    /**
     * Template class with depth computation using 1d slices of a pile of EPIs.
     */
    template<typename DataType>
    class Depth1DComputer_pile
    {
    public:
        Depth1DComputer_pile
        (
            const Vec<Mat>& epis, 
            const Vec<float>& d_list,
            int s_hat = -1, // default s_hat will be s_max / 2,
            float epi_scale_factor = -1,
            const Depth1DParameters<DataType>& parameters = Depth1DParameters<DataType>::get_default()
      );
        
        void run();
        Mat get_coloured_epi(int v = -1, int cv_colormap = cv::COLORMAP_JET);
        Mat get_disparity_map(int cv_colormap = cv::COLORMAP_JET);
        int get_s_hat() { return m_s_hat_; }
    
    private:
        Vec<Mat> m_epis_;
        const Vec<float>& m_d_list_;
        
        Mat m_edge_confidence_v_u_;
        Mat m_disp_confidence_v_u_;
        
        Vec<Mat> m_scores_v_u_d_;
        
        Mat m_best_depth_v_u_;
        Mat m_score_depth_v_u_;
        
        /**
         * Line on which to compute the depth
         */
        int m_s_hat_;
        
        const Depth1DParameters<DataType>& m_parameters_;
    };
    
    
    
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
    
    
    
    /*
     * *****************************************************************
     * IMPLEMENTATION
     * Depth1DComputer
     * *****************************************************************
     */
    
    template<typename DataType>
    Depth1DComputer<DataType>::Depth1DComputer
    (
        const Mat& epi, 
        const Vec<float>& d_list,
        int s_hat,
        float epi_scale_factor,
        const Depth1DParameters<DataType>& parameters
    ) : 
        m_d_list_(d_list),
        m_parameters_(parameters)
    {
        // If the input epi is a uchar, scale uchar to 1.0
        if (epi.depth() == CV_8U)
        {
            epi.convertTo(m_epi_, CV_32F, 1.0/255.0);
        }
        else
        {
            // If provided scale factor is invalid, scale from max in all channels
            if (epi_scale_factor < 0)
            {
                Vec<Mat> channels;
                cv::split(epi, channels);
                for (int c=0; c<channels.size(); c++) 
                {
                    double min, max;
                    cv::minMaxLoc(epi, &min, &max);
                    epi_scale_factor = std::max((float)max, epi_scale_factor);
                }
            }
            epi.convertTo(m_epi_, CV_32F, 1.0/epi_scale_factor);
        }
        
        
        // Dimensions
        int m_dim_s_ = m_epi_.rows;
        int m_dim_d_ = m_d_list_.size();
        int m_dim_u_ = m_epi_.cols;
        
        // s_hat
        if (s_hat < 0 || s_hat > m_dim_s_ - 1) 
        {
            // Default: s_hat is the center horizontal line index of the epi
            m_s_hat_ = (int) std::floor((0.0 + m_dim_s_) / 2);
        }
        else
        {
            m_s_hat_ = s_hat;
        }
        
        // Edge confidence
        m_edge_confidence_u_ = cv::Mat::zeros(1, m_epi_.cols, CV_32FC1);
        
        // Disparity confidence
        m_disp_confidence_u_ = cv::Mat::zeros(1, m_epi_.cols, CV_32FC1);
        
        // Scores and best scores & depths
        m_scores_u_d_ = cv::Mat::zeros(m_dim_u_, m_dim_d_, CV_32FC1);
        m_best_depth_u_ = cv::Mat::zeros(1, m_dim_u_, CV_32FC1);
        m_score_depth_u_ = cv::Mat::zeros(1, m_dim_u_, CV_32FC1);
        
    }

    template<typename DataType>
    void Depth1DComputer<DataType>::run() 
    {
        // Dimension
        int m_dim_s_ = m_epi_.rows;
        int m_dim_d_ = m_d_list_.size();
        
        std::cout << "Max num of threads: " << omp_get_max_threads() << std::endl;
        //~ omp_set_nested(1);
        
        /*
         * Build a matrix with indices corresponding to the lines of slope d and root s_hat
         */
        
        // Row matrix
        Vec<float> m_d_list_copy = m_d_list_;
        Mat D = Mat(1, m_dim_d_, CV_32FC1, &m_d_list_copy.front());
        
        // Col matrix
        Mat S = Mat(m_dim_s_, 1, CV_32FC1);
        for (int s=0; s<m_dim_s_; s++)
        {
            S.at<float>(s) = m_s_hat_ - s;
        }
        
        // Index matrix
        Mat indices = S * D;
        
        // Buffer
        BufferDepth1D buffer;
        
        compute_1D_depth_epi(
            m_epi_,
            m_d_list_,
            indices,
            m_s_hat_,
            m_edge_confidence_u_,
            m_disp_confidence_u_,
            m_scores_u_d_,
            m_best_depth_u_,
            m_score_depth_u_,
            m_parameters_,
            buffer
        );
    }
    
    template<typename DataType>
    Mat Depth1DComputer<DataType>::get_coloured_epi(int cv_colormap) {
        
        // Dimensions
        int m_dim_s_ = m_epi_.rows;
        int m_dim_u_ = m_epi_.cols;
        
        // Build a matrix of occlusions: each element is the max observed depth
        Mat occlusion_map(m_dim_s_, m_dim_u_, CV_32FC1, -std::numeric_limits<float>::infinity());
        
        // Build a correspondance depth->color: scale to uchar and map to 3-channel matrix
        Mat coloured_depth = rslf::copy_and_scale_uchar(m_best_depth_u_);
        cv::applyColorMap(coloured_depth.clone(), coloured_depth, cv_colormap);
        
        // Construct an EPI with overlay
        Mat coloured_epi = cv::Mat::zeros(m_epi_.rows, m_epi_.cols, CV_8UC3);
        
        // For each column of the s_hat row, draw the line, taking overlays into account
        for (int u=0; u<m_dim_u_; u++)
        {
            // Only paint if the confidence threshold was high enough
            if (m_edge_confidence_u_.at<float>(u) > m_parameters_.m_score_threshold_)
            {
                float current_depth_value = m_best_depth_u_.at<float>(u);
                for (int s=0; s<m_dim_s_; s++)
                {
                
                    int requested_index = u + (int)std::round(m_best_depth_u_.at<float>(u) * (m_s_hat_ - s));
                    if 
                    (
                        requested_index > 0 && 
                        requested_index < m_dim_u_ && 
                        occlusion_map.at<float>(s, requested_index) < current_depth_value // only draw if the current depth is higher
                    )
                    {
                        coloured_epi.at<cv::Vec3b>(s, requested_index) = coloured_depth.at<cv::Vec3b>(u);
                        occlusion_map.at<float>(s, requested_index) = current_depth_value;
                    }
                }
            }
        }
        
        return coloured_epi;
    }

    /*
     * *****************************************************************
     * IMPLEMENTATION
     * Depth1DComputer_pile
     * *****************************************************************
     */
    
    template<typename DataType>
    Depth1DComputer_pile<DataType>::Depth1DComputer_pile
    (
        const Vec<Mat>& epis, 
        const Vec<float>& d_list,
        int s_hat,
        float epi_scale_factor,
        const Depth1DParameters<DataType>& parameters
    ) : 
        m_d_list_(d_list),
        m_parameters_(parameters)
    {
        m_epis_ = Vec<Mat>(epis.size(), Mat(epis[0].rows, epis[0].cols, epis[0].type()));
    
        // Look for the max and min values across all epis
        // If provided scale factor is invalid, scale from max in all channels
        if (epis[0].depth() != CV_8U && epi_scale_factor < 0)
        {
#pragma omp parallel for
            for (int v=0; v<epis.size(); v++)
            {
                Mat epi = epis[v];
                Vec<Mat> channels;
                cv::split(epi, channels);
                for (int c=0; c<channels.size(); c++) 
                {
                    double min, max;
                    cv::minMaxLoc(epi, &min, &max);
#pragma omp critical
{
                    epi_scale_factor = std::max((float)max, epi_scale_factor);
}
                }
            }
        }
        
        // If the input epi is a uchar, scale uchar to 1.0
#pragma omp parallel for
        for (int v=0; v<epis.size(); v++) 
        {
            Mat epi = epis[v];
            Mat epi2;
            if (epi.depth() == CV_8U)
            {
                epi.convertTo(epi2, CV_32F, 1.0/255.0);
            }
            else
            {
                epi.convertTo(epi2, CV_32F, 1.0/epi_scale_factor);
            }
            m_epis_[v] = epi2;
        }
        
        // Dimensions
        int m_dim_s_ = m_epis_[0].rows;
        int m_dim_d_ = m_d_list_.size();
        int m_dim_u_ = m_epis_[0].cols;
        int m_dim_v_ = m_epis_.size();
        
        // s_hat
        if (s_hat < 0 || s_hat > m_dim_s_ - 1) 
        {
            // Default: s_hat is the center horizontal line index of the epi
            m_s_hat_ = (int) std::floor((0.0 + m_dim_s_) / 2);
        }
        else
        {
            m_s_hat_ = s_hat;
        }
        
        // Edge confidence
        m_edge_confidence_v_u_ = Mat(m_dim_v_, m_dim_u_, CV_32FC1);
        
        // Disparity confidence
        m_disp_confidence_v_u_ = Mat(m_dim_v_, m_dim_u_, CV_32FC1);
        
        // Scores and best scores & depths
        m_scores_v_u_d_ = Vec<Mat>(m_dim_v_, Mat(m_dim_u_, m_dim_d_, CV_32FC1));
#pragma omp parallel for
        for (int v=0; v<m_dim_v_; v++)
        {
            m_scores_v_u_d_[v] = Mat(m_dim_u_, m_dim_d_, CV_32FC1, cv::Scalar(0.0));
        }
        m_best_depth_v_u_ = cv::Mat::zeros(m_dim_v_, m_dim_u_, CV_32FC1);
        m_score_depth_v_u_ = cv::Mat::zeros(m_dim_v_, m_dim_u_, CV_32FC1);
    }

    template<typename DataType>
    void Depth1DComputer_pile<DataType>::run() 
    {
        
        // Dimension
        int m_dim_s_ = m_epis_[0].rows;
        int m_dim_d_ = m_d_list_.size();
        int m_dim_v_ = m_epis_.size();
        int m_dim_u_ = m_epis_[0].cols;
        
        /*
         * Build a matrix with indices corresponding to the lines of slope d and root s_hat
         */
        
        // Row matrix
        Vec<float> m_d_list_copy = m_d_list_;
        Mat D = Mat(1, m_dim_d_, CV_32FC1, &m_d_list_copy.front());
        
        // Col matrix
        Mat S = Mat(m_dim_s_, 1, CV_32FC1);
        for (int s=0; s<m_dim_s_; s++)
        {
            S.at<float>(s) = m_s_hat_ - s;
        }
        
        // Index matrix
        Mat indices = S * D;
    
        int thr_max = omp_get_max_threads();
        std::cout << "Max num of threads: " << thr_max << std::endl;
        
        Vec<BufferDepth1D*> buffer;
        for (int t=0; t<thr_max; t++)
            buffer.push_back(new BufferDepth1D());
        
        // For progress bar
        float progress = 0.0;
        int barWidth = 40;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        
#pragma omp parallel for
        for (int v=0; v<m_dim_v_; v++)
        {
            // Create views
            Mat m_epi_ = m_epis_[v];
            Mat m_edge_confidence_u_ = m_edge_confidence_v_u_.row(v);
            Mat m_disp_confidence_u_ = m_disp_confidence_v_u_.row(v);
            Mat m_scores_u_d_ = m_scores_v_u_d_[v];
            Mat m_best_depth_u_ = m_best_depth_v_u_.row(v);
            Mat m_score_depth_u_ = m_score_depth_v_u_.row(v);
            
            compute_1D_depth_epi(
                m_epi_,
                m_d_list_,
                indices,
                m_s_hat_,
                m_edge_confidence_u_,
                m_disp_confidence_u_,
                m_scores_u_d_,
                m_best_depth_u_,
                m_score_depth_u_,
                m_parameters_,
                *buffer[omp_get_thread_num()]
            );
            
#pragma omp critical
{
            // Display progress bar
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            std::cout << "[";
            int pos = barWidth * progress / m_dim_v_;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0 / m_dim_v_) << "% \t" << std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count() << "s \r";
            std::cout.flush();
}
#pragma omp atomic
           progress += 1.0;
            
        }
    
        std::cout << std::endl;
        
        // Apply median filter
        std::cout << "Applying selective median fiter" << std::endl;
        Mat tmp;
        selective_median_filter<DataType>(
            m_best_depth_v_u_, 
            tmp,
            m_epis_,
            m_s_hat_,
            m_parameters_.m_median_filter_size_, 
            m_edge_confidence_v_u_, 
            m_parameters_.m_score_threshold_,
            m_parameters_.m_median_filter_epsilon_);

        m_best_depth_v_u_ = tmp;
        
        for (int t=0; t<thr_max; t++)
            delete buffer[t];
        
    }
    
    template<typename DataType>
    Mat Depth1DComputer_pile<DataType>::get_coloured_epi(int v, int cv_colormap) {
        
        // Dimensions
        int m_dim_s_ = m_epis_[0].rows;
        int m_dim_u_ = m_epis_[0].cols;
        int m_dim_v_ = m_epis_.size();
        
        if (v < 0)
            v = (int)std::floor(m_dim_v_/2.0);
        
        Mat m_epi_ = m_epis_[v];
        Mat m_best_depth_u_ = m_best_depth_v_u_.row(v);
        Mat m_edge_confidence_u_ = m_edge_confidence_v_u_.row(v);
        
        // Build a matrix of occlusions: each element is the max observed depth
        Mat occlusion_map(m_dim_s_, m_dim_u_, CV_32FC1, -std::numeric_limits<float>::infinity());
        
        // Build a correspondance depth->color: scale to uchar and map to 3-channel matrix
        Mat coloured_depth = rslf::copy_and_scale_uchar(m_best_depth_u_);
        cv::applyColorMap(coloured_depth.clone(), coloured_depth, cv_colormap);
        
        // Construct an EPI with overlay
        Mat coloured_epi = cv::Mat::zeros(m_epi_.rows, m_epi_.cols, CV_8UC3);
        
        // For each column of the s_hat row, draw the line, taking overlays into account
#pragma omp parallel for
        for (int u=0; u<m_dim_u_; u++)
        {
            // Only paint if the confidence threshold was high enough
            if (m_edge_confidence_u_.at<float>(u) > m_parameters_.m_score_threshold_)
            {
                float current_depth_value = m_best_depth_u_.at<float>(u);
                for (int s=0; s<m_dim_s_; s++)
                {
                
                    int requested_index = u + (int)std::round(m_best_depth_u_.at<float>(u) * (m_s_hat_ - s));
                    if 
                    (
                        requested_index > 0 && 
                        requested_index < m_dim_u_ && 
                        occlusion_map.at<float>(s, requested_index) < current_depth_value // only draw if the current depth is higher
                    )
                    {
                        coloured_epi.at<cv::Vec3b>(s, requested_index) = coloured_depth.at<cv::Vec3b>(u);
                        occlusion_map.at<float>(s, requested_index) = current_depth_value;
                    }
                }
            }
        }
        
        return coloured_epi;
    }
    
    template<typename DataType>
    Mat Depth1DComputer_pile<DataType>::get_disparity_map(int cv_colormap)
    {
        // Dimensions
        int m_dim_s_ = m_epis_[0].rows;
        int m_dim_u_ = m_epis_[0].cols;
        int m_dim_v_ = m_epis_.size();
        
        Mat disparity_map;
        
        disparity_map = rslf::copy_and_scale_uchar(m_best_depth_v_u_);
        cv::applyColorMap(disparity_map, disparity_map, cv_colormap);
        
        // Threshold scores
        Mat disparity_map_with_scores = cv::Mat::zeros(m_dim_v_, m_dim_u_, disparity_map.type());
        
        Mat score_mask = m_edge_confidence_v_u_ > m_parameters_.m_score_threshold_;
        
        cv::add(disparity_map, disparity_map_with_scores, disparity_map_with_scores, score_mask);
        
        return disparity_map_with_scores;
    }
}


#endif
