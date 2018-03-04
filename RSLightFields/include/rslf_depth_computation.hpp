#ifndef _RSLF_DEPTH_COMPUTATION
#define _RSLF_DEPTH_COMPUTATION

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rslf_plot.hpp>
#include <rslf_utils.hpp>


//~ #define _RSLF_DEPTH_COMPUTATION_DEBUG

#define _MEAN_SHIFT_MAX_ITER 10
#define _MEDIAN_FILTER_SIZE 5
#define _SCORE_THRESHOLD 0.01


namespace rslf
{
    
    /*
     * Aliases
     */
    using Mat = cv::Mat;
    
    template<typename T>
    using Vec = std::vector<T>;
    
    /*
     * *****************************************************************
     * INTERPOLATION FUNCTIONS
     * *****************************************************************
     */
    
    template<typename DataType>
    DataType nan_type();
    
    template<typename DataType>
    bool is_nan_type(DataType x);
    
    /**
     * Template generic interpolation function.
     * 
     * @param line_matrix A matrix with the values taken at points 0..max_index-1
     * @param index The (float) index at which to compute the value.
     */
    template<typename DataType>
    using Interpolation1D = DataType (*)
    (
        Mat line_matrix, 
        float index
    );
    
    /**
     * Template nearest neighbour interpolation function.
     *
     * @param line_matrix A matrix with the values taken at points 0..max_index-1
     * @param index The (float) index at which to compute the value.
     */ 
    template<typename DataType> 
    DataType interpolation_1d_nearest_neighbour
    (
        Mat line_matrix, 
        float index
    );
    
    /**
     * Template linear interpolation function.
     *
     * @param line_matrix A matrix with the values taken at points 0..max_index-1
     * @param index The (float) index at which to compute the value.
     */ 
    template<typename DataType> 
    DataType interpolation_1d_linear
    (
        Mat line_matrix, 
        float index
    );
    
    /*
     * *****************************************************************
     * KERNEL CLASSES
     * *****************************************************************
     */
    
    template<typename DataType>
    class KernelClass
    {
        public:
            virtual float evaluate(DataType x) {}
    };
    
    template<typename DataType>
    class BandwidthKernel: public KernelClass<DataType>
    {
        public:
            BandwidthKernel(float h): m_h_(h) {}
            float evaluate(DataType x);
        private:
            float m_h_;
    };
    
    /*
     * *****************************************************************
     * DepthComputer1D
     * *****************************************************************
     */

    /**
     * Template class with depth computation using 1d slices of the EPI.
     */
    template<typename DataType>
    class DepthComputer1D
    {
    public:
        DepthComputer1D
        (
            Mat epi, 
            Vec<float> d_list,
            int s_hat = -1, // default s_hat will be s_max / 2
            Interpolation1D<DataType> interpolation_function = (Interpolation1D<DataType>)interpolation_1d_linear<DataType>,
            KernelClass<DataType>* kernel_class = 0 
        );
        ~DepthComputer1D();
        
        void run();
        Mat get_coloured_epi(int cv_colormap = cv::COLORMAP_JET);
    
    private:
        Mat m_epi_;
        Vec<float> m_d_list_;
        
        Vec<Mat> m_radiances_u_s_d_;
        Mat m_scores_u_d_;
        
        Vec<float> m_best_depth_u_;
        Vec<float> m_score_depth_u_;

        Interpolation1D<DataType> m_interpolation_function_;
        
        /**
         * Line on which to compute the depth
         */
        int m_s_hat_;
        
        /**
         * Dimension along s axis
         */
        int m_dim_s_;
        /**
         * Dimension along d axis
         */
        int m_dim_d_;
        /**
         * Dimension along u axis
         */
        int m_dim_u_;
        
        KernelClass<DataType>* m_kernel_class_;
        bool m_delete_kernel_on_delete_;
    };
    
    
    /*
     * *****************************************************************
     * IMPLEMENTATION
     * *****************************************************************
     */
    template<typename DataType>
    DataType interpolation_1d_nearest_neighbour
    (
        Mat line_matrix, 
        float index
    )
    {
        int rounded_index = (int)std::round(index);
        if (rounded_index < 0 || rounded_index > line_matrix.cols - 1)
            return nan_type<DataType>();
        return line_matrix.at<DataType>(0, rounded_index);
    }

    template<typename DataType>
    DataType interpolation_1d_linear
    (
        Mat line_matrix, 
        float index
    )
    {
        int rounded_index_inf = (int)std::floor(index);
        int rounded_index_sup = (int)std::ceil(index);
        
        if (rounded_index_sup < 0 || rounded_index_inf > line_matrix.cols - 1)
            return nan_type<DataType>();
        if (rounded_index_sup == 0)
            return line_matrix.at<DataType>(0, 0);
        if (rounded_index_inf == line_matrix.cols - 1)
            return line_matrix.at<DataType>(0, line_matrix.cols - 1);
        
        // Linear interpolation
        float t = index - rounded_index_inf;
        return line_matrix.at<DataType>(0, rounded_index_inf) * (1 - t) + line_matrix.at<DataType>(0, rounded_index_sup) * t;
    }
    
    template<typename DataType>
    DepthComputer1D<DataType>::DepthComputer1D
    (
        Mat epi, 
        Vec<float> d_list,
        int s_hat,
        Interpolation1D<DataType> interpolation_function,
        KernelClass<DataType>* kernel_class
    ) 
    {
        // EPI and d's
        m_epi_ = epi;
        m_d_list_ = d_list;
        
        // Interpolation function
        m_interpolation_function_ = interpolation_function;
        
        // Dimensions
        m_dim_s_ = m_epi_.rows;
        m_dim_d_ = m_d_list_.size();
        m_dim_u_ = m_epi_.cols;
        
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
        
        // Radiance vector of matrices
        m_radiances_u_s_d_ = Vec<Mat>(m_dim_u_);
        for (int u=0; u<m_dim_u_; u++)
        {
            m_radiances_u_s_d_[u] = Mat(m_dim_s_, m_dim_d_, m_epi_.type());
        }
#ifdef _RSLF_DEPTH_COMPUTATION_DEBUG
        std::cout << "created m_radiances_u_s_d_ of size " << m_radiances_u_s_d_.size() << " x " << m_radiances_u_s_d_.at(0).size << std::endl;
#endif
        
        // Scores and best scores & depths
        m_scores_u_d_ = Mat(m_dim_u_, m_dim_d_, CV_32FC1);
        m_best_depth_u_ = Vec<float>(m_dim_u_);
        m_score_depth_u_ = Vec<float>(m_dim_u_);
        
        // Kernels
        if (kernel_class == 0)
        {
            m_kernel_class_ = new BandwidthKernel<DataType>(0.2); // Defaults to a bandwidth kernel with h=0.2
            m_delete_kernel_on_delete_ = true;
        }
        else
        {
            m_kernel_class_ = kernel_class;
            m_delete_kernel_on_delete_ = false;
        }
    }
    
    template<typename DataType>
    DepthComputer1D<DataType>::~DepthComputer1D() {
        if (m_delete_kernel_on_delete_)
            delete m_kernel_class_;
    }

    template<typename DataType>
    void DepthComputer1D<DataType>::run() 
    {
        /*
         * Build a matrix with indices corresponding to the lines of slope d and root s_hat
         */
        
        // Row matrix
        Mat D = Mat(1, m_dim_d_, CV_32FC1);
        for (int d=0; d<m_dim_d_; d++) 
        {
            D.at<float>(0, d) = m_d_list_[d];
        }
        
        // Col matrix
        Mat S = Mat(m_dim_s_, 1, CV_32FC1);
        for (int s=0; s<m_dim_s_; s++)
        {
            S.at<float>(s, 0) = m_s_hat_ - s;
        }
        
        // Index matrix
        Mat indices = S * D;
        
        /*
         * Iterate over all columns of the EPI
         */
#pragma omp parallel for
        for (int u=0; u<m_dim_u_; u++)
        {
#ifdef _RSLF_DEPTH_COMPUTATION_DEBUG
            std::cout << "u=" << u << std::endl;
#endif
            
            // Matrix of indices corresponding to the lines of disparities d
            Mat I = indices + u;
            
            // Create new radiance view
            // https://stackoverflow.com/questions/23998422/how-to-access-slices-of-a-3d-matrix-in-opencv
            Mat radiances_s_d = m_radiances_u_s_d_[u];
#ifdef _RSLF_DEPTH_COMPUTATION_DEBUG
            std::cout << radiances_s_d.size << std::endl;
#endif
            
            /*
             * Fill radiances
             */
            for (int s=0; s<m_dim_s_; s++)
            {
                for (int d=0; d<m_dim_d_; d++)
                {
                    radiances_s_d.at<DataType>(s, d) = m_interpolation_function_(m_epi_.row(s), I.at<float>(s, d));
                }
            }
            
            // Compute number of non-nan radiances per column
            Vec<float> card_R(m_dim_d_);

            for (int d=0; d<m_dim_d_; d++)
            {
                for (int s=0; s<m_dim_s_; s++)
                {
                    if (!is_nan_type<DataType>(radiances_s_d.at<DataType>(s, d)))
                    {
                        card_R[d] += 1.0;
                    }
                }
            }
            
            /*
             * Compute r_bar iteratively
             */
            
            // Initialize r_bar to the values in s_hat
            Mat r_bar;
            radiances_s_d.row(m_s_hat_).copyTo(r_bar);
            
            // Perform a partial mean shift
            for (int i=0; i<_MEAN_SHIFT_MAX_ITER; i++)
            {
                for (int d=0; d<m_dim_d_; d++)
                {
                    DataType numerator = 0.0;
                    float denominator = 0.0;
                    for (int s=0; s<m_dim_s_; s++)
                    {
                        if (!is_nan_type<DataType>(radiances_s_d.at<DataType>(s, d)))
                        {
                            // Compute K(r - r_bar)
                            float kernel_r_m_r_bar = m_kernel_class_->evaluate(radiances_s_d.at<DataType>(s, d) - r_bar.at<DataType>(0, d));
                            // sum(r * K(r - r_bar))
                            numerator += radiances_s_d.at<DataType>(s, d) * kernel_r_m_r_bar;
                            // sum(K(r - r_bar))
                            denominator += kernel_r_m_r_bar;
                        }
                    }
                    if (denominator != 0.0)
                        r_bar.at<DataType>(0, d) = numerator / denominator;
                }
            }

            /*
             * Compute scores 
             */
            for (int d=0; d<m_dim_d_; d++)
            {
                if (card_R[d] > 0)//== m_dim_s_)// 
                {
                    // m_scores_u_d_
                    for (int s=0; s<m_dim_s_; s++)
                    {
                        m_scores_u_d_.at<float>(u, d) += m_kernel_class_->evaluate(radiances_s_d.at<DataType>(s, d) - r_bar.at<DataType>(0, d));
                    }
                    m_scores_u_d_.at<float>(u, d) /= card_R[d];
                }
                else
                {
                    m_scores_u_d_.at<float>(u, d) = 0.0;
                }
            }
            
            /*
             * Get best d (max score)
             */
            double minVal;
            double maxVal;
            cv::Point minIdx;
            cv::Point maxIdx;
            cv::minMaxLoc(m_scores_u_d_.row(u).clone(), &minVal, &maxVal, &minIdx, &maxIdx);
#ifdef _RSLF_DEPTH_COMPUTATION_DEBUG
            std::cout << maxVal << ", " << maxIdx << std::endl;
#endif
            
            // TODO score threshold?
            // Test : 0.02
            m_best_depth_u_[u] = (maxVal > _SCORE_THRESHOLD ? m_d_list_.at(maxIdx.x) : 0.0);
            m_score_depth_u_[u] = maxVal;
            
        }
        
        /*
         * Apply a median filter on the resulting depths
         */
        // Convert to Mat in order to apply the builtin OpenCV function
        Mat best_depth_mat(1, m_dim_u_, CV_32FC1, &m_best_depth_u_.front()); 
        cv::medianBlur(best_depth_mat.clone(), best_depth_mat, _MEDIAN_FILTER_SIZE);
        // Fill back the Vec
        const float* p = best_depth_mat.ptr<float>(0);
        m_best_depth_u_ = Vec<float>(p, p + m_dim_u_);
        
#ifdef _RSLF_DEPTH_COMPUTATION_DEBUG
        // Inspect a slice
        Mat slice = m_radiances_u_s_d_.at(1).col(0);
        std::cout << slice << std::endl;
        std::cout << slice.size << std::endl;
        std::cout << rslf::type2str(slice.type()) << std::endl;
#endif
    }
    
    template<typename DataType>
    Mat DepthComputer1D<DataType>::get_coloured_epi(int cv_colormap) {
        
        // Build a matrix of occlusions: each element is the max observed depth
        Mat occlusion_map(m_dim_s_, m_dim_u_, CV_32FC1, -std::numeric_limits<float>::infinity());
        
        // Build a correspondance depth->color: scale to uchar and map to 3-channel matrix
        Mat coloured_depth = rslf::copy_and_scale_uchar(Mat(1, m_dim_u_,CV_32FC1, &m_best_depth_u_.front()));
        cv::applyColorMap(coloured_depth.clone(), coloured_depth, cv_colormap);
        
        // Construct an EPI with overlay
        Mat coloured_epi(m_epi_.rows, m_epi_.cols, CV_8UC3);
        
        // For each column of the s_hat row, draw the line, taking overlays into account
        for (int u=0; u<m_dim_u_; u++)
        {
            float current_depth_value = m_best_depth_u_[u];
            for (int s=0; s<m_dim_s_; s++)
            {
                int requested_index = u + (int)std::round(m_best_depth_u_[u] * (m_s_hat_ - s));
                if 
                (
                    requested_index > 0 && 
                    requested_index < m_dim_u_ && 
                    occlusion_map.at<float>(s, requested_index) < current_depth_value // only draw if the current depth is higher
                )
                {
                    coloured_epi.at<cv::Vec3b>(s, requested_index) = coloured_depth.at<cv::Vec3b>(0, u);
                    occlusion_map.at<float>(s, requested_index) = current_depth_value;
                }
            }
        }
        
        return coloured_epi;
    }

}



#endif
