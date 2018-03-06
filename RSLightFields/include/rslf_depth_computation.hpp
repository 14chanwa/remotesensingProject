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
     * INTERPOLATION CLASSES
     * *****************************************************************
     */
    
    template<typename DataType>
    DataType nan_type();
    
    template<typename DataType>
    bool is_nan_type(DataType x);
    
    /**
     * Template generic 1D interpolation class.
     */
    template<typename DataType>
    class Interpolation1DClass
    {
        public:
            virtual DataType interpolate(Mat line_matrix, float index) = 0;
            virtual Mat interpolate_mat(Mat data_matrix, Mat indices) = 0;
    };

    /**
     * Template nearest neighbour interpolation class.
     */ 
    template<typename DataType> 
    class Interpolation1DNearestNeighbour : public Interpolation1DClass<DataType>
    {
        public:
            Interpolation1DNearestNeighbour() {}
            DataType interpolate(Mat line_matrix, float index);
            Mat interpolate_mat(Mat line_matrix, Mat indices);
    };
    
    /**
     * Template linear interpolation class.
     */ 
    template<typename DataType> 
    class Interpolation1DLinear : public Interpolation1DClass<DataType>
    {
        public:
            Interpolation1DLinear() {}
            DataType interpolate(Mat line_matrix, float index);
            Mat interpolate_mat(Mat data_matrix, Mat indices);
    };
    
    /*
     * *****************************************************************
     * KERNEL CLASSES
     * *****************************************************************
     */
    
    template<typename DataType>
    class KernelClass
    {
        public:
            virtual float evaluate(DataType x) = 0;
            virtual Mat evaluate_mat(Mat m) = 0;
    };
    
    /**
     * This kernel returns value:
     * 1 - norm(x/h)^2 if norm(x/h) < 1
     * 0 else
     */
    template<typename DataType>
    class BandwidthKernel: public KernelClass<DataType>
    {
        public:
            BandwidthKernel(float h): m_h_(h) {}
            float evaluate(DataType x);
            Mat evaluate_mat(Mat m);
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
            Interpolation1DClass<DataType>* interpolation_class = 0,// default is linear interp
            KernelClass<DataType>* kernel_class = 0, // default will be bandwidth kernel
            float score_threshold = _SCORE_THRESHOLD
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
        
        float m_score_threshold_;
        
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
        
        Interpolation1DClass<DataType>* m_interpolation_class_;
        KernelClass<DataType>* m_kernel_class_;
        bool m_delete_interpolation_on_delete_;
        bool m_delete_kernel_on_delete_;
    };
    
    
    /*
     * *****************************************************************
     * IMPLEMENTATION
     * Interpolation
     * *****************************************************************
     */
    
    template<typename DataType>
    DataType Interpolation1DNearestNeighbour<DataType>::interpolate
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
    Mat Interpolation1DNearestNeighbour<DataType>::interpolate_mat
    (
        Mat data_matrix, 
        Mat indices
    )
    {
        // TODO is there a better way to vectorize?
        assert(indices.rows == data_matrix.rows);
        Mat res(indices.rows, indices.cols, data_matrix.type(), cv::Scalar(0.0));
        
        // Round indices
        Mat round_indices_matrix;
        indices.convertTo(round_indices_matrix, CV_32SC1, 1.0, 0.0);
        // For each row
        for (int r=0; r<indices.rows; r++) {
            DataType* data_ptr = data_matrix.ptr<DataType>(r);
            DataType* res_ptr = res.ptr<DataType>(r);
            int* ind_ptr = round_indices_matrix.ptr<int>(r);
            // For each col
            for (int c=0; c<indices.cols; c++) {
                res_ptr[c] = (ind_ptr[c] > -1 && ind_ptr[c] < data_matrix.cols ? data_ptr[ind_ptr[c]] : nan_type<DataType>());
            }
        }
        return res;
    }

    template<typename DataType>
    DataType Interpolation1DLinear<DataType>::interpolate
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
    Mat Interpolation1DLinear<DataType>::interpolate_mat
    (
        Mat data_matrix, 
        Mat indices
    )
    {
        // TODO is there a better way to vectorize?
        assert(indices.rows == data_matrix.rows);
        Mat res(indices.rows, indices.cols, data_matrix.type(), cv::Scalar(0.0));
        
        // For each row
        for (int r=0; r<indices.rows; r++) {
            DataType* data_ptr = data_matrix.ptr<DataType>(r);
            DataType* res_ptr = res.ptr<DataType>(r);
            float* ind_ptr = indices.ptr<float>(r);
            // For each col
            for (int c=0; c<indices.cols; c++) {
                int ind_i = (int)std::floor(ind_ptr[c]);
                int ind_s = (int)std::ceil(ind_ptr[c]);
                float ind_residue = ind_ptr[c] - ind_i;
                res_ptr[c] = (ind_i <= 0 || ind_s >= data_matrix.cols - 1 ?
                    nan_type<DataType>() : (1-ind_residue)*data_ptr[ind_i] + ind_residue*data_ptr[ind_s]);
            }
        }
        return res;
    }
    
    /*
     * *****************************************************************
     * IMPLEMENTATION
     * DepthComputer1D
     * *****************************************************************
     */
    
    template<typename DataType>
    DepthComputer1D<DataType>::DepthComputer1D
    (
        Mat epi, 
        Vec<float> d_list,
        int s_hat,
        Interpolation1DClass<DataType>* interpolation_class,
        KernelClass<DataType>* kernel_class,
        float score_threshold
    ) : 
        m_score_threshold_(score_threshold),
        m_epi_(epi),
        m_d_list_(d_list)
    {
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
            m_radiances_u_s_d_[u] = Mat(m_dim_s_, m_dim_d_, m_epi_.type(), cv::Scalar(0.0));
        }
#ifdef _RSLF_DEPTH_COMPUTATION_DEBUG
        std::cout << "created m_radiances_u_s_d_ of size " << m_radiances_u_s_d_.size() << " x " << m_radiances_u_s_d_.at(0).size << std::endl;
#endif
        
        // Scores and best scores & depths
        m_scores_u_d_ = Mat(m_dim_u_, m_dim_d_, CV_32FC1);
        m_best_depth_u_ = Vec<float>(m_dim_u_);
        m_score_depth_u_ = Vec<float>(m_dim_u_);
        
        // Interpolation
        if (interpolation_class == 0)
        {
            m_interpolation_class_ = new Interpolation1DLinear<DataType>(); // Defaults to linear interpolation
            m_delete_interpolation_on_delete_ = true;
        }
        else
        {
            m_interpolation_class_ = interpolation_class;
            m_delete_interpolation_on_delete_ = false;
        }
        
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
        if (m_delete_interpolation_on_delete_)
            delete m_interpolation_class_;
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
            /*
             * Fill radiances
             */
            // Matrix of indices corresponding to the lines of disparities d
            Mat I = indices + u;
            
            // Create new radiance view
            // https://stackoverflow.com/questions/23998422/how-to-access-slices-of-a-3d-matrix-in-opencv
            Mat radiances_s_d = m_radiances_u_s_d_[u];
#ifdef _RSLF_DEPTH_COMPUTATION_DEBUG
            std::cout << radiances_s_d.size << std::endl;
#endif
            
            // Interpolate
            radiances_s_d = m_interpolation_class_->interpolate_mat(m_epi_, I);
            
            // Indicator of non-nan values
            Mat non_nan_indicator = radiances_s_d == radiances_s_d;

            // Indicator should be CV_8UC1
            if (non_nan_indicator.channels() > 1)
            {
                cv::extractChannel(non_nan_indicator, non_nan_indicator, 0);
            }

            non_nan_indicator.convertTo(non_nan_indicator, CV_32FC1, 1.0/255.0);
            Mat non_nan_invert_indicator = 1.0 - non_nan_indicator;
            
            // Compute number of non-nan radiances per column
            Mat card_R(1, m_dim_d_, CV_32FC1);
            for (int d=0; d<m_dim_d_; d++)
            {
                card_R.at<float>(0, d) = cv::countNonZero(non_nan_indicator.col(d));
            }
            
            /*
             * Compute r_bar iteratively
             */
            
            // Initialize r_bar to the values in s_hat
            Mat r_bar;
            radiances_s_d.row(m_s_hat_).copyTo(r_bar);
            
            // A col matrix with -1
            Mat row_m1 = Mat(radiances_s_d.rows, 1, CV_32FC1, cv::Scalar(-1.0));
            
            Mat r_m_r_bar;
            Mat k_r_m_r_bar_mat;
            Mat r_k_r_m_r_bar_mat;
            
            Mat sum_r_K_r_m_r_bar;
            Mat sum_K_r_m_r_bar;
            
            Mat r_bar_broadcast;
            
            Mat mask_null_denom;

            // Perform a partial mean shift to cmpute r_bar
            for (int i=0; i<_MEAN_SHIFT_MAX_ITER; i++)
            {
                // r_bar repeated over lines
                cv::repeat(r_bar, m_dim_s_, 1, r_bar_broadcast); 
                
                // r - r_bar
                cv::subtract(radiances_s_d, r_bar_broadcast, r_m_r_bar);
                
                // K(r - r_bar)
                k_r_m_r_bar_mat = m_kernel_class_->evaluate_mat(r_m_r_bar); // returns 0 if value is nan
                
                // r * K(r - r_bar)
                // Multiply should be of the same number of channels
                if (radiances_s_d.channels() > 1)
                {
                    Vec<Mat> channels;
                    for (int c=0; c<radiances_s_d.channels(); c++) {
                        channels.push_back(k_r_m_r_bar_mat);
                    }
                    cv::merge(channels, k_r_m_r_bar_mat);
                }
                cv::multiply(radiances_s_d, k_r_m_r_bar_mat, r_k_r_m_r_bar_mat);

                // Sum over lines
                cv::reduce(r_k_r_m_r_bar_mat, sum_r_K_r_m_r_bar, 0, cv::REDUCE_SUM);
                cv::reduce(k_r_m_r_bar_mat, sum_K_r_m_r_bar, 0, cv::REDUCE_SUM);
                
                // Avoir dividing by zero
                mask_null_denom = sum_K_r_m_r_bar < 1e-6;
                Mat mask_null_denom_vec = mask_null_denom.clone();
                // Indicator should be of type CV_8UC1
                if (mask_null_denom.channels() > 1)
                {
                    cv::extractChannel(mask_null_denom, mask_null_denom, 0);
                }
                sum_r_K_r_m_r_bar.setTo(cv::Scalar(0.0), mask_null_denom);
                sum_K_r_m_r_bar.setTo(cv::Scalar(1.0), mask_null_denom_vec);
                cv::divide(sum_r_K_r_m_r_bar, sum_K_r_m_r_bar, r_bar);

            }

            /*
             * Compute scores 
             */
            // Get the last sum { K(r - r_bar) }
            k_r_m_r_bar_mat = m_kernel_class_->evaluate_mat(r_m_r_bar);
            cv::reduce(k_r_m_r_bar_mat, sum_K_r_m_r_bar, 0, cv::REDUCE_SUM);
            cv::divide(sum_K_r_m_r_bar, card_R, sum_K_r_m_r_bar);
            // Get the position of the zeros of card_R and set corresponding values to 0
            sum_K_r_m_r_bar.setTo(cv::Scalar(0.0), card_R == 0);
            // Add the line where nonzero division was performed
            cv::add(m_scores_u_d_.row(u), sum_K_r_m_r_bar, m_scores_u_d_.row(u));
            
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
            
            // Score threshold
            m_best_depth_u_[u] = (maxVal > m_score_threshold_ ? m_d_list_.at(maxIdx.x) : 0.0);
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
        Mat coloured_depth = rslf::copy_and_scale_uchar(Mat(1, m_dim_u_, CV_32FC1, &m_best_depth_u_.front()));
        cv::applyColorMap(coloured_depth.clone(), coloured_depth, cv_colormap);
        
        // Construct an EPI with overlay
        Mat coloured_epi(m_epi_.rows, m_epi_.cols, CV_8UC3, cv::Scalar(0.0));
        
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
