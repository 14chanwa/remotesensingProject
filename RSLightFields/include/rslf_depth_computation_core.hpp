#ifndef _RSLF_DEPTH_COMPUTATION_CORE
#define _RSLF_DEPTH_COMPUTATION_CORE

#include <string>
#include <vector>
#include <chrono>
#include <iostream>

#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <rslf_types.hpp>
#include <rslf_interpolation.hpp>
#include <rslf_kernels.hpp>


//~ #define _RSLF_DEPTH_COMPUTATION_DEBUG

// Default parameters
#define _MEAN_SHIFT_MAX_ITER 10
#define _EDGE_CONFIDENCE_FILTER_SIZE 9
#define _MEDIAN_FILTER_SIZE 5
#define _MEDIAN_FILTER_EPSILON 0.1
#define _EDGE_SCORE_THRESHOLD 0.02
#define _DISP_SCORE_THRESHOLD 0.1
#define _PROPAGATION_EPSILON 0.1


// Useful links
// https://docs.opencv.org/3.4.1/
// https://docs.opencv.org/3.4.1/d3/d63/classcv_1_1Mat.html
// https://docs.opencv.org/3.4.1/d2/de8/group__core__array.html
// https://stackoverflow.com/questions/23998422/how-to-access-slices-of-a-3d-matrix-in-opencv
// https://stackoverflow.com/questions/2724708/is-it-a-good-practice-to-pass-struct-object-as-parameter-to-a-function-in-c


namespace rslf
{
    
    /*
     * *****************************************************************
     * Depth1DParameters
     * *****************************************************************
     */
    template<typename DataType>
    struct Depth1DParameters
    {
    private:
        static Depth1DParameters m_default_;
    public:
    
        Depth1DParameters() // initialize at default values
        {
            m_interpolation_class_ = new Interpolation1DLinear<DataType>();
            //~ m_interpolation_class_ = new Interpolation1DNearestNeighbour<DataType>();
            m_kernel_class_ = new BandwidthKernel<DataType>(0.2);
            m_edge_score_threshold_ = _EDGE_SCORE_THRESHOLD;
            m_disp_score_threshold_ = _DISP_SCORE_THRESHOLD;
            m_mean_shift_max_iter_ = _MEAN_SHIFT_MAX_ITER;
            m_edge_confidence_filter_size_ = _EDGE_CONFIDENCE_FILTER_SIZE;
            m_median_filter_size_ = _MEDIAN_FILTER_SIZE;
            m_median_filter_epsilon_ = _MEDIAN_FILTER_EPSILON;
            m_propagation_epsilon = _PROPAGATION_EPSILON;
            m_slope_factor = 1.0; // Useful when subsampling EPIs
        }

        Interpolation1DClass<DataType>* m_interpolation_class_;
        KernelClass<DataType>* m_kernel_class_;
        float m_edge_score_threshold_;
        float m_disp_score_threshold_;
        float m_mean_shift_max_iter_;
        int m_edge_confidence_filter_size_;
        int m_median_filter_size_;
        float m_median_filter_epsilon_;
        float m_propagation_epsilon;
        float m_slope_factor;
        
        ~Depth1DParameters() 
        {
            if (m_interpolation_class_ != NULL)
            {
                delete m_interpolation_class_;
                m_interpolation_class_ = NULL;
            }
            if (m_kernel_class_ != NULL)
            {
                delete m_kernel_class_;
                m_kernel_class_ = NULL;
            }
        }
        
        static Depth1DParameters& get_default() // get a static default instance
        {
            return m_default_;
        }
    };
    
    template<typename DataType>
    Depth1DParameters<DataType> Depth1DParameters<DataType>::m_default_ = Depth1DParameters();

    
    /*
     * *****************************************************************
     * compute_1D_depth_epi
     * *****************************************************************
     */
    template <typename DataType>
    struct BufferDepth1D {
        
        BufferDepth1D
        (
            int m_dim_s_,
            int m_dim_d_, 
            int m_dim_u_, 
            int data_type, 
            const Depth1DParameters<DataType>& m_parameters_
        )
        {
            filter_kernel = cv::Mat::zeros(1, m_parameters_.m_edge_confidence_filter_size_, CV_32FC1);
            m_scores_u_d_ = Mat(m_dim_u_, m_dim_d_, CV_32FC1, cv::Scalar(0.0));
            card_R = Mat(1, m_dim_d_, CV_32FC1);
            radiances_s_d = cv::Mat::zeros(m_dim_s_, m_dim_d_, data_type);
        }
        
        Mat thr_tmp;
        Vec<cv::Point> locations;
        
        Mat m_scores_u_d_;
        Mat filter_kernel;
        Mat conv_tmp;
        Mat sqsum_tmp;
        Mat I;
        Mat radiances_s_d;
        Mat radiances_s_d_un_nanified;
        Mat card_R;
        Mat r_bar;
        Mat r_m_r_bar;
        Mat K_r_m_r_bar_mat;
        Mat K_r_m_r_bar_mat_vec;
        Mat r_K_r_m_r_bar_mat;
        Mat sum_r_K_r_m_r_bar;
        Mat sum_K_r_m_r_bar;
        Mat sum_K_r_m_r_bar_vec;
        Mat r_bar_broadcast;
    };
    
    template<typename DataType>
    void compute_1D_edge_confidence(
        const Mat& m_epi_,
        int s,
        Mat& m_edge_confidence_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        BufferDepth1D<DataType>& m_buffer_
    );
    
    template<typename DataType>
    void compute_1D_depth_epi(
        const Mat& m_epi_,
        const Vec<float> m_d_list_,
        const Mat& m_indices_,
        int m_s_hat_,
        const Mat& m_edge_confidence_u_,
        Mat& m_disp_confidence_u_,
        Mat& m_best_depth_u_,
        Mat& m_rbar_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        BufferDepth1D<DataType>& m_buffer_,
        Mat& m_mask_u_
    );
    
    /*
     * *****************************************************************
     * compute_1D_depth_epi_pile
     * *****************************************************************
     */
    
    template<typename DataType>
    void compute_1D_edge_confidence_pile(
        const Vec<Mat>& m_epis_,
        int s,
        Mat& m_edge_confidence_v_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        Vec<BufferDepth1D<DataType>* >& m_buffers_
    );
    
    template<typename DataType>
    void compute_1D_depth_epi_pile(
        const Vec<Mat>& m_epis_,
        const Vec<float> m_d_list_,
        int m_s_hat_,
        const Mat& m_edge_confidence_v_u_,
        Mat& m_disp_confidence_v_u_,
        Mat& m_best_depth_v_u_,
        Mat& m_rbar_v_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        Vec<BufferDepth1D<DataType>* >& m_buffers_,
        Mat& m_mask_v_u_,
        bool m_verbose_ = true
    );
    
    
    /*
     * *****************************************************************
     * compute_2D_depth_epi
     * *****************************************************************
     */
    
    template<typename DataType>
    void compute_2D_edge_confidence(
        const Vec<Mat>& m_epis_,
        Vec<Mat>& m_edge_confidence_s_v_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        Vec<BufferDepth1D<DataType>* >& m_buffers_
    );
    
    template<typename DataType>
    void compute_2D_depth_epi(
        const Vec<Mat>& m_epis_,
        const Vec<float> m_d_list_,
        //~ const Mat& m_indices_, // u + (s_hat - s) * d not fixed anymore since s_hat varies... but what about u + s_hat*d - s*d (could add s_hat * d) at the s_hat loop ; then m_indices_ = - s * d
        const Vec<Mat>& m_edge_confidence_s_v_u_,
        Vec<Mat>& m_disp_confidence_s_v_u_,
        Vec<Mat>& m_best_depth_s_v_u_,
        Vec<Mat>& m_rbar_s_v_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        Vec<BufferDepth1D<DataType>* >& m_buffers_,
        bool m_verbose_ = true
    );
    
    
    
    /*
     * *****************************************************************
     * selective_median_filter
     * *****************************************************************
     */
    
    template<typename DataType>
    void selective_median_filter(
        const Mat& src,
        Mat& dst,
        const Vec<Mat>& epis,
        int s_hat_,
        int size,
        const Mat& edge_scores,
        float score_threshold,
        float epsilon
    );
    
    /*
     * Useful template functions
     */
    
    /**
     * Sums the squares of the values across channels of the input matrix
     * 
     * @param src Input
     * @param dst Output
     * @param buffer Buffer matrix (CV_32FC1)
     */
    template<typename DataType>
    void _square_sum_channels_into(const Mat& src, Mat& dst, Mat& buffer);
    
    /**
     * Multiply a vec matrix by a line matrix elementwise broadcasting the line matrix over channels of the vec matrix.
     * 
     * @param line_mat Input
     * @param vec_mat Input
     * @param res_mat Output
     * @param buffer Buffer matrix (CV_32FC3)
     */
    template<typename DataType>
    void _multiply_multi_channel(const Mat& line_mat, const Mat& vec_mat, Mat& res_mat, Mat& buffer);
    
    /**
     * Divide a vec matrix by a line matrix elementwise broadcasting the line matrix over channels of the vec matrix.
     * 
     * @param line_mat Input
     * @param vec_mat Input
     * @param res_mat Output
     * @param buffer Buffer matrix (CV_32FC3)
     */
    template<typename DataType>
    void _divide_multi_channel(const Mat& line_mat, const Mat& vec_mat, Mat& res_mat, Mat& buffer);
    
    
    
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

    
    
    /*
     * *****************************************************************
     * IMPLEMENTATION
     * compute_1D_depth_epi
     * *****************************************************************
     */
    template<typename DataType>
    void compute_1D_edge_confidence(
        const Mat& m_epi_,
        int s,
        Mat& m_edge_confidence_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        BufferDepth1D<DataType>& m_buffer_
    )
    {
        /*
         * Compute edge confidence
         */
        int filter_size = m_parameters_.m_edge_confidence_filter_size_;
        int center_index = (filter_size -1) / 2;

        // Get buffer variables
        Mat kernel = m_buffer_.filter_kernel;
        Mat tmp = m_buffer_.conv_tmp;
        Mat tmp2 = m_buffer_.sqsum_tmp;
        cv::Point tmp_param(-1,-1);
        
        for (int j=0; j<filter_size; j++)
        {
            if (j == center_index)
                continue;
            
            // Make filter with 1 at 1, 1 and -1 at i, j
            kernel.setTo(0.0);
            kernel.at<float>(center_index) = 1.0;
            kernel.at<float>(j) = -1.0;
            cv::filter2D(m_epi_.row(s), tmp, -1, kernel, tmp_param, 0, cv::BORDER_CONSTANT);
            
            // Sum square values into edge confidence
            _square_sum_channels_into<DataType>(tmp, m_edge_confidence_u_, tmp2);
        }
    }
    
    template<typename DataType>
    void compute_1D_depth_epi(
        const Mat& m_epi_,
        const Vec<float> m_d_list_,
        const Mat& m_indices_,
        int m_s_hat_,
        const Mat& m_edge_confidence_u_,
        Mat& m_disp_confidence_u_,
        Mat& m_best_depth_u_,
        Mat& m_rbar_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        BufferDepth1D<DataType>& m_buffer_,
        Mat& m_mask_u_
    ) 
    {
        
        // Dimensions
        int m_dim_s_ = m_epi_.rows;
        int m_dim_d_ = m_d_list_.size();
        int m_dim_u_ = m_epi_.cols;
        
        // Init score matrix
        Mat m_scores_u_d_ = m_buffer_.m_scores_u_d_;
        
        /*
         * Iterate over all columns of the EPI
         */
        
        // Get indices u to compute
        // Do not compute low-confidence values or value masked
        Mat thr_tmp = m_buffer_.thr_tmp;
        thr_tmp = m_edge_confidence_u_ > m_parameters_.m_edge_score_threshold_;
        if (!m_mask_u_.empty())
            cv::bitwise_and(thr_tmp, m_mask_u_, m_mask_u_);
        else
            m_mask_u_ = thr_tmp;
        
        Vec<cv::Point> locations = m_buffer_.locations;
        cv::findNonZero(m_mask_u_, locations);
        
//~ #pragma omp parallel for
        for (auto it = locations.begin(); it<locations.end(); it++)
        {
            int u = (*it).x;
            
            /*
             * Fill radiances
             */
            // Matrix of indices corresponding to the lines of disparities d
            Mat I = m_buffer_.I;
            cv::add(m_indices_, u, I);
            
            // Radiances
            Mat radiances_s_d = m_buffer_.radiances_s_d;
            Mat radiances_s_d_un_nanified = m_buffer_.radiances_s_d_un_nanified;
            
            // Interpolate
            // TODO this step is costly
            Mat card_R = m_buffer_.card_R;
            m_parameters_.m_interpolation_class_->interpolate_mat(m_epi_, I, radiances_s_d, card_R);
            
            //~ // Indicator of non-nan values
            //~ Vec<Mat> radiances_split;
            //~ cv::split(radiances_s_d, radiances_split);
            //~ Mat non_nan_indicator = radiances_split[0] == radiances_split[0];
            
            //~ // Compute number of non-nan radiances per column
            //~ Mat card_R = m_buffer_.card_R;

            //~ for (int d=0; d<m_dim_d_; d++)
            //~ {
                //~ card_R.at<float>(d) = cv::countNonZero(non_nan_indicator.col(d));
            //~ }
            
            /*
             * Compute r_bar iteratively
             */
            Mat r_bar = m_buffer_.r_bar;
            Mat r_m_r_bar = m_buffer_.r_m_r_bar;
            Mat K_r_m_r_bar_mat = m_buffer_.K_r_m_r_bar_mat;
            Mat K_r_m_r_bar_mat_vec = m_buffer_.K_r_m_r_bar_mat_vec;
            
            Mat r_K_r_m_r_bar_mat = m_buffer_.r_K_r_m_r_bar_mat;
            
            Mat sum_r_K_r_m_r_bar = m_buffer_.sum_r_K_r_m_r_bar;
            
            Mat sum_K_r_m_r_bar = m_buffer_.sum_K_r_m_r_bar;
            Mat sum_K_r_m_r_bar_vec = m_buffer_.sum_K_r_m_r_bar_vec;
            
            Mat r_bar_broadcast = m_buffer_.r_bar_broadcast;
            
            // Initialize r_bar to the values in s_hat
            radiances_s_d.row(m_s_hat_).copyTo(r_bar);
            
            // Replace nans with zeros (since these will be multiplied by zero anyway)
            cv::max(radiances_s_d, cv::Scalar(0.0), radiances_s_d_un_nanified);
            
            // Perform a partial mean shift to compute r_bar
            // TODO: This step is costly
            for (int i=0; i< m_parameters_.m_mean_shift_max_iter_; i++)
            {
                // r_bar repeated over lines 
                cv::repeat(r_bar, m_dim_s_, 1, r_bar_broadcast); 
                
                // r - r_bar
                // This matrix contains nans
                cv::subtract(radiances_s_d, r_bar_broadcast, r_m_r_bar);
                
                // K(r - r_bar)
                // Kernel fuction returns 0 if value is nan
                m_parameters_.m_kernel_class_->evaluate_mat(r_m_r_bar, K_r_m_r_bar_mat); 
                
                // r * K(r - r_bar)
                // Multiply should be of the same number of channels
                _multiply_multi_channel<DataType>(K_r_m_r_bar_mat, radiances_s_d_un_nanified, r_K_r_m_r_bar_mat, K_r_m_r_bar_mat_vec);
                
                //~ std::cout << radiances_s_d_un_nanified << std::endl;
                
                // Sum over lines
                cv::reduce(r_K_r_m_r_bar_mat, sum_r_K_r_m_r_bar, 0, cv::REDUCE_SUM);
                cv::reduce(K_r_m_r_bar_mat, sum_K_r_m_r_bar, 0, cv::REDUCE_SUM);
                
                //~ std::cout << sum_K_r_m_r_bar << std::endl;
                //~ std::cout << sum_r_K_r_m_r_bar << std::endl;
                
                // Divide should be of the same number of channels
                _divide_multi_channel<DataType>(sum_K_r_m_r_bar, sum_r_K_r_m_r_bar, r_bar, sum_K_r_m_r_bar_vec);
                
                // Set nans to zero
                cv::max(r_bar, cv::Scalar(0.0), r_bar);
                
                //~ std::cout << r_bar << std::endl;
                //~ assert(false);
            }

            /*
             * Compute scores 
             */
            // Get the last sum { K(r - r_bar) }
            m_parameters_.m_kernel_class_->evaluate_mat(r_m_r_bar, K_r_m_r_bar_mat);
            cv::reduce(K_r_m_r_bar_mat, sum_K_r_m_r_bar, 0, cv::REDUCE_SUM);
            
            // Get score
            cv::divide(sum_K_r_m_r_bar, card_R, sum_K_r_m_r_bar);
            // Set nans to zero
            cv::max(sum_K_r_m_r_bar, cv::Scalar(0.0), sum_K_r_m_r_bar);
            
            // Copy the line to the scores
            sum_K_r_m_r_bar.copyTo(m_scores_u_d_.row(u));
            //~ std::cout << r_bar << std::endl;
            //~ r_bar.copyTo(m_rbar_u_);
            
            
            //~ std::cout << m_rbar_u_ << std::endl;
            //~ assert(false);
            
            /*
             * Get best d (max score)
             */
            double minVal;
            double maxVal;
            cv::Point minIdx;
            cv::Point maxIdx;
            cv::minMaxLoc(m_scores_u_d_.row(u), &minVal, &maxVal, &minIdx, &maxIdx);
            
            m_best_depth_u_.at<float>(u) = m_d_list_.at(maxIdx.x);
            
            // Compute depth confidence score
            m_disp_confidence_u_.at<float>(u) = m_edge_confidence_u_.at<float>(u) * std::abs(maxVal - cv::mean(m_scores_u_d_.row(u))[0]);
            
            // Get final r_bar
            m_rbar_u_.at<DataType>(u) = r_bar.at<DataType>(maxIdx.x);
            
        }
        
    }
    
    template<typename DataType>
    void selective_median_filter(
        const Mat& src,
        Mat& dst,
        const Vec<Mat>& epis,
        int s_hat_,
        int size,
        const Mat& edge_scores,
        float score_threshold,
        float epsilon
    )
    {
        int m_dim_v_ = src.rows;
        int m_dim_u_ = src.cols;
        
        // Allocate matrix if not allocated yet
        if (dst.empty() || dst.size != src.size || dst.type() != src.type())
            dst = Mat(m_dim_v_, m_dim_u_, src.type(), cv::Scalar(0.0));
        
        int thr_max = omp_get_max_threads();
        Vec<Vec<float> > value_buffers;
        for (int t=0; t<thr_max; t++)
            value_buffers.push_back(Vec<float>());
        
        int width = (size-1)/2;
        
#pragma omp parallel for
        for (int v=0; v<m_dim_v_; v++)
        {
            Vec<float> buffer = value_buffers[omp_get_thread_num()];
            for (int u=0; u<m_dim_u_; u++)
            {
                // If mask is null, skip
                if (edge_scores.at<float>(v, u) > score_threshold)
                {
                    buffer.clear();
                    for (int k=std::max(0, v-width); k<std::min(m_dim_v_, v+width+1); k++)
                    {
                        for (int l=std::max(0, u-width); l<std::min(m_dim_u_, u+width+1); l++)
                        {
                            if (edge_scores.at<float>(k, l) > score_threshold && 
                                norm(
                                    epis[v].at<DataType>(s_hat_, u) - 
                                    epis[k].at<DataType>(s_hat_, l)
                                    ) < epsilon)
                            {
                                buffer.push_back(src.at<float>(k, l));
                            }
                        }
                    }
                    // Compute the median
                    std::nth_element(buffer.begin(), buffer.begin() + buffer.size() / 2, buffer.end());
                    dst.at<float>(v, u) = buffer[buffer.size() / 2];
                }
            }
        }
    }
    
    
    /*
     * *****************************************************************
     * IMPLEMENTATION
     * compute_1D_depth_epi_pile
     * *****************************************************************
     */
    
    template<typename DataType>
    void compute_1D_edge_confidence_pile(
        const Vec<Mat>& m_epis_,
        int s,
        Mat& m_edge_confidence_v_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        Vec<BufferDepth1D<DataType>* >& m_buffers_
    )
    {
        int m_v_dim_ = m_epis_.size();
        
        // Compute edge confidence for all rows
#pragma omp parallel for
        for (int v=0; v<m_v_dim_; v++)
        {
            Mat m_edge_confidence_u_ = m_edge_confidence_v_u_.row(v);
            
            compute_1D_edge_confidence<DataType>(
                m_epis_[v], 
                s, 
                m_edge_confidence_u_, 
                m_parameters_, 
                *m_buffers_[omp_get_thread_num()]
            );
        }
    }
    
    template<typename DataType>
    void compute_1D_depth_epi_pile(
        const Vec<Mat>& m_epis_,
        const Vec<float> m_d_list_,
        int m_s_hat_,
        const Mat& m_edge_confidence_v_u_,
        Mat& m_disp_confidence_v_u_,
        Mat& m_best_depth_v_u_,
        Mat& m_rbar_v_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        Vec<BufferDepth1D<DataType>* >& m_buffers_,
        Mat& m_mask_v_u_,
        bool m_verbose_
    )
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
        indices *= m_parameters_.m_slope_factor; // multiply by the slope factor
        
        // For progress bar
        float progress = 0.0;
        int barWidth = 40;
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        
#pragma omp parallel for
        for (int v=0; v<m_dim_v_; v++)
        {
            // Create views
            Mat m_epi_ = m_epis_[v];
            Mat m_edge_confidence_u_ = m_edge_confidence_v_u_.row(v);
            Mat m_disp_confidence_u_ = m_disp_confidence_v_u_.row(v);
            Mat m_best_depth_u_ = m_best_depth_v_u_.row(v);
            Mat m_rbar_u_ = m_rbar_v_u_.row(v);
            
            // Empty mask -> no masked point
            Mat mask;
            if (!m_mask_v_u_.empty())
                mask = m_mask_v_u_.row(v);
            
            //~ compute_1D_edge_confidence(
                //~ m_epi_,
                //~ m_s_hat_,
                //~ m_edge_confidence_u_,
                //~ m_parameters_,
                //~ *buffer[omp_get_thread_num()]
            //~ );
            
            compute_1D_depth_epi(
                m_epi_,
                m_d_list_,
                indices,
                m_s_hat_,
                m_edge_confidence_u_,
                m_disp_confidence_u_,
                m_best_depth_u_,
                m_rbar_u_,
                m_parameters_,
                *m_buffers_[omp_get_thread_num()],
                mask
            );
            
            //~ std::cout << rslf::type2str(m_rbar_u_.type()) << std::endl;
            //~ std::cout << m_rbar_u_.size << std::endl;
            //~ std::cout << m_rbar_u_ << std::endl;
            
            if (m_verbose_)
            {
#pragma omp critical
{
                // Display progress bar
                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
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
        }
        
        // Apply median filter
        if (m_verbose_)
            std::cout << std::endl << "Applying selective median fiter" << std::endl;
        
        Mat tmp;
        selective_median_filter<DataType>(
            m_best_depth_v_u_, 
            tmp,
            m_epis_,
            m_s_hat_,
            m_parameters_.m_median_filter_size_, 
            m_edge_confidence_v_u_, 
            m_parameters_.m_edge_score_threshold_,
            m_parameters_.m_median_filter_epsilon_
        );

        m_best_depth_v_u_ = tmp;
    }
    
    /*
     * *****************************************************************
     * IMPLEMENTATION
     * compute_2D_depth_epi
     * *****************************************************************
     */
    template<typename DataType>
    void compute_2D_edge_confidence(
        const Vec<Mat>& m_epis_,
        Vec<Mat>& m_edge_confidence_s_v_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        Vec<BufferDepth1D<DataType>* >& m_buffers_
    )
    {
        int m_s_dim_ = m_epis_[0].rows;
        
        // Compute edge confidence for all rows
        for (int s=0; s<m_s_dim_; s++)
        {
            Mat m_edge_confidence_v_u_ = m_edge_confidence_s_v_u_[s];
            
            compute_1D_edge_confidence_pile(
                m_epis_,
                s,
                m_edge_confidence_v_u_,
                m_parameters_,
                m_buffers_
            );
        }
    }
     
    template<typename DataType>
    void compute_2D_depth_epi(
        const Vec<Mat>& m_epis_,
        const Vec<float> m_d_list_,
        const Vec<Mat>& m_edge_confidence_s_v_u_,
        Vec<Mat>& m_disp_confidence_s_v_u_,
        Vec<Mat>& m_best_depth_s_v_u_,
        Vec<Mat>& m_rbar_s_v_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        Vec<BufferDepth1D<DataType>* >& m_buffers_,
        bool m_verbose_
    )
    {
        int m_dim_s_ = m_epis_[0].rows;
        int m_dim_d_ = m_d_list_.size();
        int m_dim_u_ = m_epis_[0].cols;
        int m_dim_v_ = m_epis_.size();
        int m_s_hat_ = (int)std::floor(m_dim_s_ / 2.0);
        
        // Assume edge confidence is computed on all the image
        
        // Create a mask (CV_8UC1) and init to C_e > thr
        Vec<Mat> mask_s_v_u_ = Vec<Mat>(m_dim_s_);
        for (int s=0; s<m_dim_s_; s++)
            mask_s_v_u_[s] = m_edge_confidence_s_v_u_[s] > m_parameters_.m_edge_score_threshold_;

        
        Mat m_edge_confidence_v_u_;
        Mat m_disp_confidence_v_u_;
        Mat m_best_depth_v_u_;
        Mat m_rbar_v_u_;
        Mat m_mask_v_u_;
        
        Vec<int> s_values;
        s_values.push_back(m_s_hat_);
        for (int s_offset=1; s_offset<m_dim_s_-m_s_hat_; s_offset++)
        {
            int s_sup = m_s_hat_ + s_offset;
            int s_inf = m_s_hat_ - s_offset;
            s_values.push_back(s_sup);
            if (s_inf > -1)
                s_values.push_back(s_inf);
        }
        
        // Iterate over lines of the epi (s) starting from the middle
        for (auto it = s_values.begin(); it < s_values.end(); it++) 
        {
            int s_hat = *it;
            
            if (m_verbose_)
                std::cout << "Computing s_hat=" << s_hat << std::endl;
            
            m_edge_confidence_v_u_ = m_edge_confidence_s_v_u_[s_hat];
            m_disp_confidence_v_u_ = m_disp_confidence_s_v_u_[s_hat];
            m_best_depth_v_u_ = m_best_depth_s_v_u_[s_hat];
            m_rbar_v_u_ = m_rbar_s_v_u_[s_hat];
            m_mask_v_u_ = mask_s_v_u_[s_hat];

            compute_1D_depth_epi_pile<DataType>(
                m_epis_,
                m_d_list_,
                s_hat,
                m_edge_confidence_v_u_,
                m_disp_confidence_v_u_,
                m_best_depth_v_u_,
                m_rbar_v_u_,
                m_parameters_,
                m_buffers_,
                m_mask_v_u_,
                false // verbose
            );
            
            if (m_verbose_)
                std::cout << "Propagation..." << std::endl;
            
            // Propagate depths over lines and update further mask lines
            // For each column of the s_hat row, draw the line, taking overlays into account
#pragma omp parallel for
            for (int v=0; v<m_dim_v_; v++)
            {
                Mat m_edge_confidence_u_ = m_edge_confidence_v_u_.row(v);
                Mat m_best_depth_u_ = m_best_depth_v_u_.row(v);
                Mat m_rbar_u_ = m_rbar_v_u_.row(v);
                for (int u=0; u<m_dim_u_; u++)
                {
                    //~ std::cout << rslf::type2str(m_rbar_u_.type()) << std::endl;
                    //~ std::cout << m_rbar_u_.size << std::endl;
                    //~ std::cout << m_rbar_u_ << std::endl;
                    // Only paint if the confidence threshold was high enough
                    //~ std::cout << m_disp_confidence_v_u_.at<float>(u) << std::endl;
                    //~ if (m_disp_confidence_v_u_.at<float>(u) > m_parameters_.m_disp_score_threshold_)
                    if (m_edge_confidence_u_.at<float>(u) > m_parameters_.m_edge_score_threshold_)
                    {
                        float current_depth_value = m_best_depth_u_.at<float>(u);
                        for (int s=0; s<m_dim_s_; s++)
                        {
                        
                            int requested_index = u + (int)std::round(m_best_depth_u_.at<float>(u) * (s_hat - s));
                            //~ std::cout << "u=" << u << ", reqindex=" << requested_index << ", s=" << s << ", v=" << v << std::endl;
                            //~ std::cout << m_epis_[v].at<DataType>(s, requested_index) << std::endl;
                            //~ std::cout << m_rbar_u_.at<DataType>(u) << std::endl;
                            //~ std::cout << norm<DataType>(m_epis_[v].at<DataType>(s, requested_index) - m_rbar_u_.at<DataType>(u)) << std::endl;
                            if 
                            (
                                requested_index > -1 && 
                                requested_index < m_dim_u_ && 
                                mask_s_v_u_[s].at<uchar>(v, requested_index) == 255 &&
                                norm<DataType>(m_epis_[v].at<DataType>(s, requested_index) - m_rbar_u_.at<DataType>(u)) < m_parameters_.m_propagation_epsilon 
                            )
                            {
                                m_best_depth_s_v_u_[s].at<float>(v, requested_index) = current_depth_value;
                                mask_s_v_u_[s].at<uchar>(v, requested_index) = 0;
                            }
                        }
                    }
                }
            }
            
        }
        
    }

}



#endif
