#ifndef _RSLF_DEPTH_COMPUTATION_CORE
#define _RSLF_DEPTH_COMPUTATION_CORE

#include <string>
#include <vector>
#include <chrono>

#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <rslf_types.hpp>
#include <rslf_interpolation.hpp>
#include <rslf_kernels.hpp>


//~ #define _RSLF_DEPTH_COMPUTATION_DEBUG

#define _MEAN_SHIFT_MAX_ITER 10
#define _EDGE_CONFIDENCE_FILTER_SIZE 9
#define _MEDIAN_FILTER_SIZE 5
#define _MEDIAN_FILTER_EPSILON 0.1
#define _SCORE_THRESHOLD 0.02


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
            m_kernel_class_ = new BandwidthKernel<DataType>(0.2);
            m_score_threshold_ = _SCORE_THRESHOLD;
            m_mean_shift_max_iter_ = _MEAN_SHIFT_MAX_ITER;
            m_edge_confidence_filter_size_ = _EDGE_CONFIDENCE_FILTER_SIZE;
            m_median_filter_size_ = _MEDIAN_FILTER_SIZE;
            m_median_filter_epsilon_ = _MEDIAN_FILTER_EPSILON;
        }

        Interpolation1DClass<DataType>* m_interpolation_class_;
        KernelClass<DataType>* m_kernel_class_;
        float m_score_threshold_;
        float m_mean_shift_max_iter_;
        int m_edge_confidence_filter_size_;
        int m_median_filter_size_;
        float m_median_filter_epsilon_;
        
        ~Depth1DParameters() 
        {
            delete m_interpolation_class_;
            delete m_kernel_class_;
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
    
    struct BufferDepth1D {
        Mat filter_kernel;
        Mat conv_tmp;
        Mat sqsum_tmp;
        Mat I;
        Mat radiances_s_d;
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
    void compute_1D_depth_epi(
        const Mat& m_epi_,
        const Vec<float> m_d_list_,
        const Mat& m_indices_,
        int m_s_hat_,
        Mat& m_edge_confidence_u_,
        Mat& m_disp_confidence_u_,
        Mat& m_scores_u_d_,
        Mat& m_best_depth_u_,
        Mat& m_score_depth_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        BufferDepth1D& m_buffer_
    );
    
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
    

    
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
    
    
    template<typename DataType>
    void _square_sum_into_multi_channel(const Mat& vec_mat, Mat& res_mat, Mat& buffer);
        
    template<typename DataType>
    void _multiply_multi_channel(const Mat& line_mat, const Mat& vec_mat, Mat& res_mat, Mat& buffer);
    
    template<typename DataType>
    void _divide_multi_channel(const Mat& line_mat, const Mat& vec_mat, Mat& res_mat, Mat& buffer);
    
    template<typename DataType>
    void _square_sum_channels_into(const Mat& src, Mat& dst, Mat& buffer);
    
    
    /*
     * *****************************************************************
     * IMPLEMENTATION
     * compute_1D_depth_epi
     * *****************************************************************
     */
    
    template<typename DataType>
    void compute_1D_depth_epi(
        const Mat& m_epi_,
        const Vec<float> m_d_list_,
        const Mat& m_indices_,
        int m_s_hat_,
        Mat& m_edge_confidence_u_,
        Mat& m_disp_confidence_u_,
        Mat& m_scores_u_d_,
        Mat& m_best_depth_u_,
        Mat& m_score_depth_u_,
        const Depth1DParameters<DataType>& m_parameters_,
        BufferDepth1D& m_buffer_
    ) 
    {
        
        // Dimensions
        int m_dim_s_ = m_epi_.rows;
        int m_dim_d_ = m_d_list_.size();
        int m_dim_u_ = m_epi_.cols;
        
        /*
         * Compute edge confidence
         */
        int filter_size = m_parameters_.m_edge_confidence_filter_size_;
        int center_index = (filter_size -1) / 2;

        // Get buffer variables
        Mat kernel = filter_kernel;
        if (kernel.empty())
            kernel = cv::Mat::zeros(1, filter_size, CV_32FC1);
        Mat tmp = m_buffer_.conv_tmp;
        Mat tmp2 = m_buffer_.sqsum_tmp;
        
        for (int j=0; j<filter_size; j++)
        {
            if (j == center_index)
                continue;
            
            // Make filter with 1 at 1, 1 and -1 at i, j
            kernel.setTo(0.0);
            kernel.at<float>(center_index) = 1.0;
            kernel.at<float>(j) = -1.0;
            cv::filter2D(m_epi_.row(m_s_hat_), tmp, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
            
            // Sum square values into edge confidence
            _square_sum_channels_into<DataType>(tmp, m_edge_confidence_u_, tmp2);
        }
        
        /*
         * Iterate over all columns of the EPI
         */
        
//~ #pragma omp parallel for
        for (int u=0; u<m_dim_u_; u++)
        {
            
            // Do not compute low-confidence values
            if (m_edge_confidence_u_.at<float>(u) < m_parameters_.m_score_threshold_)
                continue;
            
            /*
             * Fill radiances
             */
            // Matrix of indices corresponding to the lines of disparities d
            Mat I = m_buffer_.I;
            cv::add(m_indices_, u, I);
            
            // Radiances
            Mat radiances_s_d = m_buffer_.radiances_s_d;
            
            // Interpolate
            // TODO this step is costly
            m_parameters_.m_interpolation_class_->interpolate_mat(m_epi_, I, radiances_s_d);
            
            // Indicator of non-nan values
            Vec<Mat> radiances_split;
            cv::split(radiances_s_d, radiances_split);
            Mat non_nan_indicator = radiances_split[0] == radiances_split[0];
            
            // Compute number of non-nan radiances per column
            Mat card_R = m_buffer_.card_R;
            if (card_R.empty())
                card_R = Mat(1, m_dim_d_, CV_32FC1);

            for (int d=0; d<m_dim_d_; d++)
            {
                card_R.at<float>(d) = cv::countNonZero(non_nan_indicator.col(d));
            }
            
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

            // Perform a partial mean shift to compute r_bar
            // TODO: This step is costly
            for (int i=0; i< m_parameters_.m_mean_shift_max_iter_; i++)
            {
                // r_bar repeated over lines 
                cv::repeat(r_bar, m_dim_s_, 1, r_bar_broadcast); 
                
                // r - r_bar
                cv::subtract(radiances_s_d, r_bar_broadcast, r_m_r_bar);
                
                // K(r - r_bar)
                m_parameters_.m_kernel_class_->evaluate_mat(r_m_r_bar, K_r_m_r_bar_mat); // returns 0 if value is nan
                
                // r * K(r - r_bar)
                // Multiply should be of the same number of channels
                _multiply_multi_channel<DataType>(K_r_m_r_bar_mat, radiances_s_d, r_K_r_m_r_bar_mat, K_r_m_r_bar_mat_vec);
        
                // Sum over lines
                cv::reduce(r_K_r_m_r_bar_mat, sum_r_K_r_m_r_bar, 0, cv::REDUCE_SUM);
                cv::reduce(K_r_m_r_bar_mat, sum_K_r_m_r_bar, 0, cv::REDUCE_SUM);
                       
                // Divide should be of the same number of channels
                _divide_multi_channel<DataType>(sum_K_r_m_r_bar, sum_r_K_r_m_r_bar, r_bar, sum_K_r_m_r_bar_vec);
                
                // Set nans to zero
                cv::max(r_bar, cv::Scalar(0.0), r_bar);
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
            
            /*
             * Get best d (max score)
             */
            double minVal;
            double maxVal;
            cv::Point minIdx;
            cv::Point maxIdx;
            cv::minMaxLoc(m_scores_u_d_.row(u), &minVal, &maxVal, &minIdx, &maxIdx);
            
            m_best_depth_u_.at<float>(u) = m_d_list_.at(maxIdx.x);
            m_score_depth_u_.at<float>(u) = maxVal;
            
            // Compute depth confidence score
            m_disp_confidence_u_.at<float>(u) = m_edge_confidence_u_.at<float>(u) * std::abs(m_score_depth_u_.at<float>(u) - cv::mean(m_scores_u_d_.row(u))[0]);
 
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

}



#endif
