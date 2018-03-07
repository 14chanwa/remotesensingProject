#ifndef _RSLF_DEPTH_COMPUTATION
#define _RSLF_DEPTH_COMPUTATION

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rslf_types.hpp>
#include <rslf_plot.hpp>
#include <rslf_utils.hpp>


//~ #define _RSLF_DEPTH_COMPUTATION_DEBUG

#define _MEAN_SHIFT_MAX_ITER 10
#define _EDGE_CONFIDENCE_FILTER_SIZE 9
#define _MEDIAN_FILTER_SIZE 5
#define _SCORE_THRESHOLD 0.02


namespace rslf
{
    
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
     * Depth1DComputer
     * *****************************************************************
     */
    template<typename DataType>
    struct Depth1DParameters
    {
    private:
        Depth1DParameters() {}
        static Depth1DParameters m_default_;
        static bool m_default_inited_;
    public:
        // TODO enable to define different classes of parameters
        Interpolation1DClass<DataType>* m_interpolation_class_;
        KernelClass<DataType>* m_kernel_class_;
        float m_score_threshold_;
        float m_mean_shift_max_iter_;
        int m_edge_confidence_filter_size_;
        int m_median_filter_size_;
        
        ~Depth1DParameters() {
            delete m_interpolation_class_;
            delete m_kernel_class_;
        }
        
        static Depth1DParameters& get_default() {
            if (!m_default_inited_) {
                m_default_.m_interpolation_class_ = new Interpolation1DLinear<DataType>();
                m_default_.m_kernel_class_ = new BandwidthKernel<DataType>(0.2);
                m_default_.m_score_threshold_ = _SCORE_THRESHOLD;
                m_default_.m_mean_shift_max_iter_ = _MEAN_SHIFT_MAX_ITER;
                m_default_.m_edge_confidence_filter_size_ = _EDGE_CONFIDENCE_FILTER_SIZE;
                m_default_.m_median_filter_size_ = _MEDIAN_FILTER_SIZE;
            }
            return m_default_;
        }
    };
    
    template<typename DataType>
    Depth1DParameters<DataType> Depth1DParameters<DataType>::m_default_ = Depth1DParameters();
    
    template<typename DataType>
    bool Depth1DParameters<DataType>::m_default_inited_ = false;

    /**
     * Template class with depth computation using 1d slices of the EPI.
     */
    template<typename DataType>
    class Depth1DComputer
    {
    public:
        Depth1DComputer
        (
            const Mat epi, 
            const Vec<float> d_list,
            int s_hat = -1, // default s_hat will be s_max / 2,
            float epi_scale_factor = -1,
            const Depth1DParameters<DataType>& parameters = Depth1DParameters<DataType>::get_default()
        );
        ~Depth1DComputer();
        
        void run();
        Mat get_coloured_epi(int cv_colormap = cv::COLORMAP_JET);
    
    private:
        Mat m_epi_;
        Vec<float> m_d_list_;
        
        Mat m_edge_confidence_u_;
        Mat m_disp_confidence_u_;
        
        Mat m_scores_u_d_;
        
        Mat m_best_depth_u_;
        Mat m_score_depth_u_;
        
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
            Vec<Mat> epis, 
            const Vec<float> d_list,
            int s_hat = -1, // default s_hat will be s_max / 2,
            float epi_scale_factor = -1,
            const Depth1DParameters<DataType>& parameters = Depth1DParameters<DataType>::get_default()
      );
        ~Depth1DComputer_pile();
        
        void run();
        Mat get_coloured_epi(int v = -1, int cv_colormap = cv::COLORMAP_JET);
        Mat get_disparity_map(int cv_colormap = cv::COLORMAP_JET);
        int get_s_hat() { return m_s_hat_; }
    
    private:
        Vec<Mat> m_epis_;
        Vec<float> m_d_list_;
        
        Mat m_edge_confidence_v_u_;
        Mat m_disp_confidence_v_u_;
        
        Vec<Mat> m_scores_v_u_d_;
        
        Mat m_best_depth_v_u_;
        Mat m_score_depth_v_u_;
        
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
        /**
         * Dimension along v axis
         */
        int m_dim_v_;
        
        const Depth1DParameters<DataType>& m_parameters_;
    };
    
    /*
     * *****************************************************************
     * compute_1D_depth_epi
     * *****************************************************************
     */
    
    template<typename DataType>
    void compute_1D_depth_epi(
        const Mat& m_epi_,
        const Vec<float> m_d_list_,
        int m_s_hat_,
        Mat& m_edge_confidence_u_,
        Mat& m_disp_confidence_u_,
        Mat& m_scores_u_d_,
        Mat& m_best_depth_u_,
        Mat& m_score_depth_u_,
        const Depth1DParameters<DataType>& m_parameters_
    );
    
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
        Mat res = cv::Mat::zeros(indices.rows, indices.cols, data_matrix.type());
        
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
        Mat res = cv::Mat::zeros(indices.rows, indices.cols, data_matrix.type());
        
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
     * Depth1DComputer
     * *****************************************************************
     */
    
    template<typename DataType>
    Depth1DComputer<DataType>::Depth1DComputer
    (
        const Mat epi, 
        const Vec<float> d_list,
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
        
        // Edge confidence
        m_edge_confidence_u_ = cv::Mat::zeros(1, m_epi_.cols, CV_32FC1);
        
        // d confidence
        m_disp_confidence_u_ = cv::Mat::zeros(1, m_epi_.cols, CV_32FC1);
        
#ifdef _RSLF_DEPTH_COMPUTATION_DEBUG
        std::cout << "created m_radiances_u_s_d_ of size " << m_radiances_u_s_d_.size() << " x " << m_radiances_u_s_d_.at(0).size << std::endl;
#endif
        
        // Scores and best scores & depths
        m_scores_u_d_ = cv::Mat::zeros(m_dim_u_, m_dim_d_, CV_32FC1);
        m_best_depth_u_ = cv::Mat::zeros(1, m_dim_u_, CV_32FC1);
        m_score_depth_u_ = cv::Mat::zeros(1, m_dim_u_, CV_32FC1);
        
    }
    
    template<typename DataType>
    Depth1DComputer<DataType>::~Depth1DComputer() {

    }

    template<typename DataType>
    void Depth1DComputer<DataType>::run() 
    {
        compute_1D_depth_epi(
            m_epi_,
            m_d_list_,
            m_s_hat_,
            m_edge_confidence_u_,
            m_disp_confidence_u_,
            m_scores_u_d_,
            m_best_depth_u_,
            m_score_depth_u_,
            m_parameters_
        );
    }
    
    template<typename DataType>
    void compute_1D_depth_epi(
        const Mat& m_epi_,
        const Vec<float> m_d_list_,
        int m_s_hat_,
        Mat& m_edge_confidence_u_,
        Mat& m_disp_confidence_u_,
        Mat& m_scores_u_d_,
        Mat& m_best_depth_u_,
        Mat& m_score_depth_u_,
        const Depth1DParameters<DataType>& m_parameters_
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
        for (int j=0; j<filter_size; j++)
        {
            if (j != center_index)
            {
                // Make filter with 1 at 1, 1 and -1 at i, j
                Mat kernel = cv::Mat::zeros(1, filter_size, CV_32FC1);
                kernel.at<float>(center_index) = 1.0;
                kernel.at<float>(j) = -1.0;
                Mat tmp;
                cv::filter2D(m_epi_.row(m_s_hat_), tmp, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
                
                if (tmp.channels() > 1)
                {
                    Vec<Mat> channels;
                    cv::split(tmp, channels);
                    for (int c=0; c<channels.size(); c++) 
                    {
                        cv::pow(channels[c], 2, channels[c]);
                        m_edge_confidence_u_ += channels[c];
                    }
                }
                else
                {
                    cv::pow(tmp, 2, tmp);
                    m_edge_confidence_u_ += tmp;
                }
            }
            
        }
        
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
        
        /*
         * Iterate over all columns of the EPI
         */
#pragma omp parallel for
        for (int u=0; u<m_dim_u_; u++)
        {
#ifdef _RSLF_DEPTH_COMPUTATION_DEBUG
            std::cout << "u=" << u << std::endl;
#endif
            
            // Do not compute low-confidence values
            if (m_edge_confidence_u_.at<float>(u) < m_parameters_.m_score_threshold_)
                continue;
            
            /*
             * Fill radiances
             */
            // Matrix of indices corresponding to the lines of disparities d
            Mat I = indices + u;
            
            // Create new radiance view
            // https://stackoverflow.com/questions/23998422/how-to-access-slices-of-a-3d-matrix-in-opencv
            Mat radiances_s_d = Mat(m_dim_s_, m_dim_d_, m_epi_.type(), cv::Scalar(0.0));
#ifdef _RSLF_DEPTH_COMPUTATION_DEBUG
            std::cout << radiances_s_d.size << std::endl;
#endif
            
            // Interpolate
            radiances_s_d = m_parameters_.m_interpolation_class_->interpolate_mat(m_epi_, I);
            
            // Indicator of non-nan values
            Mat non_nan_indicator = radiances_s_d == radiances_s_d;

            // Indicator should be CV_8UC1
            if (non_nan_indicator.channels() > 1)
            {
                cv::extractChannel(non_nan_indicator, non_nan_indicator, 0);
            }
            
            // Compute number of non-nan radiances per column
            Mat card_R(1, m_dim_d_, CV_32FC1);
            for (int d=0; d<m_dim_d_; d++)
            {
                card_R.at<float>(d) = cv::countNonZero(non_nan_indicator.col(d));
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
            for (int i=0; i< m_parameters_.m_mean_shift_max_iter_; i++)
            {
                // r_bar repeated over lines
                cv::repeat(r_bar, m_dim_s_, 1, r_bar_broadcast); 
                
                // r - r_bar
                cv::subtract(radiances_s_d, r_bar_broadcast, r_m_r_bar);
                
                // K(r - r_bar)
                k_r_m_r_bar_mat = m_parameters_.m_kernel_class_->evaluate_mat(r_m_r_bar); // returns 0 if value is nan
                
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
                
                // Divide
                cv::divide(sum_r_K_r_m_r_bar, sum_K_r_m_r_bar, r_bar);
                
                // Set nans to zero
                cv::max(r_bar, cv::Scalar(0.0), r_bar);
            }

            /*
             * Compute scores 
             */
            // Get the last sum { K(r - r_bar) }
            k_r_m_r_bar_mat = m_parameters_.m_kernel_class_->evaluate_mat(r_m_r_bar);
            cv::reduce(k_r_m_r_bar_mat, sum_K_r_m_r_bar, 0, cv::REDUCE_SUM);
            
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
            cv::minMaxLoc(m_scores_u_d_.row(u).clone(), &minVal, &maxVal, &minIdx, &maxIdx);

#ifdef _RSLF_DEPTH_COMPUTATION_DEBUG
            std::cout << maxVal << ", " << maxIdx << std::endl;
#endif
            
            m_best_depth_u_.at<float>(u) = m_d_list_.at(maxIdx.x);
            m_score_depth_u_.at<float>(u) = maxVal;
            
            // Confidence
            // m_score_threshold_
            m_disp_confidence_u_.at<float>(u) = m_edge_confidence_u_.at<float>(u) * std::abs(m_score_depth_u_.at<float>(u) - cv::mean(m_scores_u_d_.row(u))[0]);
 
        }
        
        /*
         * Apply a median filter on the resulting depths
         */
        //cv::medianBlur(m_best_depth_u_, m_best_depth_u_, _MEDIAN_FILTER_SIZE);

        
#ifdef _RSLF_DEPTH_COMPUTATION_DEBUG
        // Inspect a slice
        Mat slice = m_radiances_u_s_d_.at(1).col(0);
        std::cout << slice << std::endl;
        std::cout << slice.size << std::endl;
        std::cout << rslf::type2str(slice.type()) << std::endl;
#endif
    }
    
    template<typename DataType>
    Mat Depth1DComputer<DataType>::get_coloured_epi(int cv_colormap) {
        
        //~ std::cout << m_edge_confidence_u_ << std::endl;
        //~ std::cout << "m_parameters_.m_score_threshold_ " << m_parameters_.m_score_threshold_ << std::endl;
        
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
        const Vec<Mat> epis, 
        const Vec<float> d_list,
        int s_hat,
        float epi_scale_factor,
        const Depth1DParameters<DataType>& parameters
    ) : 
        m_d_list_(d_list),
        m_parameters_(parameters)
    {
        m_epis_ = Vec<Mat>(epis.size(), Mat(epis[0].rows, epis[0].cols, epis[0].type()));
        
        std::cout << "epi_scale_factor" << epi_scale_factor << std::endl;
        
        // Look for the max and min values across all epis
        // If provided scale factor is invalid, scale from max in all channels
        if (epis[0].depth() != CV_8U && epi_scale_factor < 0)
        {
            for (int v=0; v<epis.size(); v++)
            {
                Mat epi = epis[v];
                Vec<Mat> channels;
                cv::split(epi, channels);
                for (int c=0; c<channels.size(); c++) 
                {
                    double min, max;
                    cv::minMaxLoc(epi, &min, &max);
                    epi_scale_factor = std::max((float)max, epi_scale_factor);
                }
            }
        }
        std::cout << "epi_scale_factor" << epi_scale_factor << std::endl;
        
        // If the input epi is a uchar, scale uchar to 1.0
//~ #pragma omp parallel for
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
        
        //~ std::cout << "m_epis_[50]" << m_epis_[50] << std::endl;
        
        std::cout << "Init EPI OK" << std::endl;
        
        // Dimensions
        m_dim_s_ = m_epis_[0].rows;
        m_dim_d_ = m_d_list_.size();
        m_dim_u_ = m_epis_[0].cols;
        m_dim_v_ = m_epis_.size();
        
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
        
        std::cout << "Init m_edge_confidence_v_u_ OK" << std::endl;
        
        // Disp confidence
        m_disp_confidence_v_u_ = Mat(m_dim_v_, m_dim_u_, CV_32FC1);

        
        std::cout << "Init m_disp_confidence_v_u_ OK" << std::endl;
        
        
        // Scores and best scores & depths
        m_scores_v_u_d_ = Vec<Mat>(m_dim_v_, Mat(m_dim_u_, m_dim_d_, CV_32FC1));
        for (int v=0; v<m_dim_v_; v++)
        {
            m_scores_v_u_d_[v] = Mat(m_dim_u_, m_dim_d_, CV_32FC1, cv::Scalar(0.0));
        }
        m_best_depth_v_u_ = cv::Mat::zeros(m_dim_v_, m_dim_u_, CV_32FC1);
        m_score_depth_v_u_ = cv::Mat::zeros(m_dim_v_, m_dim_u_, CV_32FC1);
        
        std::cout << "Init ended" << std::endl;
        
    }
    
    template<typename DataType>
    Depth1DComputer_pile<DataType>::~Depth1DComputer_pile() {

    }

    template<typename DataType>
    void Depth1DComputer_pile<DataType>::run() 
    {
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
                m_s_hat_,
                m_edge_confidence_u_,
                m_disp_confidence_u_,
                m_scores_u_d_,
                m_best_depth_u_,
                m_score_depth_u_,
                m_parameters_
            );
            
       }
        
    }
    
    template<typename DataType>
    Mat Depth1DComputer_pile<DataType>::get_coloured_epi(int v, int cv_colormap) {
        
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
        Mat disparity_map;
        
        std::cout << "m_best_depth_v_u_ " << rslf::type2str(m_best_depth_v_u_.type()) << std::endl;
        
        disparity_map = rslf::copy_and_scale_uchar(m_best_depth_v_u_);
        cv::applyColorMap(disparity_map, disparity_map, cv_colormap);
        std::cout << "disparity_map " << rslf::type2str(disparity_map.type()) << ", " << disparity_map.size << std::endl;
        
        // Threshold scores
        Mat disparity_map_with_scores = cv::Mat::zeros(m_dim_v_, m_dim_u_, disparity_map.type());
        
        std::cout << "m_edge_confidence_v_u_ " << rslf::type2str(m_edge_confidence_v_u_.type()) << ", " << m_edge_confidence_v_u_.size << std::endl;
        
        Mat score_mask = m_edge_confidence_v_u_ > m_parameters_.m_score_threshold_;
        //~ std::cout << score_mask << std::endl;
        //~ std::cout << m_score_threshold_ << std::endl;
        std::cout << "score_mask " << rslf::type2str(score_mask.type()) << ", " << score_mask.size << std::endl;
        
        cv::add(disparity_map, disparity_map_with_scores, disparity_map_with_scores, score_mask);
        
        return disparity_map_with_scores;
    }

}



#endif
