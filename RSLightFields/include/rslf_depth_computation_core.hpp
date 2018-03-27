#ifndef _RSLF_DEPTH_COMPUTATION_CORE
#define _RSLF_DEPTH_COMPUTATION_CORE

#include <chrono>
#include <omp.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <rslf_interpolation.hpp>
#include <rslf_kernels.hpp>


//~ #define _RSLF_DEPTH_COMPUTATION_DEBUG

// Default parameters
#define _MEAN_SHIFT_MAX_ITER 10
#define _EDGE_CONFIDENCE_FILTER_SIZE 9
#define _MEDIAN_FILTER_SIZE 5
#define _MEDIAN_FILTER_EPSILON 0.1
#define _EDGE_SCORE_THRESHOLD 0.02
#define _DISP_SCORE_THRESHOLD 0.01
#define _PROPAGATION_EPSILON 0.1

#define _BANDWIDTH_KERNEL_PARAMETER 0.2

#define _EDGE_CONFIDENCE_OPENING_TYPE cv::MORPH_ELLIPSE
#define _EDGE_CONFIDENCE_OPENING_SIZE 1

#define _SHADOW_NORMALIZED_LEVEL 0.05 * 1.73205080757 
// between 0 and 1, 0 being dark and 1 being blank
// multiplied by sqrt(3) for consistency with 3channels

//~ #define _USE_DISP_CONFIDENCE_SCORE
// if not defined, will use C_e as the propagation threshold, else will
// use C_d > par_disp_score_threshold.


// Useful links
// https://docs.opencv.org/3.4.1/
// https://docs.opencv.org/3.4.1/d3/d63/classcv_1_1Mat.html
// https://docs.opencv.org/3.4.1/d2/de8/group__core__array.html
// https://stackoverflow.com/questions/23998422/how-to-access-slices-of-a-3d-matrix-in-opencv
// https://stackoverflow.com/questions/2724708/is-it-a-good-practice-to-pass-struct-object-as-parameter-to-a-function-in-c


/*! 
 * \file 
 * \brief Implement low-level depth computation functions
 */ 


namespace rslf
{
    
/*
 * *****************************************************************
 * Depth1DParameters
 * *****************************************************************
 */

/**
 * \brief Impelment a structure containing all the algorithm parameters.
 */
template<typename DataType>
struct Depth1DParameters
{
private:
    static Depth1DParameters s_default;
    bool m_owner_classes = false;
    
public:
    Depth1DParameters() // initialize at default values
    {
        par_interpolation_class = new Interpolation1DLinear<DataType>();
        //~ par_interpolation_class = new Interpolation1DNearestNeighbour<DataType>();
        par_kernel_class = new BandwidthKernel<DataType>(_BANDWIDTH_KERNEL_PARAMETER);
        
        par_edge_score_threshold = _EDGE_SCORE_THRESHOLD;
        par_disp_score_threshold = _DISP_SCORE_THRESHOLD;
        par_mean_shift_max_iter = _MEAN_SHIFT_MAX_ITER;
        
        par_edge_confidence_filter_size = _EDGE_CONFIDENCE_FILTER_SIZE;
        par_edge_confidence_opening_type = _EDGE_CONFIDENCE_OPENING_TYPE;
        par_edge_confidence_opening_size = _EDGE_CONFIDENCE_OPENING_SIZE;
        
        par_median_filter_size = _MEDIAN_FILTER_SIZE;
        par_median_filter_epsilon = _MEDIAN_FILTER_EPSILON;
        par_propagation_epsilon = _PROPAGATION_EPSILON;
        
        par_slope_factor = 1.0; // Useful when subsampling EPIs
        
        
    }
    
    Depth1DParameters(const Depth1DParameters<DataType>& a_other)
    {
        Depth1DParameters<DataType>::operator=(a_other);
        // Notify the non-ownership of the class pointers
        m_owner_classes = false;
    }

    Interpolation1DClass<DataType>* par_interpolation_class;
    KernelClass<DataType>*          par_kernel_class;
    
    float   par_edge_score_threshold;
    float   par_disp_score_threshold;
    float   par_mean_shift_max_iter;
    int     par_edge_confidence_filter_size;
    int     par_edge_confidence_opening_type;
    int     par_edge_confidence_opening_size;
    int     par_median_filter_size;
    float   par_median_filter_epsilon;
    float   par_propagation_epsilon;
    float   par_slope_factor;
    
    ~Depth1DParameters() 
    {
        if (m_owner_classes)
        {
            delete par_interpolation_class;
            delete par_kernel_class;
        }
    }
    
    static Depth1DParameters& get_default() // get a static default instance
    {
        return s_default;
    }
};

template<typename DataType>
Depth1DParameters<DataType> Depth1DParameters<DataType>::s_default = Depth1DParameters();


/*
 * *****************************************************************
 * compute_1D_depth_epi
 * *****************************************************************
 */
/**
 * \brief Implement a buffer containing re-usable temporary variables
 * in order to avoid multiple unnecessary allocations.
 */
template <typename DataType>
struct BufferDepth1D {
    
    BufferDepth1D
    (
        int a_dim_s,
        int a_dim_d, 
        int a_dim_u, 
        int a_data_type, 
        const Depth1DParameters<DataType>& a_parameters
    )
    {
        buf_filter_kernel = cv::Mat::zeros(1, a_parameters.par_edge_confidence_filter_size, CV_32FC1);
        buf_scores_u_d = Mat(a_dim_u, a_dim_d, CV_32FC1, cv::Scalar(0.0));

        buf_S = Mat(a_dim_s, 1, CV_32FC1);
        buf_D = Mat(1, a_dim_d, CV_32FC1);

        buf_card_R = Mat(1, a_dim_d, CV_32FC1);
        buf_radiances_s_d = cv::Mat::zeros(a_dim_s, a_dim_d, a_data_type);
    }
    
    Vec<cv::Point>  buf_locations;
    
    Mat             buf_scores_u_d;

    Mat             buf_filter_kernel;
    Mat             buf_conv_tmp;
    Mat             buf_sqsum_tmp;

    Mat             buf_I;
    Mat             buf_S;
    Mat             buf_D;

    Mat             buf_radiances_s_d;
    Mat             buf_radiances_s_d_un_nanified;
    Mat             buf_card_R;

    Mat             buf_r_bar;
    Mat             buf_r_m_r_bar;
    Mat             buf_K_r_m_r_bar_mat;
    Mat             buf_K_r_m_r_bar_mat_vec;
    Mat             buf_r_K_r_m_r_bar_mat;
    Mat             buf_sum_r_K_r_m_r_bar;
    Mat             buf_sum_K_r_m_r_bar;
    Mat             buf_sum_K_r_m_r_bar_vec;
    Mat             buf_r_bar_broadcast;
};

/**
 * \brief Compute the edge confidence score for a single line along the
 * dimension u.
 */
template<typename DataType>
void compute_1D_edge_confidence(
    const Mat&  a_epi,
    int         a_s,
    Mat&        a_edge_confidence_u,
    Mat&        a_edge_confidence_mask_u,
    const Depth1DParameters<DataType>&  a_parameters,
    BufferDepth1D<DataType>&            a_buffer
);

/**
 * \brief Given a single EPI along dimensions (s, u) (v is fixed), compute
 * the slopes only looking at a single line (given by s_hat).
 */
template<typename DataType>
void compute_1D_depth_epi(
    const Mat&  a_epi,
    const Mat&  a_dmin_u,
    const Mat&  a_dmax_u,
    int         a_dim_d,
    int         a_s_hat,
    const Mat&  a_edge_confidence_u,
    const Mat&  a_edge_confidence_mask_u,
    Mat&        a_disp_confidence_u,
    Mat&        a_best_depth_u,
    Mat&        a_rbar_u,
    const Depth1DParameters<DataType>&  a_parameters,
    BufferDepth1D<DataType>&            a_buffer,
    Mat&        a_mask_u
);

/*
 * *****************************************************************
 * compute_1D_depth_epi_pile
 * *****************************************************************
 */

/**
 * \brief Given a vector of EPIs, compute the edge confidence along the line
 * of given index s along the dimension u.
 */
template<typename DataType>
void compute_1D_edge_confidence_pile(
    const Vec<Mat>& a_epis,
    int             a_s,
    Mat&            a_edge_confidence_v_u,
    Mat&            a_edge_confidence_mask_v_u,
    const Depth1DParameters<DataType>&  a_parameters,
    Vec<BufferDepth1D<DataType>* >&     a_buffers
);

/**
 * \brief For every EPI in the given vector (i.e. for all v), compute the
 * slopes only looking at a single line (given by s_hat).
 */
template<typename DataType>
void compute_1D_depth_epi_pile(
    const Vec<Mat>& a_epis,
    const Mat&      a_dmin_v_u,
    const Mat&      a_dmax_v_u,
    int             a_dim_d,
    int             a_s_hat,
    const Mat&      a_edge_confidence_v_u,
    const Mat&      a_edge_confidence_mask_v_u,
    Mat&            a_disp_confidence_v_u,
    Mat&            a_best_depth_v_u,
    Mat&            a_rbar_v_u,
    const Depth1DParameters<DataType>&  a_parameters,
    Vec<BufferDepth1D<DataType>* >&     a_buffers,
    Mat&            a_mask_v_u,
    bool            a_verbose = true
);


/*
 * *****************************************************************
 * compute_2D_depth_epi
 * *****************************************************************
 */

/**
 * \brief Given a vector of EPIs, compute the edge confidence on all lines
 * for all EPIs (so that an edge confidence value is given for all (s, v, u)).
 */
template<typename DataType>
void compute_2D_edge_confidence(
    const Vec<Mat>& a_epis,
    Vec<Mat>&       a_edge_confidence_s_v_u,
    Vec<Mat>&       a_edge_confidence_mask_s_v_u,
    const Depth1DParameters<DataType>&  a_parameters,
    Vec<BufferDepth1D<DataType>* >&     a_buffers
);

/**
 * \brief Given a vector of EPIs, compute the disparities for all points (s, v, u)
 * using the propagation process along the temporal axis s.
 */
template<typename DataType>
void compute_2D_depth_epi(
    const Vec<Mat>&     a_epis,
    const Vec<Mat>&     a_dmin_s_v_u,
    const Vec<Mat>&     a_dmax_s_v_u,
    int                 a_dim_d,
    const Vec<Mat>&     a_edge_confidence_s_v_u,
    const Vec<Mat>&     a_edge_confidence_mask_s_v_u,
    Vec<Mat>&           a_disp_confidence_s_v_u,
    Vec<Mat>&           a_best_depth_s_v_u,
    Vec<Mat>&           a_rbar_s_v_u,
    const Depth1DParameters<DataType>&  a_parameters,
    Vec<BufferDepth1D<DataType>* >&     a_buffers,
    bool                a_verbose = true
);



/*
 * *****************************************************************
 * selective_median_filter
 * *****************************************************************
 */

/**
 * \brief Filter the given spatial image of dimensions (v, u) using a
 * selective median filter (only points with high edge confidence and
 * similar color are taken into account). 
 */
template<typename DataType>
void selective_median_filter(
    const Mat&  a_src,
    Mat&        a_dst,
    const Vec<Mat>& a_epis,
    int         a_s_hat,
    int         a_size,
    const Mat&  a_mask_v_u,
    float       a_epsilon
);

/*
 * Useful template functions
 */

/**
 * \brief Sum the squares of the values across channels of the input matrix
 * 
 * @param src Input
 * @param dst Output
 * @param buffer Buffer matrix (CV_32FC1)
 */
template<typename DataType>
void _square_sum_channels_into(const Mat& src, Mat& dst, Mat& buffer);

/**
 * \brief Multiply a vec matrix by a line matrix elementwise broadcasting the line matrix over channels of the vec matrix.
 * 
 * @param line_mat Input
 * @param vec_mat Input
 * @param res_mat Output
 * @param buffer Buffer matrix (CV_32FC3)
 */
template<typename DataType>
void _multiply_multi_channel(const Mat& line_mat, const Mat& vec_mat, Mat& res_mat, Mat& buffer);

/**
 * \brief Divide a vec matrix by a line matrix elementwise broadcasting the line matrix over channels of the vec matrix.
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
    const Mat&  a_epi,
    int         a_s,
    Mat&        a_edge_confidence_u,
    Mat&        a_edge_confidence_mask_u,
    const Depth1DParameters<DataType>&  a_parameters,
    BufferDepth1D<DataType>&            a_buffer
)
{
    /*
     * Compute edge confidence
     */
    int filter_size = a_parameters.par_edge_confidence_filter_size;
    int center_index = (filter_size -1) / 2;

    // Get buffer variables
    Mat kernel = a_buffer.buf_filter_kernel;
    Mat tmp = a_buffer.buf_conv_tmp;
    Mat tmp2 = a_buffer.buf_sqsum_tmp;
    
    cv::Point tmp_param(-1,-1);
    
    for (int j=0; j<filter_size; j++)
    {
        if (j == center_index)
            continue;
        
        // Make filter with 1 at 1, 1 and -1 at i, j
        kernel.setTo(0.0);
        kernel.at<float>(center_index) = 1.0;
        kernel.at<float>(j) = -1.0;
        cv::filter2D(a_epi.row(a_s), tmp, -1, kernel, tmp_param, 0, cv::BORDER_REFLECT_101);
        
        // Sum square values into edge confidence
        _square_sum_channels_into<DataType>(tmp, a_edge_confidence_u, tmp2);
    }
    
    // TODO
    // Dirty way to remove all dark pixels
    // Get elementwise norm
    Mat im_norm = Mat(1, a_edge_confidence_u.cols, CV_32FC1);
    for (int u=0; u<a_edge_confidence_u.cols; u++)
    {
        im_norm.at<float>(u) = norm<DataType>(a_epi.row(a_s).at<DataType>(u));
    }
    a_edge_confidence_u.setTo(0.0, im_norm < _SHADOW_NORMALIZED_LEVEL);
    
    a_edge_confidence_mask_u = a_edge_confidence_u > a_parameters.par_edge_score_threshold;
    
}

template<typename DataType>
void compute_1D_depth_epi(
    const Mat&  a_epi,
    const Mat&  a_dmin_u,
    const Mat&  a_dmax_u,
    int         a_dim_d,
    int         a_s_hat,
    const Mat&  a_edge_confidence_u,
    const Mat&  a_edge_confidence_mask_u,
    Mat&        a_disp_confidence_u,
    Mat&        a_best_depth_u,
    Mat&        a_rbar_u,
    const Depth1DParameters<DataType>&  a_parameters,
    BufferDepth1D<DataType>&            a_buffer,
    Mat&        a_mask_u
) 
{
    // Dimensions
    int dim_s = a_epi.rows;
    
    // Init score matrix
    Mat scores_u_d = a_buffer.buf_scores_u_d;
    
    /*
     * Iterate over all columns of the EPI
     */
    
    // Get indices u to compute
    // Do not compute low-confidence values or value masked
    if (!a_mask_u.empty())
        cv::bitwise_and(a_edge_confidence_mask_u, a_mask_u, a_mask_u);
    else
        a_mask_u = a_edge_confidence_mask_u;
    
    Vec<cv::Point> locations = a_buffer.buf_locations;
    cv::findNonZero(a_mask_u, locations);
    
    for (auto it = locations.begin(); it<locations.end(); it++)
    {
        int u = (*it).x;
        
        /*
         * Fill radiances
         */
        // Matrix of indices corresponding to the lines of disparities d
        Mat I = a_buffer.buf_I;
        Mat S = a_buffer.buf_S;
        Mat D = a_buffer.buf_D;
        
        // Col matrix
        for (int s=0; s<dim_s; s++)
        {
            S.at<float>(s) = a_s_hat - s;
        }
        
        float dmin = a_dmin_u.at<float>(u);
        float dmax = a_dmax_u.at<float>(u);
        for (int d=0; d<a_dim_d; d++)
            D.at<float>(d) = dmin + d * (dmax - dmin) / (a_dim_d-1);
        
        I = S * D;
        I *= a_parameters.par_slope_factor;
        I += u;
        
        // Radiances
        Mat radiances_s_d       = a_buffer.buf_radiances_s_d;
        Mat radiances_s_d_un_nanified = a_buffer.buf_radiances_s_d_un_nanified;
        
        // Interpolate
        // TODO this step is costly
        Mat card_R              = a_buffer.buf_card_R;
        a_parameters.par_interpolation_class->interpolate_mat(a_epi, I, radiances_s_d, card_R);
        
        /*
         * Compute r_bar iteratively
         */
        Mat r_bar               = a_buffer.buf_r_bar;
        Mat r_m_r_bar           = a_buffer.buf_r_m_r_bar;
        Mat K_r_m_r_bar_mat     = a_buffer.buf_K_r_m_r_bar_mat;
        Mat K_r_m_r_bar_mat_vec = a_buffer.buf_K_r_m_r_bar_mat_vec;
        Mat r_K_r_m_r_bar_mat   = a_buffer.buf_r_K_r_m_r_bar_mat;
        Mat sum_r_K_r_m_r_bar   = a_buffer.buf_sum_r_K_r_m_r_bar;
        Mat sum_K_r_m_r_bar     = a_buffer.buf_sum_K_r_m_r_bar;
        Mat sum_K_r_m_r_bar_vec = a_buffer.buf_sum_K_r_m_r_bar_vec;
        Mat r_bar_broadcast     = a_buffer.buf_r_bar_broadcast;
        
        // Initialize r_bar to the values in s_hat
        radiances_s_d.row(a_s_hat).copyTo(r_bar);
        
        // Replace nans with zeros (since these will be multiplied by zero anyway)
        cv::max(radiances_s_d, cv::Scalar(0.0), radiances_s_d_un_nanified);
        
        // Perform a partial mean shift to compute r_bar
        // TODO: This step is costly
        for (int i=0; i< a_parameters.par_mean_shift_max_iter; i++)
        {
            // r_bar repeated over lines 
            cv::repeat(r_bar, dim_s, 1, r_bar_broadcast); 
            
            // r - r_bar
            // This matrix contains nans
            cv::subtract(radiances_s_d, r_bar_broadcast, r_m_r_bar);
            
            // K(r - r_bar)
            // Kernel fuction returns 0 if value is nan
            a_parameters.par_kernel_class->evaluate_mat(r_m_r_bar, K_r_m_r_bar_mat); 
            
            // r * K(r - r_bar)
            // Multiply should be of the same number of channels
            _multiply_multi_channel<DataType>(K_r_m_r_bar_mat, radiances_s_d_un_nanified, r_K_r_m_r_bar_mat, K_r_m_r_bar_mat_vec);
            
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
        a_parameters.par_kernel_class->evaluate_mat(r_m_r_bar, K_r_m_r_bar_mat);
        cv::reduce(K_r_m_r_bar_mat, sum_K_r_m_r_bar, 0, cv::REDUCE_SUM);
        
        // Get score
        cv::divide(sum_K_r_m_r_bar, card_R, sum_K_r_m_r_bar);
        // Set nans to zero
        cv::max(sum_K_r_m_r_bar, cv::Scalar(0.0), sum_K_r_m_r_bar);
        
        // Copy the line to the scores
        sum_K_r_m_r_bar.copyTo(scores_u_d.row(u));
        
        /*
         * Get best d (max score)
         */
        double minVal;
        double maxVal;
        cv::Point minIdx;
        cv::Point maxIdx;
        cv::minMaxLoc(scores_u_d.row(u), &minVal, &maxVal, &minIdx, &maxIdx);
        
        a_best_depth_u.at<float>(u) = D.at<float>(maxIdx.x);
        
        // Compute depth confidence score
        a_disp_confidence_u.at<float>(u) = a_edge_confidence_u.at<float>(u) * std::abs(maxVal - cv::mean(scores_u_d.row(u))[0]);
        
        // Get final r_bar
        a_rbar_u.at<DataType>(u) = r_bar.at<DataType>(maxIdx.x);
        
    }
    
}

template<typename DataType>
void selective_median_filter(
    const Mat&  a_src,
    Mat&        a_dst,
    const Vec<Mat>& a_epis,
    int         a_s_hat,
    int         a_size,
    const Mat&  a_mask_v_u,
    float       a_epsilon
)
{
    int dim_v = a_src.rows;
    int dim_u = a_src.cols;
    
    // Allocate matrix if not allocated yet
    if (a_dst.empty() || a_dst.size != a_src.size || a_dst.type() != a_src.type())
        a_dst = Mat(dim_v, dim_u, a_src.type(), cv::Scalar(0.0));
    
    int thr_max = omp_get_max_threads();
    Vec<Vec<float> > value_buffers;
    for (int t=0; t<thr_max; t++)
        value_buffers.push_back(Vec<float>());
    
    int width = (a_size-1)/2;
    
#pragma omp parallel for
    for (int v=0; v<dim_v; v++)
    {
        Vec<float> buffer = value_buffers[omp_get_thread_num()];
        for (int u=0; u<dim_u; u++)
        {
            // If mask is null, skip
            if (a_mask_v_u.at<uchar>(v, u))
            {
                buffer.clear();
                for (int k=std::max(0, v-width); k<std::min(dim_v, v+width+1); k++)
                {
                    for (int l=std::max(0, u-width); l<std::min(dim_u, u+width+1); l++)
                    {
                        if (a_mask_v_u.at<uchar>(k, l) && 
                            norm(
                                a_epis[v].at<DataType>(a_s_hat, u) - 
                                a_epis[k].at<DataType>(a_s_hat, l)
                                ) < a_epsilon)
                        {
                            buffer.push_back(a_src.at<float>(k, l));
                        }
                    }
                }
                // Compute the median
                std::nth_element(buffer.begin(), buffer.begin() + buffer.size() / 2, buffer.end());
                a_dst.at<float>(v, u) = buffer[buffer.size() / 2];
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
    const Vec<Mat>& a_epis,
    int             a_s,
    Mat&            a_edge_confidence_v_u,
    Mat&            a_edge_confidence_mask_v_u,
    const Depth1DParameters<DataType>&  a_parameters,
    Vec<BufferDepth1D<DataType>* >&     a_buffers
)
{
    int dim_v = a_epis.size();
    
    a_edge_confidence_mask_v_u = Mat(a_edge_confidence_v_u.rows, a_edge_confidence_v_u.cols, CV_8UC1);
    
    // Compute edge confidence for all rows
#pragma omp parallel for
    for (int v=0; v<dim_v; v++)
    {
        Mat edge_confidence_u = a_edge_confidence_v_u.row(v);
        Mat edge_confidence_mask_u = a_edge_confidence_mask_v_u.row(v);
        
        compute_1D_edge_confidence<DataType>(
            a_epis[v], 
            a_s, 
            edge_confidence_u, 
            edge_confidence_mask_u, 
            a_parameters, 
            *a_buffers[omp_get_thread_num()]
        );
    }
    
    if (a_parameters.par_edge_confidence_opening_size > 1)
    {
        // Perform a morphological opening on the result
        Mat kernel = cv::getStructuringElement
        (
            a_parameters.par_edge_confidence_opening_type,
            cv::Size(a_parameters.par_edge_confidence_opening_size, a_parameters.par_edge_confidence_opening_size)
        );
        
        cv::morphologyEx(a_edge_confidence_mask_v_u, a_edge_confidence_mask_v_u, cv::MORPH_OPEN, kernel);
    }
}

template<typename DataType>
void compute_1D_depth_epi_pile(
    const Vec<Mat>& a_epis,
    const Mat&      a_dmin_v_u,
    const Mat&      a_dmax_v_u,
    int             a_dim_d,
    int             a_s_hat,
    const Mat&      a_edge_confidence_v_u,
    const Mat&      a_edge_confidence_mask_v_u,
    Mat&            a_disp_confidence_v_u,
    Mat&            a_best_depth_v_u,
    Mat&            a_rbar_v_u,
    const Depth1DParameters<DataType>&  a_parameters,
    Vec<BufferDepth1D<DataType>* >&     a_buffers,
    Mat&            a_mask_v_u,
    bool            a_verbose
)
{
    // Dimension
    int dim_v = a_epis.size();
    
    // For progress bar
    float progress = 0.0;
    int barWidth = 40;
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
#pragma omp parallel for
    for (int v=0; v<dim_v; v++)
    {
        // Create views
        Mat epi                 = a_epis[v];
        Mat dmin_u              = a_dmin_v_u.row(v);
        Mat dmax_u              = a_dmax_v_u.row(v);
        Mat edge_confidence_u   = a_edge_confidence_v_u.row(v);
        Mat edge_confidence_mask_u = a_edge_confidence_mask_v_u.row(v);
        Mat disp_confidence_u   = a_disp_confidence_v_u.row(v);
        Mat best_depth_u        = a_best_depth_v_u.row(v);
        Mat rbar_u              = a_rbar_v_u.row(v);
        
        // Empty mask -> no masked point
        Mat mask_u;
        if (!a_mask_v_u.empty())
            mask_u = a_mask_v_u.row(v);
        
        compute_1D_depth_epi<DataType>(
            epi,
            dmin_u,
            dmax_u,
            a_dim_d,
            a_s_hat,
            edge_confidence_u,
            edge_confidence_mask_u,
            disp_confidence_u,
            best_depth_u,
            rbar_u,
            a_parameters,
            *a_buffers[omp_get_thread_num()],
            mask_u
        );
        
        if (a_verbose)
        {
#pragma omp critical
{
            // Display progress bar
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::cout << "[";
            int pos = barWidth * progress / dim_v;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0 / dim_v) << "% \t" << std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count() << "s \r";
            std::cout.flush();
}
#pragma omp atomic
           progress += 1.0;
        }
    }
    
    // Apply median filter
    if (a_verbose)
        std::cout << std::endl << "Applying selective median fiter" << std::endl;
    
    Mat tmp;
    selective_median_filter<DataType>(
        a_best_depth_v_u, 
        tmp,
        a_epis,
        a_s_hat,
        a_parameters.par_median_filter_size, 
        a_edge_confidence_mask_v_u, 
        a_parameters.par_median_filter_epsilon
    );

    a_best_depth_v_u = tmp;
}

/*
 * *****************************************************************
 * IMPLEMENTATION
 * compute_2D_depth_epi
 * *****************************************************************
 */
template<typename DataType>
void compute_2D_edge_confidence(
    const Vec<Mat>& a_epis,
    Vec<Mat>&       a_edge_confidence_s_v_u,
    Vec<Mat>&       a_edge_confidence_mask_s_v_u,
    const Depth1DParameters<DataType>&  a_parameters,
    Vec<BufferDepth1D<DataType>* >&     a_buffers
)
{
    int dim_s = a_epis[0].rows;
    
    a_edge_confidence_mask_s_v_u.clear();
    
    // Compute edge confidence for all rows
    for (int s=0; s<dim_s; s++)
    {
        Mat edge_confidence_v_u = a_edge_confidence_s_v_u[s];
        Mat edge_confidence_mask_v_u;
        
        compute_1D_edge_confidence_pile(
            a_epis,
            s,
            edge_confidence_v_u,
            edge_confidence_mask_v_u,
            a_parameters,
            a_buffers
        );
        
        a_edge_confidence_mask_s_v_u.push_back(edge_confidence_mask_v_u);
    }
}
 
template<typename DataType>
void compute_2D_depth_epi(
    const Vec<Mat>&     a_epis,
    const Vec<Mat>&     a_dmin_s_v_u,
    const Vec<Mat>&     a_dmax_s_v_u,
    int                 a_dim_d,
    const Vec<Mat>&     a_edge_confidence_s_v_u,
    const Vec<Mat>&     a_edge_confidence_mask_s_v_u,
    Vec<Mat>&           a_disp_confidence_s_v_u,
    Vec<Mat>&           a_best_depth_s_v_u,
    Vec<Mat>&           a_rbar_s_v_u,
    const Depth1DParameters<DataType>&  a_parameters,
    Vec<BufferDepth1D<DataType>* >&     a_buffers,
    bool                a_verbose
)
{
    int dim_s = a_epis[0].rows;
    int dim_u = a_epis[0].cols;
    int dim_v = a_epis.size();
    int s_hat = (int)std::floor(dim_s / 2.0);
    
    // Assume edge confidence is computed on all the image
    
    // Create a mask (CV_8UC1) and init to C_e > thr
    Vec<Mat> mask_s_v_u = Vec<Mat>(dim_s);
    for (int s=0; s<dim_s; s++)
    {
        // TODO other criteria?
        mask_s_v_u[s] = a_edge_confidence_mask_s_v_u[s].clone();
    }
    
    Mat dmin_v_u;
    Mat dmax_v_u;
    Mat edge_confidence_v_u;
    Mat edge_confidence_mask_v_u;
    Mat disp_confidence_v_u;
    Mat best_depth_v_u;
    Mat rbar_v_u;
    Mat mask_v_u;
    
    Vec<int> s_values;
    s_values.push_back(s_hat);
    for (int s_offset=1; s_offset<dim_s-s_hat; s_offset++)
    {
        int s_sup = s_hat + s_offset;
        int s_inf = s_hat - s_offset;
        s_values.push_back(s_sup);
        if (s_inf > -1)
            s_values.push_back(s_inf);
    }
    
    // Iterate over lines of the epi (s) starting from the middle
    for (auto it = s_values.begin(); it < s_values.end(); it++) 
    {
        int s_hat = *it;
        
        if (a_verbose)
            std::cout << "Computing s_hat=" << s_hat << std::endl;
        
        dmin_v_u            = a_dmin_s_v_u[s_hat];
        dmax_v_u            = a_dmax_s_v_u[s_hat];
        edge_confidence_v_u = a_edge_confidence_s_v_u[s_hat];
        edge_confidence_mask_v_u = a_edge_confidence_mask_s_v_u[s_hat];
        disp_confidence_v_u = a_disp_confidence_s_v_u[s_hat];
        best_depth_v_u      = a_best_depth_s_v_u[s_hat];
        rbar_v_u            = a_rbar_s_v_u[s_hat];
        mask_v_u            = mask_s_v_u[s_hat];

        compute_1D_depth_epi_pile<DataType>(
            a_epis,
            dmin_v_u,
            dmax_v_u,
            a_dim_d,
            s_hat,
            edge_confidence_v_u,
            edge_confidence_mask_v_u,
            disp_confidence_v_u,
            best_depth_v_u,
            rbar_v_u,
            a_parameters,
            a_buffers,
            mask_v_u,
            false // verbose
        );
        
        if (a_verbose)
            std::cout << "Propagation..." << std::endl;
        
        // Propagate depths over lines and update further mask lines
        // For each column of the s_hat row, draw the line, taking overlays into account
#pragma omp parallel for
        for (int v=0; v<dim_v; v++)
        {
            Mat edge_confidence_u   = edge_confidence_v_u.row(v);
            Mat best_depth_u        = best_depth_v_u.row(v);
            Mat rbar_u              = rbar_v_u.row(v);
            for (int u=0; u<dim_u; u++)
            {
                // Only paint if the confidence threshold was high enough
#ifdef _USE_DISP_CONFIDENCE_SCORE
                if (disp_confidence_v_u.at<float>(v, u) > a_parameters.par_disp_score_threshold)
#else
                if (edge_confidence_mask_v_u.at<uchar>(v, u))
#endif
                {
                    float current_depth_value = best_depth_u.at<float>(u);
                    for (int s=0; s<dim_s; s++)
                    {
                    
                        int requested_index = u + (int)std::round(best_depth_u.at<float>(u) * (s_hat - s) * a_parameters.par_slope_factor);
                        
                        if 
                        (
                            requested_index > -1 && 
                            requested_index < dim_u && 
                            mask_s_v_u[s].at<uchar>(v, requested_index) == 255 &&
                            norm<DataType>(a_epis[v].at<DataType>(s, requested_index) - rbar_u.at<DataType>(u)) < a_parameters.par_propagation_epsilon 
                        )
                        {
                            a_best_depth_s_v_u[s].at<float>(v, requested_index) = current_depth_value;
                            mask_s_v_u[s].at<uchar>(v, requested_index) = 0;
                            a_disp_confidence_s_v_u[s].at<float>(v, requested_index) = disp_confidence_v_u.at<float>(v, u);
                        }
                    }
                }
            }
        }
        
    }
    
}

}



#endif
