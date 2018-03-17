#include <rslf_fine_to_coarse.hpp>


#define _GAUSSIAN_KSIZE 7


/*
 * *****************************************************************
 * IMPLEMENTATION
 * downsample_EPIs
 * *****************************************************************
 */
void rslf::downsample_EPIs(const Vec<Mat>& in_epis, Vec<Mat>& out_epis)
{
    out_epis.clear();
    
    int m_dim_s_ = in_epis[0].rows;
    int m_dim_u_ = in_epis[0].cols;
    int m_dim_v_ = in_epis.size();
    
    int dtype = in_epis[0].type();
    
    // Get a copy of the output epis in the dimensions s, v, u
    Vec<Mat> images_s_v_u_;
    Mat tmp = Mat(m_dim_v_, m_dim_u_, dtype, cv::Scalar(0.0));
    Mat tmp2;
    for (int s=0; s<m_dim_s_; s++)
    {
        Mat tmp3;
        for (int v=0; v<m_dim_v_; v++)
        {
            in_epis[v].row(s).copyTo(tmp.row(v));
        }
        // Filter by a gaussian filter along the spatial dimensions
        // TODO: better border condition useful?
        cv::GaussianBlur(tmp, tmp2, cv::Size(_GAUSSIAN_KSIZE, _GAUSSIAN_KSIZE), 0, 0, cv::BORDER_REFLECT);
        
        // Subsample by a factor 2 (since we already filtered, nn should be ok)
        cv::resize(tmp2, tmp3, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
            
        // Store the result
        images_s_v_u_.push_back(tmp3);
    }
    
    // Rearrange lines to out_epis
    for (int v=0; v<m_dim_v_; v++)
    {
        Mat tmp = Mat(m_dim_s_, m_dim_u_, dtype, cv::Scalar(0.0));
        for (int s=0; s<m_dim_s_; s++)
        {
            images_s_v_u_[s].row(v).copyTo(tmp.row(s));
        }
        out_epis.push_back(tmp);
    }
}


/*
 * *****************************************************************
 * IMPLEMENTATION
 * downsample_EPIs
 * *****************************************************************
 */
void rslf::fuse_disp_maps(const Vec<Vec<Mat >>& disp_pyr_p_s_v_u_, const Vec<Vec<Mat> >& validity_indicators_p_s_v_u_, Vec<Mat>& out_map_s_v_u_, Vec<Mat>& out_validity_s_v_u_)
{
    int m_dim_u_ = disp_pyr_p_s_v_u_[0][0].cols;
    int m_dim_v_ = disp_pyr_p_s_v_u_[0][0].rows;
    int m_dim_s_ = disp_pyr_p_s_v_u_[0].size();
    
    int m_pyr_size = disp_pyr_p_s_v_u_.size();
    
    out_map_s_v_u_.clear();
    out_validity_s_v_u_.clear();
    
    for (int s=0; s<m_dim_s_; s++)
    {
        Vec<Mat> disp_pyr_v_u_ = Vec<Mat>();
        Vec<Mat> mask_pyr_v_u_ = Vec<Mat>();
        for (int p=0; p<m_pyr_size; p++)
        {
            disp_pyr_v_u_.push_back(disp_pyr_p_s_v_u_[p][s]);
            mask_pyr_v_u_.push_back(validity_indicators_p_s_v_u_[p][s]);
        }
    
        Mat tmp_map_down = disp_pyr_v_u_[m_pyr_size-1];
        Mat tmp_mask_down = mask_pyr_v_u_[m_pyr_size-1];
        Mat tmp_map_up;
        Mat tmp_mask_up;
        
        for (int p = m_pyr_size-1; p > 0; p--)
        {
            
            Mat upscaled_disp;
            
            // Upscale
            // TODO: interpolation type?
            cv::resize(tmp_map_down, tmp_map_up, disp_pyr_v_u_[p-1].size(), 0, 0, cv::INTER_LINEAR);
            cv::resize(tmp_mask_down, tmp_mask_up, mask_pyr_v_u_[p-1].size(), 0, 0, cv::INTER_NEAREST);
            
            // Fill unknown points 
            
            // TODO violent... in the article, suggested to put boundaries on downscaled disp calculation
            // did not yet implement this since the structure of the code would be quite modified by this
            
            // One could refine the way the unknown disp are affected, for instance by checking the bounds
            // on the horizontal lines as described in the article
            tmp_map_down = disp_pyr_v_u_[p-1].clone();
            tmp_map_down.setTo(tmp_map_up, mask_pyr_v_u_[p-1] == 0);
            
            cv::bitwise_or(mask_pyr_v_u_[p-1], tmp_mask_up, tmp_mask_down);
        }
        out_map_s_v_u_.push_back(tmp_map_down);
        out_validity_s_v_u_.push_back(tmp_mask_down);
    }
}
