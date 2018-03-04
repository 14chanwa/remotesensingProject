#ifndef _RSLF_DEPTH_COMPUTATION
#define _RSLF_DEPTH_COMPUTATION

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rslf_plot.hpp>
#include <rslf_utils.hpp>


#define _MEAN_SHIFT_MAX_ITER 10


namespace rslf
{
    
    /**
     * Template generic interpolation function.
     * 
     * @param line_matrix A matrix with the values taken at points 0..max_index-1
     * @param index The (float) index at which to compute the value.
     */
    template<typename DataType>
    using Interpolation1D = DataType (*)
    (
        cv::Mat line_matrix, 
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
        cv::Mat line_matrix, 
        float index
    );
    
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
            BandwidthKernel(float h);
            float evaluate(DataType x);
        private:
            float h_;
    };

    /**
     * Template class with depth computation using 1d slices of the EPI.
     */
    template<typename DataType>
    class DepthComputer1D
    {
    public:
        DepthComputer1D
        (
            cv::Mat epi, 
            std::vector<float> d_list,
            Interpolation1D<DataType> interpolation_function = (Interpolation1D<DataType>)interpolation_1d_nearest_neighbour<DataType>,
            KernelClass<DataType>* kernel_class = 0 
        );
        ~DepthComputer1D();
        
        void run();
        cv::Mat get_coloured_epi();
    
    private:
        cv::Mat epi_;
        std::vector<float> d_list_;
        
        std::vector<cv::Mat> radiances_u_s_d_;
        cv::Mat scores_u_d_;
        
        cv::Mat best_depth_u_;
        cv::Mat score_depth_u_;

        Interpolation1D<DataType> interpolation_function_;
        
        /**
         * Dimension along s axis
         */
        int dim_s_;
        int s_hat_;
        /**
         * Dimension along d axis
         */
        int dim_d_;
        /**
         * Dimension along u axis
         */
        int dim_u_;
        
        KernelClass<DataType>* kernel_class_;
        bool delete_kernel_on_delete_;
    };
    
    
    /*
     * Template function implementations
     */
    
    template<typename DataType>
    DepthComputer1D<DataType>::DepthComputer1D
    (
        cv::Mat epi, 
        std::vector<float> d_list,
        Interpolation1D<DataType> interpolation_function,
        KernelClass<DataType>* kernel_class
    ) 
    {
        this->epi_ = epi;
        this->d_list_ = d_list;
        
        this->interpolation_function_ = interpolation_function;
        
        // Build radiance matrix 
        this->dim_s_ = this->epi_.rows;
        this->dim_d_ = this->d_list_.size();
        this->dim_u_ = this->epi_.cols;
        
        // s_hat is the center horizontal line index of the epi
        this->s_hat_ = (int) std::floor((0.0 + this->dim_s_) / 2);
        
        // Create radiance vector of matrix
        this->radiances_u_s_d_ = 
            std::vector<cv::Mat>(this->dim_u_);
        //~ std::cout << this->radiances_u_s_d_.size() << std::endl;
        for (int u=0; u<this->dim_u_; u++)
            this->radiances_u_s_d_[u] =cv::Mat(this->dim_s_, this->dim_d_, CV_32FC1, cv::Scalar(0.0));
        std::cout << "created radiances_u_s_d_ of size " << this->radiances_u_s_d_.size() << " x " << this->radiances_u_s_d_.at(0).size << std::endl;
        
        this->scores_u_d_ = cv::Mat(this->dim_u_, this->dim_d_, CV_32FC1);
        
        this->best_depth_u_ = cv::Mat(1, this->dim_u_, CV_32FC1);
        this->score_depth_u_ = cv::Mat(1, this->dim_u_, CV_32FC1);
        
        if (kernel_class == 0)
        {
            this->kernel_class_ = new BandwidthKernel<DataType>(0.2); // Defaults to a bandwidth kernel with h=0.2
            this->delete_kernel_on_delete_ = true;
        }
        else
        {
            this->kernel_class_ = kernel_class;
            this->delete_kernel_on_delete_ = false;
        }
    }
    
    template<typename DataType>
    DepthComputer1D<DataType>::~DepthComputer1D() {
        if (this->delete_kernel_on_delete_)
            delete kernel_class_;
    }

    template<typename DataType>
    void DepthComputer1D<DataType>::run() 
    {
        
        // Row matrix
        cv::Mat D = cv::Mat(1, this->dim_d_, CV_32FC1);
        for (int d=0; d<this->dim_d_; d++)
            D.at<float>(0, d) = d_list_[d];
        
        // Col matrix
        cv::Mat S = cv::Mat(this->dim_s_, 1, CV_32FC1);
        for (int s=0; s<this->dim_s_; s++)
            S.at<float>(s, 0) = this->s_hat_ - s;
        
        // Index matrix
        cv::Mat indices = S * D;
        
        // Iterate over all columns of the EPI
        for (int u=0; u<this->dim_u_; u++) // u<1; u++)//
        {
            
            std::cout << "u=" << u << std::endl;
            
            cv::Mat I;
            indices.copyTo(I);
            I += u;
            
            // Create new radiance view
            // https://stackoverflow.com/questions/23998422/how-to-access-slices-of-a-3d-matrix-in-opencv
            cv::Mat radiances_s_d = this->radiances_u_s_d_[u];
            std::cout << radiances_s_d.size << std::endl;
            
            /*
             * Fill radiances
             */
//~ #pragma omp parallel for
            for (int s=0; s<this->dim_s_; s++)
            {
//~ #pragma omp parallel for
                for (int d=0; d<this->dim_d_; d++)
                {
                    radiances_s_d.at<float>(s, d) =
                        this->interpolation_function_
                        (
                            this->epi_.row(s), 
                            I.at<float>(s, d)
                        );
                }
            }
            
            // Compute number of non nan radiances per column
            cv::Mat card_R(1, this->dim_d_, CV_32FC1);

            for (int d=0; d<this->dim_d_; d++)
            {
                for (int s=0; s<this->dim_s_; s++)
                {
                    if (radiances_s_d.at<float>(s, d) == radiances_s_d.at<float>(s, d))
                        card_R.at<float>(0, d) += 1.0;
                }
            }
            
            
            /*
             * Compute r_bar iteratively
             */
            
            // Initialize r_bar to the values in s_hat
            cv::Mat r_bar;
            radiances_s_d.row(this->s_hat_).copyTo(r_bar);
            
            // Perform a partial mean shift
            for (int i=0; i<_MEAN_SHIFT_MAX_ITER; i++)
            {
                for (int d=0; d<dim_d_; d++)
                {
                    DataType numerator = 0.0;
                    float denominator = 0.0;
                    for (int s=0; s<dim_s_; s++)
                    {
                        if (radiances_s_d.at<DataType>(s, d) == radiances_s_d.at<DataType>(s, d))
                        {
                            // Compute K(r - r_bar)
                            float kernel_r_m_r_bar = 
                                this->kernel_class_->evaluate(radiances_s_d.at<DataType>(s, d) - r_bar.at<DataType>(0, d));
                            // sum(r * K(r - r_bar))
                            numerator += 
                                radiances_s_d.at<DataType>(s, d) * kernel_r_m_r_bar;
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
            for (int d=0; d<dim_d_; d++)
            {
                if (card_R.at<float>(0, d) > 0)
                {
                    // scores_u_d_
                    for (int s=0; s<dim_s_; s++)
                    {
                        scores_u_d_.at<float>(u, d) += this->kernel_class_->evaluate(radiances_s_d.at<DataType>(s, d) - r_bar.at<DataType>(0, d));
                    }
                    scores_u_d_.at<float>(u, d) /= card_R.at<float>(0, d);
                }
                else
                {
                    scores_u_d_.at<float>(u, d) = 0.0;
                }
            }
            
            /*
             * Get best d (max score)
             */
            double minVal;
            double maxVal;
            cv::Point minIdx;
            cv::Point maxIdx;
            cv::minMaxLoc(scores_u_d_.row(u).clone(), &minVal, &maxVal, &minIdx, &maxIdx);
            std::cout << maxVal << ", " << maxIdx << std::endl;
            
            // TODO score threshold?
            // Test : 0.02
            this->best_depth_u_.at<float>(0, u) = (maxVal > 0.02 ? this->d_list_.at(maxIdx.x) : 0.0);
            this->score_depth_u_.at<float>(0, u) = maxVal;
            
            //~ std::cout << this->best_depth_u_.at(u) << std::endl;
        }
        
        // Apply a median filter on the depths... TODO brutal
        //~ cv::Mat tmp;
        //~ this->score_depth_u_.copyTo(tmp);
        cv::medianBlur(this->score_depth_u_.clone(), this->score_depth_u_, 5);
        
        //~ std::cout << scores_u_d_ << std::endl;
        
        // DEBUG
        //~ std::cout << this->best_depth_u_ << std::endl;
        //~ std::cout << this->best_depth_u_.size << std::endl;
        
        // DEBUG
        // Try to get a 2D slice
        
        //~ cv::Mat slice = this->radiances_u_s_d_.at(1);
        
        //~ std::cout << slice << std::endl;
        
    }
    
    template<typename DataType>
    cv::Mat DepthComputer1D<DataType>::get_coloured_epi() {
        
        // Get point color
        cv::Mat depth_map_with_occlusions(this->dim_s_, this->dim_u_, CV_32FC1, -std::numeric_limits<float>::infinity());
        cv::Mat coloured_depth = rslf::copy_and_scale_uchar(this->best_depth_u_);
        cv::applyColorMap(coloured_depth, coloured_depth, cv::COLORMAP_JET);
        
        //~ std::cout << coloured_depth << std::endl;
        //~ std::cout << "Image size " << coloured_depth.size <<
            //~ rslf::type2str(coloured_depth.type()) << std::endl;
        
        // Construct an EPI with overlay
        cv::Mat coloured_epi(this->epi_.rows, this->epi_.cols, CV_8UC3);
        //~ std::cout << "Image size " << coloured_epi.size <<
            //~ rslf::type2str(coloured_epi.type()) << std::endl;
        
        for (int u=0; u<this->dim_u_; u++)
        {
            float current_depth_value = this->best_depth_u_.at<float>(0, u);
            for (int s=0; s<this->dim_s_; s++)
            {
                int requested_index = u + (int)std::round(this->best_depth_u_.at<float>(0, u) * (this->s_hat_ - s));
                if 
                (
                    requested_index > 0 && 
                    requested_index < this->dim_u_ && 
                    depth_map_with_occlusions.at<float>(s, requested_index) < current_depth_value // handle occlusions
                )
                {
                    coloured_epi.at<cv::Vec3b>(s, requested_index) = coloured_depth.at<cv::Vec3b>(0, u);
                    depth_map_with_occlusions.at<float>(s, requested_index) = current_depth_value;
                }
            }
        }
        
        return coloured_epi;
    }

}



#endif
