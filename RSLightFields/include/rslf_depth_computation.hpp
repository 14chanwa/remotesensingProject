#ifndef _RSLF_DEPTH_COMPUTATION
#define _RSLF_DEPTH_COMPUTATION

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>


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
            //virtual float evaluate_sum(cv::Mat column) {}
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
            Interpolation1D<DataType> interpolation_function = (Interpolation1D<DataType>)interpolation_1d_nearest_neighbour,
            KernelClass<DataType>* kernel_class = 0 
        );
        ~DepthComputer1D();
        void run();
    
    private:
        cv::Mat epi_;
        std::vector<float> d_list_;
        std::vector<cv::Mat> radiances_u_s_d_;
        cv::Mat scores_u_d_;
        std::vector<float> best_depth_u_;

        Interpolation1D<DataType> interpolation_function_;
        
        /**
         * Dimension along s axis
         */
        int dim_s_;
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
        
        // Create radiance vector of matrix
        this->radiances_u_s_d_ = 
            std::vector<cv::Mat>(this->dim_u_);
        for (int u=0; u<this->dim_u_; u++)
            this->radiances_u_s_d_.at(u) =cv::Mat(this->dim_s_, this->dim_d_, CV_32FC1, cv::Scalar(0.0));
        std::cout << "created radiances_u_s_d_ of size " << this->radiances_u_s_d_.size() << "x" << this->radiances_u_s_d_.at(0).size << std::endl;
        
        this->scores_u_d_ = cv::Mat(this->dim_u_, this->dim_d_, CV_32FC1, cv::Scalar(0.0));
        
        this->best_depth_u_ = std::vector<float>(this->dim_u_);
        
        if (kernel_class == 0)
        {
            this->kernel_class_ = new BandwidthKernel<DataType>(0.2);
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
        
        // s_hat is the center horizontal line index of the epi
        int s_hat = (int) std::floor((0.0 + this->dim_s_) / 2);
        
        // Col matrix
        cv::Mat S = cv::Mat(this->dim_s_, 1, CV_32FC1);
        for (int s=0; s<this->dim_s_; s++)
            S.at<float>(s, 0) = s_hat - s;
        
        // Iterate over all columns of the EPI
        for (int u=0; u<this->dim_u_; u++) // u<1; u++)//
        {
            
            std::cout << "u=" << u << std::endl;
            
            // Index matrix
            cv::Mat I = S * D;
            I += u;
            
            // Create new radiance view
            // https://stackoverflow.com/questions/23998422/how-to-access-slices-of-a-3d-matrix-in-opencv
            cv::Mat radiances_s_d = this->radiances_u_s_d_.at(u);
            std::cout << radiances_s_d.size << std::endl;
            
            /*
             * Fill radiances
             */
#pragma omp parallel for
            for (int s=0; s<this->dim_s_; s++)
            {
#pragma omp parallel for
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
            cv::Mat card_R(1, this->dim_d_, CV_32FC1, cv::Scalar(0.0));

            for (int d=0; d<I.cols; d++)
            {
                for (int s=0; s<I.rows; s++)
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
            radiances_s_d.row(s_hat).copyTo(r_bar);
            
            // Perform a partial mean shift
            
            for (int d=0; d<dim_d_; d++)
            {
                for (int i=0; i<_MEAN_SHIFT_MAX_ITER; i++)
                {
                    DataType numerator;
                    float denominator;
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
                    r_bar.at<DataType>(0, d) = numerator / denominator;
                }
            }

            /*
             * Compute scores 
             */
            for (int d=0; d<dim_d_; d++)
            {
                // scores_u_d_
                for (int s=0; s<dim_s_; s++)
                {
                    scores_u_d_.at<float>(u, d) += this->kernel_class_->evaluate(radiances_s_d.at<DataType>(s, d) - r_bar.at<DataType>(0, d));
                }
                scores_u_d_.at<float>(u, d) /= card_R.at<float>(0, d);
            }
            
            /*
             * Get best d (max score)
             */
            double minVal;
            double maxVal;
            cv::Point minIdx;
            cv::Point maxIdx;
            cv::minMaxLoc(scores_u_d_.row(u), &minVal, &maxVal, &minIdx, &maxIdx);
            //~ std::cout << maxVal << ", " << maxIdx << std::endl;
            
            this->best_depth_u_.at(u) = (maxVal > 0 ? this->d_list_.at(maxIdx.x) : 0.0);
            
            //~ std::cout << this->best_depth_u_.at(u) << std::endl;
        }
        
        //~ std::cout << scores_u_d_ << std::endl;
        
        // DEBUG
        //~ for (int u=0; u<this->dim_u_; u++)
            //~ std::cout << this->best_depth_u_.at(u) << "\t";
        //~ std::cout << std::endl;
        
        // DEBUG
        // Try to get a 2D slice
        
        //~ cv::Mat slice = this->radiances_u_s_d_.at(1);
        
        //~ std::cout << slice << std::endl;
        
    }

}



#endif
