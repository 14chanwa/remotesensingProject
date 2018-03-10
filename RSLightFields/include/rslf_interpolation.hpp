#ifndef _RSLF_INTERPOLATION
#define _RSLF_INTERPOLATION

#include <rslf_types.hpp>


namespace rslf
{
    
    /*
     * *****************************************************************
     * INTERPOLATION CLASSES
     * *****************************************************************
     */
    
    /**
     * Template generic 1D interpolation class.
     */
    template<typename DataType>
    class Interpolation1DClass
    {
        public:
            virtual DataType interpolate(const Mat& line_matrix, float index) = 0;
            virtual void interpolate_mat(const Mat& data_matrix, const Mat& indices, Mat& res) = 0;
    };

    /**
     * Template nearest neighbour interpolation class.
     */ 
    template<typename DataType> 
    class Interpolation1DNearestNeighbour : public Interpolation1DClass<DataType>
    {
        public:
            Interpolation1DNearestNeighbour() {}
            DataType interpolate(const Mat& line_matrix, float index);
            void interpolate_mat(const Mat& data_matrix, const Mat& indices, Mat& res);
    };
    
    /**
     * Template linear interpolation class.
     */ 
    template<typename DataType> 
    class Interpolation1DLinear : public Interpolation1DClass<DataType>
    {
        public:
            Interpolation1DLinear() {}
            DataType interpolate(const Mat& line_matrix, float index);
            void interpolate_mat(const Mat& data_matrix, const Mat& indices, Mat& res);
    };
    
    
    
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
    
    
    
    /*
     * *****************************************************************
     * IMPLEMENTATION
     * Interpolation
     * *****************************************************************
     */
    
    template<typename DataType>
    DataType Interpolation1DNearestNeighbour<DataType>::interpolate
    (
        const Mat& line_matrix, 
        float index
    )
    {
        int rounded_index = (int)std::round(index);
        if (rounded_index < 0 || rounded_index > line_matrix.cols - 1)
            return nan_type<DataType>();
        return line_matrix.at<DataType>(0, rounded_index);
    }
    
    template<typename DataType>
    void Interpolation1DNearestNeighbour<DataType>::interpolate_mat
    (
        const Mat& data_matrix, 
        const Mat& indices,
        Mat& res
    )
    {
        // TODO is there a better way to vectorize?
        assert(indices.rows == data_matrix.rows);
        if (res.empty() || res.size != data_matrix.size || res.type() != data_matrix.type())
            res = cv::Mat::zeros(indices.rows, indices.cols, data_matrix.type());
        
        // Round indices
        Mat round_indices_matrix;
        indices.convertTo(round_indices_matrix, CV_32SC1, 1.0, 0.0);
        // For each row
        for (int r=0; r<indices.rows; r++) {
            const DataType* data_ptr = data_matrix.ptr<DataType>(r);
            DataType* res_ptr = res.ptr<DataType>(r);
            int* ind_ptr = round_indices_matrix.ptr<int>(r);
            // For each col
            for (int c=0; c<indices.cols; c++) {
                res_ptr[c] = (ind_ptr[c] > -1 && ind_ptr[c] < data_matrix.cols ? data_ptr[ind_ptr[c]] : nan_type<DataType>());
            }
        }
    }

    template<typename DataType>
    DataType Interpolation1DLinear<DataType>::interpolate
    (
        const Mat& line_matrix, 
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
    void Interpolation1DLinear<DataType>::interpolate_mat
    (
        const Mat& data_matrix, 
        const Mat& indices,
        Mat& res
    )
    {
        // TODO is there a better way to vectorize?
        assert(indices.rows == data_matrix.rows);
        if (res.empty() || res.size != data_matrix.size || res.type() != data_matrix.type())
            res = cv::Mat::zeros(indices.rows, indices.cols, data_matrix.type());
        
        // For each row
        for (int r=0; r<indices.rows; r++) {
            const DataType* data_ptr = data_matrix.ptr<DataType>(r);
            DataType* res_ptr = res.ptr<DataType>(r);
            const float* ind_ptr = indices.ptr<float>(r);
            // For each col
            for (int c=0; c<indices.cols; c++) {
                int ind_i = (int)std::floor(ind_ptr[c]);
                int ind_s = (int)std::ceil(ind_ptr[c]);
                float ind_residue = ind_ptr[c] - ind_i;
                res_ptr[c] = (ind_i <= 0 || ind_s >= data_matrix.cols - 1 ?
                    nan_type<DataType>() : (1-ind_residue)*data_ptr[ind_i] + ind_residue*data_ptr[ind_s]);
            }
        }
    }
    
}


#endif
