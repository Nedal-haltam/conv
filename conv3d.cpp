#include <vector>
#include <opencv2/opencv.hpp>

float clampf(float val) {
    if (val < 0.0f) return 0.0f;
    if (val > 255.0f) return 255.0f;
    return val;
}

extern "C" void conv3d_old(float** inputs, float* kernel, float* output, int width, int height, int depth, int kw, int kh, int kd, int cs)
{
    int padH = kh / 2;
    int padW = kw / 2;
    const int input_row_stride = width * cs;
    
    for (int y = padH; y < height - padH; ++y)
    {
        for (int x = padW; x < width - padW; ++x)
        {
            int out_idx_start = y * input_row_stride + x * cs;
            float sum[3] = {0.0f, 0.0f, 0.0f};
            
            for (int dz = 0; dz < kd; ++dz)
            {
                float* current_input_frame = inputs[dz];

                for (int dy = 0; dy < kh; ++dy)
                {
                    for (int dx = 0; dx < kw; ++dx)
                    {
                        int y_in = y + dy - padH;
                        int x_in = x + dx - padW;
                        
                        int idx_pixel = y_in * input_row_stride + x_in * cs;
                        
                        int kidx = dz * (kw * kh) + dy * kw + dx;
                        float k = kernel[kidx];

                        for (int c = 0; c < cs; ++c)
                        {
                            sum[c] += current_input_frame[idx_pixel + c] * k;
                        }
                    }
                }
            }
            
            for (int c = 0; c < cs; ++c)
            {
                output[out_idx_start + c] = clampf(sum[c]);
            }
        }
    }
}

extern "C" void conv3d(float** inputs, float* kernel, float* output, int width, int height, int depth, int kw, int kh, int kd, int cs, int tc)
{
    cv::Mat out_mat(height, width, CV_MAKETYPE(CV_32F, cs), output);
    out_mat.setTo(cv::Scalar::all(0));

    #pragma omp parallel for num_threads(tc) collapse(3)
    for (int dz = 0; dz < kd; ++dz)
    {
        cv::Mat frame(height, width, CV_MAKETYPE(CV_32F, cs), inputs[dz]);
        cv::Mat k_slice(kh, kw, CV_32F, kernel + (dz * kw * kh));
        cv::Mat filtered;
        cv::filter2D(frame, filtered, CV_32F, k_slice, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
        out_mat += filtered;
    }

    cv::threshold(out_mat, out_mat, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::threshold(out_mat, out_mat, 255.0, 255.0, cv::THRESH_TRUNC);

    int padH = kh / 2;
    int padW = kw / 2;

    if (padH > 0)
    {
        out_mat.rowRange(0, padH).setTo(cv::Scalar::all(0));
        out_mat.rowRange(height - padH, height).setTo(cv::Scalar::all(0));
    }
    if (padW > 0)
    {
        out_mat.colRange(0, padW).setTo(cv::Scalar::all(0));
        out_mat.colRange(width - padW, width).setTo(cv::Scalar::all(0));
    }
}