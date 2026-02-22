#include <vector>

float clampf(float val) {
    if (val < 0.0f) return 0.0f;
    if (val > 255.0f) return 255.0f;
    return val;
}

extern "C" void conv3d(float** inputs, float* kernel, float* output, int width, int height, int depth, int kw, int kh, int kd, int cs)
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