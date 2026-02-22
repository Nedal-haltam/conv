#include <chrono>
#include <iostream>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <thread>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "fftw-3.3.10/fftw-3.3.10/api/fftw3.h"
#include "conv3d.cpp"

using namespace cv;

typedef struct {
    int width;
    int height;
    int max_val;
    int channels;
    unsigned char* data;
} PPMImage;

#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define KERNEL_DEPTH 3
#define THREAD_COUNT 12


int KERNEL2D_EDGE_DETECTOR[KERNEL_HEIGHT][KERNEL_WIDTH] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1},
};

int KERNEL2D_IDENTITY[KERNEL_HEIGHT][KERNEL_WIDTH] = {
    {0, 0, 0},
    {0, 1, 0},
    {0, 0, 0},
};
auto k2d = KERNEL2D_EDGE_DETECTOR;

float KERNEL3D_EDGE_DETECTOR[KERNEL_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH] = {
    {
        {-1.0f, -2.0f, -1.0f},
        {-2.0f, -4.0f, -2.0f},
        {-1.0f, -2.0f, -1.0f}
    },
    {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    },
    {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f}
    }
};
auto k3d = KERNEL3D_EDGE_DETECTOR;

// misc funcs
clock_t Clocker;
void StartClock()
{
    Clocker = clock();
}
double EndClock(bool Verbose = false)
{
    clock_t t = clock() - Clocker;
    double TimeTaken = (double)(t) / CLOCKS_PER_SEC;
    if (Verbose)
    {
        std::cout << "Time taken: " << std::fixed << std::setprecision(8) << TimeTaken << "s\n";
    }
    std::cout.unsetf(std::ios::fixed);
    std::cout.precision(6);
    return TimeTaken;
}

unsigned char clamp(int val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return (unsigned char)val;
}

void PadImage(PPMImage& img, int padH, int padW, bool PadZero)
{
    if (PadZero)
    {
        // pad with zeros on the border
        int newWidth = img.width + 2 * padW;
        int newHeight = img.height + 2 * padH;
        unsigned char* newData = (unsigned char*)malloc(newWidth * newHeight * img.channels);
        memset(newData, 0, newWidth * newHeight * img.channels);
        for (int y = 0; y < img.height; y++) {
            for (int x = 0; x < img.width; x++) {
                for (int c = 0; c < img.channels; c++) {
                    newData[img.channels * ((y + padH) * newWidth + (x + padW)) + c] = img.data[img.channels * (y * img.width + x) + c];
                }
            }
        }
    }
    else
    {
        int newWidth = img.width + 2 * padW;
        int newHeight = img.height + 2 * padH;
        unsigned char* newData = (unsigned char*)malloc(newWidth * newHeight * img.channels);
        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                int srcX = std::clamp(x - padW, 0, img.width - 1);
                int srcY = std::clamp(y - padH, 0, img.height - 1);
                for (int c = 0; c < img.channels; c++) {
                    newData[img.channels * (y * newWidth + x) + c] = img.data[img.channels * (srcY * img.width + srcX) + c];
                }
            }
        }
        free(img.data);
        img.data = newData;
        img.width = newWidth;
        img.height = newHeight;
    }
}

template<class T, class V>
void conv1d(T* a, int alen, V* b, int blen, T* c)
{
    for (size_t i = 0; i < alen; i++) {
        for (size_t j = 0; j < blen; j++) {
            c[i + j] += a[i] * b[j];
        }
    }
}
template<class T, class V>
void conv2d(T* input, V k[KERNEL_HEIGHT][KERNEL_WIDTH], T* output, int w, int h, int marginx, int marginy, int channels, bool ClampAndAbs)
{
    for (int y = marginy; y < h - marginy; y++) {
        for (int x = marginx; x < w - marginx; x++) {
            std::vector<V>cs(channels);
            for (int ky = 0; ky < KERNEL_HEIGHT; ky++) {
                for (int kx = 0; kx < KERNEL_WIDTH; kx++) {
                    int px = x + kx - (KERNEL_WIDTH / 2);
                    int py = y + ky - (KERNEL_HEIGHT / 2);
                    if (!(px < 0 || px >= w || py < 0 || py >= h))
                    {
                        V e = k[ky][kx];
                        for (int i = 0; i < channels; i++)
                            cs[i] += input[(channels * (py * w + px)) + i] * e;
                    }
                }
            }
            for (int i = 0; i < channels; i++)
                output[(channels * ((y - marginy) * w + (x - marginx))) + i] = (!ClampAndAbs) ? cs[i] : clamp(abs(cs[i]));
        }
    }
}
template<class T, class V>
void conv3d(T* input, V k[KERNEL_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH], T* output, int w, int h, int d, int marginx, int marginy, int marginz, int channels, bool ClampAndAbs)
{
    for (int z = marginz; z < d - marginz; z++) {
        for (int y = marginy; y < h - marginy; y++) {
            for (int x = marginx; x < w - marginx; x++) {
                std::vector<V>cs(channels);
                for (int kz = 0; kz < KERNEL_DEPTH; kz++) {
                    for (int ky = 0; ky < KERNEL_HEIGHT; ky++) {
                        for (int kx = 0; kx < KERNEL_WIDTH; kx++) {
                            int px = x + kx - (KERNEL_WIDTH / 2);
                            int py = y + ky - (KERNEL_HEIGHT / 2);
                            int pz = z + kz - (KERNEL_DEPTH / 2);
                            if (!(px < 0 || px >= w || py < 0 || py >= h || pz < 0 || pz >= h))
                            {
                                V e = k[kz][ky][kx];
                                for (int i = 0; i < channels; i++)
                                    cs[i] += input[(channels * (pz * w * h + py * w + px)) + i] * e;
                            }
                        }
                    }
                }
                for (int i = 0; i < channels; i++)
                    output[(channels * ((z - marginz) * w * h + (y - marginy) * w + (x - marginx))) + i] = (!ClampAndAbs) ? cs[i] : clamp(abs(cs[i]));
            }
        }
    }
}
template<class T>
void ImageConvKernel(PPMImage& input, T* k, PPMImage& output)
{
    int padH = KERNEL_HEIGHT / 2;
    int padW = KERNEL_WIDTH / 2;
    int cs = input.channels;
    PadImage(input, padH, padW, false);
    output.width = input.width;
    output.height = input.height;
    free(output.data);
    output.data = (unsigned char*)malloc(output.width * output.height * cs);
    memset(output.data, 0, output.width * output.height * cs);

    conv2d(input.data, k, output.data, input.width, input.height, padW, padH, cs, true);

    // unpad the output PPMImage
    PPMImage unpadded;
    unpadded.width = output.width - 2 * padW;
    unpadded.height = output.height - 2 * padH;
    unpadded.max_val = output.max_val;
    unpadded.channels = cs;
    unpadded.data = (unsigned char*)malloc(unpadded.width * unpadded.height * cs);
    for (int y = 0; y < unpadded.height; y++) {
        for (int x = 0; x < unpadded.width; x++) {
            for (int c = 0; c < cs; c++) {
                unpadded.data[cs * (y * unpadded.width + x) + c] =
                    output.data[cs * ((y) * output.width + (x)) + c];
            }
        }
    }
    free(output.data);
    output = unpadded;
}
// PPM functions
PPMImage* read_ppm(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("Cannot open file");
        return NULL;
    }

    char format[3];
    if (fscanf(fp, "%2s", format) != 1 || strcmp(format, "P6") != 0) {
        std::cout << "Unsupported PPM format. Only P6 is supported.\n";
        fclose(fp);
        return NULL;
    }

    PPMImage* img = (PPMImage*)malloc(sizeof(PPMImage));
    img->channels = 3;

    
    int c;
    while ((c = fgetc(fp)) == '#') {
        while (fgetc(fp) != '\n');
    }
    ungetc(c, fp);

    auto ret = fscanf(fp, "%d %d %d", &img->width, &img->height, &img->max_val);
    fgetc(fp);  

    int size = img->width * img->height * img->channels;
    img->data = (unsigned char*)malloc(size);

    ret = fread(img->data, 1, size, fp);
    fclose(fp);
    return img;
}
int write_ppm(const char* filename, PPMImage* img) {

    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Cannot write to file");
        return 0;
    }
    fprintf(fp, "P6\n%d %d\n%d\n", img->width, img->height, img->max_val);
    fwrite(img->data, 1, img->width * img->height * img->channels, fp);
    fclose(fp);
    return 1;
}
void WriteColoredPPM(const char* filename, int width, int height, uint32_t color) {
    PPMImage img;
    img.width = width;
    img.height = height;
    img.max_val = 255;
    img.channels = 3;
    img.data = (unsigned char*)malloc(width * height * img.channels);
    
    for (int i = 0; i < width * height; i++) {
        for (int j = 0; j < img.channels; j++) {
            img.data[img.channels * i + j] = (color >> (8 * (img.channels - j))) & 0xFF;
        }
    }

    if (!write_ppm(filename, &img)) {
        std::cerr << "Failed to write random PPM image.\n";
    }
    
    free(img.data);
}

void fftw()
{
    double *in;
    fftw_complex *out;
    int N = 16;
    in = (double*) fftw_malloc(sizeof(double) * N);
    in[0] = 1.0;
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

    fftw_execute(p);

    fftw_destroy_plan(p);

    for (int i = 0; i < N; i++) {
        std::cout << "out[" << i << "] = " << out[i][0] << " + " << out[i][1] << "i\n";
    }

    fftw_free(in);
    fftw_free(out);
}
void fftw_conv2d(const PPMImage* input, PPMImage* output) {
    int H = input->height;
    int W = input->width;
    int KH = KERNEL_HEIGHT;
    int KW = KERNEL_WIDTH;

    int outH = H;
    int outW = W;

    int fftH = H + KH - 1;
    int fftW = W + KW - 1;

    output->width = outW;
    output->height = outH;
    output->max_val = 255;
    output->channels = input->channels;
    output->data = (unsigned char*)malloc(outW * outH * output->channels);

    for (int c = 0; c < output->channels; ++c) {
        // Allocate FFTW arrays
        fftw_complex *A = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftH * (fftW / 2 + 1));
        fftw_complex *B = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftH * (fftW / 2 + 1));
        fftw_complex *C = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftH * (fftW / 2 + 1));

        double *imgPadded = (double*)fftw_malloc(sizeof(double) * fftH * fftW);
        double *kerPadded = (double*)fftw_malloc(sizeof(double) * fftH * fftW);
        double *result = (double*)fftw_malloc(sizeof(double) * fftH * fftW);

        // Zero-pad both arrays
        memset(imgPadded, 0, sizeof(double) * fftH * fftW);
        memset(kerPadded, 0, sizeof(double) * fftH * fftW);

        // Copy image channel to padded input
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                imgPadded[y * fftW + x] = (double)input->data[output->channels * (y * W + x) + c];

        // Copy kernel (flipped for true convolution)
        for (int y = 0; y < KH; ++y)
            for (int x = 0; x < KW; ++x)
                kerPadded[y * fftW + x] = k2d[KH - 1 - y][KW - 1 - x];

        // Plan FFTs
        fftw_plan planA = fftw_plan_dft_r2c_2d(fftH, fftW, imgPadded, A, FFTW_ESTIMATE);
        fftw_plan planB = fftw_plan_dft_r2c_2d(fftH, fftW, kerPadded, B, FFTW_ESTIMATE);
        fftw_plan planC = fftw_plan_dft_c2r_2d(fftH, fftW, C, result, FFTW_ESTIMATE);

        // Execute FFTs
        fftw_execute(planA);
        fftw_execute(planB);

        // Multiply frequency-domain terms (complex multiply)
        int nfreq = fftH * (fftW / 2 + 1);
        for (int i = 0; i < nfreq; ++i) {
            float a_re = A[i][0], a_im = A[i][1];
            float b_re = B[i][0], b_im = B[i][1];
            C[i][0] = a_re * b_re - a_im * b_im;
            C[i][1] = a_re * b_im + a_im * b_re;
        }

        // Inverse FFT
        fftw_execute(planC);

        // Normalize result (FFTW doesn't)
        for (int i = 0; i < fftH * fftW; ++i)
            result[i] /= (fftH * fftW);

        // Crop back to original size
        int yOffset = (KH - 1) / 2;
        int xOffset = (KW - 1) / 2;
        for (int y = 0; y < outH; ++y) {
            for (int x = 0; x < outW; ++x) {
                float val = result[(y + yOffset) * fftW + (x + xOffset)];
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                output->data[output->channels * (y * outW + x) + c] = (unsigned char)(val);
            }
        }

        // Cleanup
        fftw_destroy_plan(planA);
        fftw_destroy_plan(planB);
        fftw_destroy_plan(planC);
        fftw_free(A); fftw_free(B); fftw_free(C);
        fftw_free(imgPadded); fftw_free(kerPadded); fftw_free(result);
    }
}
template<class T>
void HandlePPM(const char* input_ppm, T* k, const char* output_ppm)
{
    PPMImage* img = read_ppm(input_ppm);
    if (!img) exit(1);
    PPMImage out;
    out.width = img->width;
    out.height = img->height;
    out.max_val = 255;
    out.channels = img->channels;
    out.data = (unsigned char*)malloc(img->width * img->height * out.channels);
    if (!out.data) exit(1);
    ImageConvKernel(*img, k, out);
    write_ppm(output_ppm, &out);
}
template<class T>
void HandlePNG(const char* input_png, T* k, const char* output_png) {
    int width, height, channels;
    unsigned char* img = stbi_load(input_png, &width, &height, &channels, 3);
    if (!img) {
        std::cerr << "Failed to load PNG: " << stbi_failure_reason() << "\n";
        exit(1);
    }

    PPMImage ppm_img;
    ppm_img.width = width;
    ppm_img.height = height;
    ppm_img.max_val = 255;
    ppm_img.channels = channels;
    ppm_img.data = img;

    PPMImage out;
    out.width = width;
    out.height = height;
    out.max_val = 255;
    out.channels = ppm_img.channels;
    out.data = (unsigned char*)malloc(width * height * out.channels);
    ImageConvKernel(ppm_img, k, out);
    if (!stbi_write_png(output_png, out.width, out.height, out.channels, out.data, out.width * out.channels)) {
        std::cerr << "Failed to write PNG: " << stbi_failure_reason() << "\n";
        exit(1);
    }
    free(out.data);
    stbi_image_free(img);
}


void HandleMP4_2D(const char* input_path, const char* output_path)
{
    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open input video.\n";
        exit(1);
    }

    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    std::cout << "Video properties: " << width << "x" << height << " at " << fps << " FPS, total frames: " << total_frames << "\n";

    VideoWriter writer(output_path, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video.\n";
        exit(1);
    }

    cv::Mat frame, filtered;
    std::cout << "Processing video...\n";

    StartClock();
    std::cout << "clock started\n";
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        filtered = Mat::zeros(frame.size(), frame.type());
        for (int y = 1; y < frame.rows - 1; y++) {
            for (int x = 1; x < frame.cols - 1; x++) {
                int r_sum = 0, g_sum = 0, b_sum = 0;
                for (int ky = 0; ky < KERNEL_HEIGHT; ky++) {
                    for (int kx = 0; kx < KERNEL_WIDTH; kx++) {
                        Vec3b pixel = frame.at<Vec3b>(y + ky - 1, x + kx - 1);
                        int k = k2d[ky][kx];
                        r_sum += pixel[2] * k;
                        g_sum += pixel[1] * k;
                        b_sum += pixel[0] * k;
                    }
                }
                filtered.at<Vec3b>(y, x)[0] = std::clamp(std::abs(b_sum), 0, 255);
                filtered.at<Vec3b>(y, x)[1] = std::clamp(std::abs(g_sum), 0, 255);
                filtered.at<Vec3b>(y, x)[2] = std::clamp(std::abs(r_sum), 0, 255);
            }
        }
        writer.write(filtered);
    }
    EndClock(true);

    cap.release();
    writer.release();
    std::cout << "Output saved to " << output_path << "\n";
}

void HandleMP4_3D_RGB_Sliding(const char* input_path, const char* output_path)
{
    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cout << "Failed to open input video\n";
        exit(1);
    }

    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    int total_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    std::cout << "Capture initialized: " << width << "x" << height << " at " << fps << " FPS, total frames: " << total_frames << "\n";

    VideoWriter writer(output_path, VideoWriter::fourcc('m','p','4','v'), fps, Size(width, height), true);
    if (!writer.isOpened()) {
        std::cout << "Failed to open output video\n";
        exit(1);
    }
    
    std::cout << "Processing video with sliding 3D convolution...\n";

    std::vector<Mat> frames; 
    for(int i = 0; i < 2; i++) {
        Mat f, f32;
        cap >> f;
        if (f.empty()) break;
        f.convertTo(f32, CV_32FC3);
        frames.push_back(f32);
    }
    
    std::cout << "Processing started...\n";
    std::cout << "clock started\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true)
    {
        for (int i = 0; i < THREAD_COUNT; i++) {
            Mat f, f32;
            cap >> f;
            if (f.empty()) break;
            f.convertTo(f32, CV_32FC3);
            frames.push_back(f32);
        }
        if (frames.size() < KERNEL_DEPTH) break;

        std::vector<Mat> results(THREAD_COUNT);
        std::vector<std::thread> workers;
        auto worker_task = [&](int idx) {
            int cs = 3;
            float* inputs[KERNEL_DEPTH];
            for (int i = 0; i < KERNEL_DEPTH; i++) {
                inputs[i] = (float*)frames[idx + i].data;
            }
            Mat out_frame = Mat::zeros(height, width, CV_32FC3);

            conv3d(
                &inputs[0],
                &k3d[0][0][0],
                (float*)out_frame.data,

                width, 
                height, 
                KERNEL_DEPTH,

                KERNEL_WIDTH, 
                KERNEL_HEIGHT, 
                KERNEL_DEPTH, 

                cs
            );
            
            results[idx] = out_frame;
        };
        for (int i = 0; i < THREAD_COUNT; i++) {
            workers.emplace_back(worker_task, i);
        }
        for (auto& t : workers) {
            if(t.joinable()) t.join();
        }

        for (int i = 0; i < THREAD_COUNT; i++) {
            Mat save_img;
            results[i].convertTo(save_img, CV_8UC3); 
            writer.write(save_img);
        }
        
        for (int i = 0; i < THREAD_COUNT; i++) {
            frames.erase(frames.begin());
        }
        // if (all_frames.size() >= 2) {
        //     current_buffer.clear();
        //     current_buffer.push_back(all_frames[all_frames.size() - 2]);
        //     current_buffer.push_back(all_frames[all_frames.size() - 1]);
        // }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Time taken: " << elapsed.count() << "s\n";

    cap.release();
    writer.release();
    std::cout << "All frames processed and saved to " << output_path << "\n";
}

void usage(const char* prog_name)
{
    std::cout << "Usage: " << prog_name << " <input> <output>\n";
    std::cout << "Supported input/output formats: .ppm, .png, .mp4\n";
}

#include "./test_conv.cpp"

int main(int argc, char* argv[])
{
    const char* input_path = NULL;
    const char* output_path = NULL;
    int i = 1;
    while (i < argc)
    {
        if (strcmp(argv[i], "-i") == 0)
        {
            i++;
            if (i < argc)
            {
                input_path = argv[i];
            }
            else
            {
                std::cerr << "Error: Missing input file after -i\n";
                return 1;
            }
        }
        else if (strcmp(argv[i], "-o") == 0)
        {
            i++;
            if (i < argc)
            {
                output_path = argv[i];
            }
            else
            {
                std::cerr << "Error: Missing output file after -o\n";
                return 1;
            }
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--h") == 0 || strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-help") == 0)
        {
            usage(argv[0]);
            return 0;
        }
        else 
        {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            usage(argv[0]);
            return 1;
        }
        i++;
    }
    if (input_path == NULL || output_path == NULL)
    {
        std::cerr << "Error: Input and output file paths are required.\n";
        usage(argv[0]);
        return 1;
    }

    const char* extension = strrchr(input_path, '.');
    if (strcmp(extension, ".ppm") == 0)
    {
        HandlePPM(input_path, k2d, output_path);
    }
    else if (strcmp(extension, ".png") == 0) 
    {
        HandlePNG(input_path, k2d, output_path);
    }
    else if (strcmp(extension, ".mp4") == 0)
    {
        std::string out_arg = output_path;
        size_t dot = out_arg.rfind('.');
        std::string base = (dot == std::string::npos) ? out_arg : out_arg.substr(0, dot);
        std::string ext = (dot == std::string::npos) ? std::string() : out_arg.substr(dot);

        std::string out_2d = base + "_2d" + ext;
        std::string out_3d_gray = base + "_3d_gray" + ext;
        std::string out_3d_rgb = base + "_3d_rgb" + ext;

        // std::cout << "---------------------------------------------------------------\n";
        // std::cout << "Running 2D per-frame convolution -> " << out_2d << "\n";
        // HandleMP4_2D(input_path, out_2d.c_str());
        // std::cout << "---------------------------------------------------------------\n";
                
        std::cout << "---------------------------------------------------------------\n";
        HandleMP4_3D_RGB_Sliding(input_path, out_3d_rgb.c_str());
        std::cout << "---------------------------------------------------------------\n";
    }
    else 
    {
        std::cerr << "Unsupported file format: " << extension << "\n";
        return 1;
    }
    return 0;
}