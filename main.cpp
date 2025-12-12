#include <iostream>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <thread>
#include "fftw-3.3.10/fftw-3.3.10/api/fftw3.h"
using namespace cv;

#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define KERNEL_DEPTH 3

bool pseudo = false;
bool verbose = false;

typedef struct {
    int width;
    int height;
    int max_val;
    int channels;
    unsigned char* data;
} PPMImage;

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
float clampf(float val) {
    if (val < 0.0f) return 0.0f;
    if (val > 255.0f) return 255.0f;
    return val;
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

// 1d conv
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
                    int px = x + kx - marginx;
                    int py = y + ky - marginy;
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

    if (verbose) std::cout << "Video properties: " << width << "x" << height << " at " << fps << " FPS, total frames: " << total_frames << "\n";

    VideoWriter writer(output_path, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video.\n";
        exit(1);
    }

    cv::Mat frame, filtered;
    if (verbose) std::cout << "Processing video...\n";

    StartClock();
    if (verbose) std::cout << "clock started\n";
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
    if (verbose) std::cout << "Output saved to " << output_path << "\n";
}
void HandleMP4_3D(const char* input_path, const char* output_path)
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

    if (verbose) std::cout << "Video properties: " << width << "x" << height << " at " << fps << " FPS, total frames: " << total_frames << "\n";
    if (verbose) std::cout << "Loading frames...\n";
    std::vector<Mat> input_frames;
    for (int i = 0; i < total_frames; ++i) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        Mat frame_f;
        frame.convertTo(frame_f, CV_32F);
        input_frames.push_back(frame_f);
    }
    cap.release();

    int padD = KERNEL_DEPTH / 2, padH = KERNEL_HEIGHT / 2, padW = KERNEL_WIDTH / 2;

    int D = static_cast<int>(input_frames.size());
    int H = input_frames[0].rows;
    int W = input_frames[0].cols;

    std::vector<Mat> output_frames(D);
    for (int t = 0; t < D; ++t)
        output_frames[t] = Mat::zeros(H, W, CV_32F);

    if (verbose) std::cout << "Applying 3D convolution...\n";
    // 3D convolution
    StartClock();
    if (verbose) std::cout << "clock started\n";
    for (int z = padD; z < D - padD; ++z) {
        for (int y = padH; y < H - padH; ++y) {
            for (int x = padW; x < W - padW; ++x) {
                float sum = 0.0f;
                for (int dz = 0; dz < KERNEL_DEPTH; ++dz)
                    for (int dy = 0; dy < KERNEL_HEIGHT; ++dy)
                        for (int dx = 0; dx < KERNEL_WIDTH; ++dx)
                        {
                            int tz = z + dz - padD;
                            int ty = y + dy - padH;
                            int tx = x + dx - padW;
                            sum += input_frames[tz].at<float>(ty, tx) * k3d[dz][dy][dx];
                        }
                output_frames[z].at<float>(y, x) = clampf((sum));
            }
        }
    }
    EndClock(true);

    if (verbose) std::cout << "Writing output video...\n";
    // Convert float frames to 8-bit for VideoWriter
    VideoWriter writer(output_path, VideoWriter::fourcc('m','p','4','v'), fps, Size(width, height), false);
    if (!writer.isOpened()) {
        std::cout << "Failed to open output video\n";
        exit(1);
    }

    for (auto &f : output_frames) {
        Mat f8u;
        f.convertTo(f8u, CV_8U, 1.0, 0);
        writer.write(f8u);
    }

    if (verbose) std::cout << "Output saved to " << output_path << "\n";
}
void HandleMP4_3D_RGB(const char* input_path, const char* output_path)
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

    if (verbose) std::cout << "Video properties: " << width << "x" << height << " at " << fps << " FPS, total frames: " << total_frames << "\n";
    if (verbose) std::cout << "Loading frames...\n";
    // Load all frames (color)
    std::vector<Mat> input_frames;
    for (int i = 0; i < total_frames; ++i) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        Mat frame_f;
        frame.convertTo(frame_f, CV_8UC3);  // convert to float with 3 channels
        input_frames.push_back(frame_f);
    }
    cap.release();

    int padD = KERNEL_DEPTH / 2, padH = KERNEL_HEIGHT / 2, padW = KERNEL_WIDTH / 2;

    int D = static_cast<int>(input_frames.size());
    int H = input_frames[0].rows;
    int W = input_frames[0].cols;
    
    // Initialize output frames
    std::vector<Mat> output_frames(D);
    for (int t = 0; t < D; ++t)
        output_frames[t] = Mat::zeros(H, W, CV_8UC3);

    if (verbose) std::cout << "Applying 3D convolution on RGB frames...\n";
    // 3D convolution
    StartClock();
    if (verbose) std::cout << "clock started\n";
    for (int z = padD; z < D - padD; ++z) {
        for (int y = padH; y < H - padH; ++y) {
            for (int x = padW; x < W - padW; ++x) {
                Vec3f sum(0, 0, 0);
                for (int dz = 0; dz < KERNEL_DEPTH; ++dz)
                    for (int dy = 0; dy < KERNEL_HEIGHT; ++dy)
                        for (int dx = 0; dx < KERNEL_WIDTH; ++dx)
                        {
                            int tz = z + dz - padD;
                            int ty = y + dy - padH;
                            int tx = x + dx - padW;
                            Vec3f pixel = input_frames[tz].at<Vec3f>(ty, tx);
                            float k = k3d[dz][dy][dx];
                            sum[0] += pixel[0] * k; // Blue
                            sum[1] += pixel[1] * k; // Green
                            sum[2] += pixel[2] * k; // Red
                        }
                output_frames[z].at<Vec3f>(y, x) = sum;
            }
        }
    }
    EndClock(true);

    if (verbose) std::cout << "Writing output video...\n";
    // Write output video
    VideoWriter writer(output_path, VideoWriter::fourcc('m','p','4','v'), fps, Size(width, height), true);
    if (!writer.isOpened()) {
        std::cout << "Failed to open output video\n";
        exit(1);
    }

    for (auto &f : output_frames) {
        Mat f8u;
        f.convertTo(f8u, CV_8UC3, 1.0, 0);
        writer.write(f8u);
    }

    if (verbose) std::cout << "Output saved to " << output_path << "\n";
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

    if (verbose) std::cout << "Video properties: " << width << "x" << height << " at " << fps << " FPS, total frames: " << total_frames << "\n";

    int padH = KERNEL_HEIGHT / 2, padW = KERNEL_WIDTH / 2;
    int padD = KERNEL_DEPTH / 2;

    // Prepare VideoWriter
    VideoWriter writer(output_path, VideoWriter::fourcc('m','p','4','v'), fps, Size(width, height), true);
    if (!writer.isOpened()) {
        std::cout << "Failed to open output video\n";
        exit(1);
    }

    if (verbose) std::cout << "Processing video with sliding 3D convolution...\n";

    int frame_idx = 0;
    Mat frame;
    StartClock();
    if (verbose) std::cout << "clock started\n";

    int num_threads = std::thread::hardware_concurrency();
    std::cout << "Detected " << num_threads << " hardware threads.\n";
    if (num_threads == 0) num_threads = 4;

    auto worker = [&](int out_index, const std::vector<Mat>& frames, Mat& out)
    {
        const Mat& A = frames[out_index];
        const Mat& B = frames[out_index + 1];
        const Mat& C = frames[out_index + 2];

        Mat temp = Mat::zeros(height, width, CV_32FC3);
        const Mat buf[KERNEL_DEPTH] = { A, B, C };

        for (int y = padH; y < height - padH; y++)
        {
            for (int x = padW; x < width - padW; x++)
            {
                Vec3f sum(0,0,0);
                for (int dz = 0; dz < KERNEL_DEPTH; dz++)
                    for (int dy = 0; dy < KERNEL_HEIGHT; dy++)
                        for (int dx = 0; dx < KERNEL_WIDTH; dx++)
                        {
                            int ty = y + dy - padH;
                            int tx = x + dx - padW;

                            Vec3f p = buf[dz].at<Vec3f>(ty, tx);
                            float k = k3d[dz][dy][dx];

                            sum[0] += p[0] * k;
                            sum[1] += p[1] * k;
                            sum[2] += p[2] * k;
                        }
                sum[0] = clampf(sum[0]);
                sum[1] = clampf(sum[1]);
                sum[2] = clampf(sum[2]);

                temp.at<Vec3f>(y, x) = sum;
            }
        }
        out = temp;
    };

    std::vector<Mat> frame_buffer;  
    frame_buffer.reserve(num_threads + 2);
    for (int i = 0; i < num_threads + 2; i++)
    {
        Mat f;
        cap >> f;
        if (f.empty()) break;
        Mat f32;
        f.convertTo(f32, CV_32FC3);
        frame_buffer.push_back(f32);
    }

    int base = 0;
    while (frame_buffer.size() >= KERNEL_DEPTH)
    {
        int batch_size = std::min(num_threads, (int)frame_buffer.size() - 2);

        std::vector<Mat> outputs(batch_size);
        std::vector<std::thread> threads;
        for (int t = 0; t < batch_size; t++)
        {
            threads.emplace_back(worker, t, std::ref(frame_buffer), std::ref(outputs[t]));
        }
        for (auto& th : threads) th.join();

        for (int t = 0; t < batch_size; t++)
        {
            Mat out8;
            outputs[t].convertTo(out8, CV_8UC3);
            writer.write(out8);
        }
        for (int i = 0; i < batch_size; i++)
            frame_buffer.erase(frame_buffer.begin());
        for (int i = 0; i < batch_size; i++)
        {
            Mat f;
            cap >> f;
            if (f.empty()) break;
            Mat f32;
            f.convertTo(f32, CV_32FC3);
            frame_buffer.push_back(f32);
        }
    }

    EndClock(true);
    
    cap.release();
    writer.release();
    if (verbose) std::cout << "Output saved to " << output_path << "\n";
}

void usage(const char* prog_name)
{
    std::cout << "Usage: " << prog_name << " <input> <output>\n";
    std::cout << "Supported input/output formats: .ppm, .png, .mp4\n";
}

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
        else if (strcmp(argv[i], "-verbose") == 0)
        {
            verbose = true;
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
        
        // std::cout << "---------------------------------------------------------------\n";
        // std::cout << "Running 3D temporal convolution (grayscale) -> " << out_3d_gray << "\n";
        // HandleMP4_3D(input_path, out_3d_gray.c_str());
        // std::cout << "---------------------------------------------------------------\n";
        
        std::cout << "---------------------------------------------------------------\n";
        std::cout << "Running 3D temporal convolution (RGB) -> " << out_3d_rgb << "\n";
        // HandleMP4_3D_RGB(input_path, out_3d_rgb.c_str()); // consumes too much memory
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