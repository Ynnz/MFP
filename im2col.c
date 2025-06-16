/*
void im2col_c(
    const float* input,      // shape: [C, H, W] 或 [H, W, C]
    float* output,           // shape: depends on layout
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int height_col, int width_col,
    int channels_last        // 1: NHWC, 0: NCHW
);

*/

#include <string.h> // for memcpy
#include <stdint.h>

void im2col_c(
    const float* input,
    float* output,
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int height_col, int width_col,
    int channels_last
) {
    if (channels_last) {
        // NHWC 模式
        for (int i_col = 0; i_col < height_col * width_col; ++i_col) {
            int h_col = i_col / width_col;
            int w_col = i_col % width_col;

            for (int h_offset = 0; h_offset < kernel_h; ++h_offset) {
                int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

                for (int w_offset = 0; w_offset < kernel_w; ++w_offset) {
                    int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

                    float* slice_col = output + 
                        ((i_col * kernel_h * kernel_w + h_offset * kernel_w + w_offset) * channels);

                    if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
                        const float* slice_im = input + (h_im * width + w_im) * channels;
                        memcpy(slice_col, slice_im, sizeof(float) * channels);
                    } else {
                        memset(slice_col, 0, sizeof(float) * channels);
                    }
                }
            }
        }
    } else {
        // NCHW 模式
        int channels_col = channels * kernel_h * kernel_w;
        for (int c_col = 0; c_col < channels_col; ++c_col) {
            int w_offset = c_col % kernel_w;
            int h_offset = (c_col / kernel_w) % kernel_h;
            int c_im = c_col / (kernel_h * kernel_w);

            for (int h_col = 0; h_col < height_col; ++h_col) {
                int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

                for (int w_col = 0; w_col < width_col; ++w_col) {
                    int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

                    float val = 0.0f;
                    if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
                        val = input[(c_im * height + h_im) * width + w_im];
                    }

                    output[(c_col * height_col + h_col) * width_col + w_col] = val;
                }
            }
        }
    }
}
