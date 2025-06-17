#include <stddef.h>
#include <stdint.h>
#include <string.h> // for memset, memcpy

// --- im2col ---
void im2col(
    const float* input,   // [C_in][H_in][W_in]
    float* col,           // [C_in * kH * kW][H_out * W_out]
    int C_in, int H_in, int W_in,
    int kH, int kW,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int H_out, int W_out
) {
    int col_index = 0;
    for (int h_out = 0; h_out < H_out; ++h_out) {
        for (int w_out = 0; w_out < W_out; ++w_out) {
            int patch = 0;
            for (int c = 0; c < C_in; ++c) {
                for (int kh = 0; kh < kH; ++kh) {
                    for (int kw = 0; kw < kW; ++kw) {
                        int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                        int w_in = w_out * stride_w - pad_w + kw * dilation_w;

                        int im_idx = c * H_in * W_in + h_in * W_in + w_in;
                        int col_idx = patch * H_out * W_out + col_index;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            col[col_idx] = input[im_idx];
                        } else {
                            col[col_idx] = 0.0f;
                        }

                        patch++;
                    }
                }
            }
            col_index++;
        }
    }
}

// --- GEMM: Y = W * X ---
void gemm(
    const float* A,  // [M][K]
    const float* B,  // [K][N]
    float* C,        // [M][N]
    int M, int K, int N
) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// --- conv2d using im2col + gemm ---
void conv2d_im2col(
    const float* input,    // [C_in][H_in][W_in] flattened
    const float* kernel,   // [C_out][C_in][kH][kW] flattened
    const float* bias,     // [C_out]
    float* output,         // [C_out][H_out][W_out] flattened
    int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
) {
    // compute output shape
    int H_out = (H_in + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;

    // allocate im2col buffer: [C_in * kH * kW][H_out * W_out]
    int K = C_in * kH * kW;
    int N = H_out * W_out;
    float* col = (float*)malloc(sizeof(float) * K * N);
    if (!col) return;

    im2col(input, col, C_in, H_in, W_in,
           kH, kW, pad_h, pad_w,
           stride_h, stride_w, dilation_h, dilation_w,
           H_out, W_out);

    // reshape kernel to [C_out][K]
    // do GEMM: [C_out x K] × [K x N] = [C_out x N]
    float* kernel_flat = (float*)malloc(sizeof(float) * C_out * K);
    for (int co = 0; co < C_out; ++co) {
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int src_idx = co * C_in * kH * kW + ci * kH * kW + kh * kW + kw;
                    int dst_idx = co * K + ci * kH * kW + kh * kW + kw;
                    kernel_flat[dst_idx] = kernel[src_idx];
                }
            }
        }
    }

    float* out_col = (float*)malloc(sizeof(float) * C_out * N);
    gemm(kernel_flat, col, out_col, C_out, K, N);

    // add bias and reshape to [C_out][H_out][W_out]
    for (int co = 0; co < C_out; ++co) {
        for (int i = 0; i < N; ++i) {
            int idx = co * N + i;
            output[idx] = out_col[idx] + (bias ? bias[co] : 0.0f);
        }
    }

    free(col);
    free(kernel_flat);
    free(out_col);
}
/*
每列 = [C0-k00, C0-k01, C0-k10, C0-k11,
        C1-k00, C1-k01, C1-k10, C1-k11]
*/

/*
import torch
import torch.nn.functional as F

x = torch.tensor([[[[ 1.,  2.,  3.,  4.],
                    [ 5.,  6.,  7.,  8.],
                    [ 9., 10., 11., 12.],
                    [13., 14., 15., 16.]]]])

# 卷积参数
kH, kW = 2, 2
stride = 1
padding = 0

# unfold = im2col
out = F.unfold(x, kernel_size=(kH, kW), stride=stride, padding=padding)

# 转置使其按列排列，便于与 C 实现对比
print("im2col patches (each row is one patch):")
print(out[0].T)
*/