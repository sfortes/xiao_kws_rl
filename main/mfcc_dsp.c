#include "mfcc_dsp.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "esp_log.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static const char *TAG = "MFCC_DSP";

// Local Macros (Non-conflicting with mfcc_dsp.h)
#define FFT_SIZE        512
#define NUM_MEL_FILTERS 26
#define NUM_MFCC_COEFS  13
#define PREEMPH_ALPHA   0.97f

// State Structures
typedef struct {
    float alpha;
    float prev_x;
    float prev_y;
} dc_blocker_t;

static dc_blocker_t s_dc_filter;
static bool s_dsp_initialized = false;
static float s_hamming_window[FRAME_LEN];
static float s_mel_filterbank[NUM_MEL_FILTERS][(FFT_SIZE / 2) + 1];
static float s_dct_matrix[NUM_MFCC_COEFS][NUM_MEL_FILTERS];

// Helper Functions: Hz <-> Mel Conversion
static inline float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// ----------------------------------------------------------------------------
// 1. Core DSP Sub-routines
// ----------------------------------------------------------------------------
static void dc_blocker_init(dc_blocker_t *filter, float alpha) {
    filter->alpha = alpha;
    filter->prev_x = 0.0f;
    filter->prev_y = 0.0f;
}

static void dc_blocker_process(dc_blocker_t *filter, float *buffer, size_t samples) {
    float a = filter->alpha;
    float x_prev = filter->prev_x;
    float y_prev = filter->prev_y;

    for (size_t i = 0; i < samples; i++) {
        float x = buffer[i];
        float y = x - x_prev + (a * y_prev);
        x_prev = x;
        y_prev = y;
        buffer[i] = y;
    }
    filter->prev_x = x_prev;
    filter->prev_y = y_prev;
}

// In-place Radix-2 Cooley-Tukey FFT
static void fft_radix2(float *real, float *imag, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            float temp_r = real[i];
            float temp_i = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = temp_r;
            imag[j] = temp_i;
        }
        int k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    for (int len = 2; len <= n; len <<= 1) {
        float half_len = len / 2.0f;
        float angle = -2.0f * M_PI / len;
        float wlen_r = cosf(angle);
        float wlen_i = sinf(angle);

        for (int i = 0; i < n; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (int k = 0; k < half_len; k++) {
                int u = i + k;
                int v = i + k + (int)half_len;
                float u_r = real[u];
                float u_i = imag[u];
                float v_r = real[v] * w_r - imag[v] * w_i;
                float v_i = real[v] * w_i + imag[v] * w_r;

                real[u] = u_r + v_r;
                imag[u] = u_i + v_i;
                real[v] = u_r - v_r;
                imag[v] = u_i - v_i;

                float next_w_r = w_r * wlen_r - w_i * wlen_i;
                float next_w_i = w_r * wlen_i + w_i * wlen_r;
                w_r = next_w_r;
                w_i = next_w_i;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 2. Filterbank & DCT Initialization
// ----------------------------------------------------------------------------
void mfcc_dsp_init(void) {
    if (s_dsp_initialized) {
        return;
    }

    // DC Blocker Setup (~20Hz cutoff @ 16kHz)
    dc_blocker_init(&s_dc_filter, 0.985f);

    // 1. Pre-calculate Hamming Window
    for (int i = 0; i < FRAME_LEN; i++) {
        s_hamming_window[i] = 0.54f - 0.46f * cosf((2.0f * M_PI * i) / (FRAME_LEN - 1));
    }

    // 2. Pre-calculate Triangular Mel Filterbank
    memset(s_mel_filterbank, 0, sizeof(s_mel_filterbank));
    float low_mel = hz_to_mel(300.0f);     // Voice band bottom
    float high_mel = hz_to_mel(8000.0f);   // Voice band top cutoff
    float mel_step = (high_mel - low_mel) / (NUM_MEL_FILTERS + 1);

    int bin_points[NUM_MEL_FILTERS + 2];
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        float hz = mel_to_hz(low_mel + i * mel_step);
        bin_points[i] = (int)floorf((FFT_SIZE + 1) * hz / (float)SAMPLE_RATE);
    }

    for (int m = 1; m <= NUM_MEL_FILTERS; m++) {
        int left = bin_points[m - 1];
        int center = bin_points[m];
        int right = bin_points[m + 1];

        for (int k = left; k < center; k++) {
            if (center != left) {
                s_mel_filterbank[m - 1][k] = (float)(k - left) / (center - left);
            }
        }
        for (int k = center; k < right; k++) {
            if (right != center) {
                s_mel_filterbank[m - 1][k] = (float)(right - k) / (right - center);
            }
        }
    }

    // 3. Pre-calculate DCT-II Matrix
    for (int i = 0; i < NUM_MFCC_COEFS; i++) {
        for (int j = 0; j < NUM_MEL_FILTERS; j++) {
            s_dct_matrix[i][j] = cosf((M_PI * i / NUM_MEL_FILTERS) * (j + 0.5f));
        }
    }

    s_dsp_initialized = true;
    ESP_LOGI(TAG, "MFCC DSP pipeline fully initialized.");
}

// ----------------------------------------------------------------------------
// 3. Full Feature Extraction Process
// ----------------------------------------------------------------------------
void mfcc_compute(const int16_t* pcm_data, float* out_features) {
    if (!s_dsp_initialized || pcm_data == NULL || out_features == NULL) {
        return;
    }

    // Step 0: PCM Normalization + Digital Gain Boost (16.0x for PDM Mic)
    float work_buf[FRAME_LEN];
    for (size_t i = 0; i < FRAME_LEN; i++) {
        float norm = ((float)pcm_data[i] / 32768.0f) * 16.0f;
        // Hard clip to [-1.0, 1.0] bounds
        if (norm > 1.0f) {
            norm = 1.0f;
        } else if (norm < -1.0f) {
            norm = -1.0f;
        }
        work_buf[i] = norm;
    }

    // Step 1: Dynamic DC Offset Removal
    dc_blocker_process(&s_dc_filter, work_buf, FRAME_LEN);

    // Step 2: Pre-emphasis & Hamming Windowing
    float real[FFT_SIZE];
    float imag[FFT_SIZE];
    memset(imag, 0, sizeof(imag));

    real[0] = work_buf[0] * s_hamming_window[0];
    for (size_t i = 1; i < FRAME_LEN; i++) {
        float preemph = work_buf[i] - PREEMPH_ALPHA * work_buf[i - 1];
        real[i] = preemph * s_hamming_window[i];
    }
    // Zero-pad frame from 400 to 512 points for FFT
    for (size_t i = FRAME_LEN; i < FFT_SIZE; i++) {
        real[i] = 0.0f;
    }

    // Step 3: Compute FFT
    fft_radix2(real, imag, FFT_SIZE);

    // Step 4: Compute Power Spectrum
    float power_spectrum[(FFT_SIZE / 2) + 1];
    for (int i = 0; i <= FFT_SIZE / 2; i++) {
        power_spectrum[i] = (real[i] * real[i] + imag[i] * imag[i]) / FFT_SIZE;
    }

    // Step 5: Mel-Filter Bank Energies with Base-10 Decibel Log Scale
    float mel_energies[NUM_MEL_FILTERS];
    for (int m = 0; m < NUM_MEL_FILTERS; m++) {
        float sum = 0.0f;
        for (int k = 0; k <= FFT_SIZE / 2; k++) {
            sum += power_spectrum[k] * s_mel_filterbank[m][k];
        }
        // Match standard PyTorch/Librosa dB log scaling: 10 * log10(sum + 1e-4)
        mel_energies[m] = 10.0f * log10f(sum + 1e-4f);
    }

    // Step 6: Discrete Cosine Transform (DCT-II) -> 13 MFCCs
    for (int i = 0; i < NUM_MFCC_COEFS; i++) {
        float sum = 0.0f;
        for (int j = 0; j < NUM_MEL_FILTERS; j++) {
            sum += mel_energies[j] * s_dct_matrix[i][j];
        }
        out_features[i] = sum;
    }
}