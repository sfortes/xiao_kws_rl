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
#define NUM_MEL_FILTERS 32
#define NUM_MFCC_COEFS  13
static bool s_dsp_initialized = false;
static float s_window[FRAME_LEN];
static float s_mel_filterbank[NUM_MEL_FILTERS][(FFT_SIZE / 2) + 1];
static float s_dct_matrix[NUM_MFCC_COEFS][NUM_MEL_FILTERS];
static float s_mel_db[NUM_FRAMES][NUM_MEL_FILTERS];

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

    // 1. Pre-calculate PyTorch's periodic Hann window.
    for (int i = 0; i < FRAME_LEN; i++) {
        s_window[i] = 0.5f - 0.5f * cosf((2.0f * M_PI * i) / FRAME_LEN);
    }

    // 2. Pre-calculate the torchaudio-style triangular Mel filterbank.
    // This matches Librosa/torchaudio's `melscale_fbanks` construction using
    // `torch.linspace(0, sample_rate / 2, n_freqs)` and a `n_mels + 2` point mel grid.
    memset(s_mel_filterbank, 0, sizeof(s_mel_filterbank));
    float m_min = hz_to_mel(0.0f);
    float m_max = hz_to_mel((float)SAMPLE_RATE / 2.0f);
    float mel_pts[NUM_MEL_FILTERS + 2];
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        float mel = m_min + ((m_max - m_min) * (float)i) / (NUM_MEL_FILTERS + 1);
        mel_pts[i] = mel_to_hz(mel);
    }

    for (int m = 0; m < NUM_MEL_FILTERS; m++) {
        float left = mel_pts[m];
        float center = mel_pts[m + 1];
        float right = mel_pts[m + 2];

        for (int k = 0; k <= FFT_SIZE / 2; k++) {
            float freq_hz = ((float)k * (float)SAMPLE_RATE) / (float)FFT_SIZE;
            if (freq_hz >= left && freq_hz <= center) {
                if (center != left) {
                    s_mel_filterbank[m][k] = (freq_hz - left) / (center - left);
                }
            } else if (freq_hz > center && freq_hz <= right) {
                if (right != center) {
                    s_mel_filterbank[m][k] = (right - freq_hz) / (right - center);
                }
            }
        }
    }

    // 3. Pre-calculate DCT-II Matrix with ortho normalization to match torchaudio's
    // default ``norm='ortho'`` behavior for the MFCC transform.
    for (int i = 0; i < NUM_MFCC_COEFS; i++) {
        for (int j = 0; j < NUM_MEL_FILTERS; j++) {
            float angle = (M_PI / NUM_MEL_FILTERS) * (j + 0.5f) * i;
            float base = cosf(angle);
            s_dct_matrix[i][j] = (i == 0)
                ? base * sqrtf(1.0f / NUM_MEL_FILTERS)
                : base * sqrtf(2.0f / NUM_MEL_FILTERS);
        }
    }

    s_dsp_initialized = true;
    ESP_LOGI(TAG, "MFCC DSP pipeline fully initialized.");
}

// ----------------------------------------------------------------------------
// 3. Full Feature Extraction Process
// ----------------------------------------------------------------------------
void mfcc_compute_batch(const int16_t *pcm_data, float *out_features) {
    if (!s_dsp_initialized || pcm_data == NULL || out_features == NULL) {
        return;
    }

    float max_mel_db = -INFINITY;
    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        float real[FFT_SIZE] = {0};
        float imag[FFT_SIZE] = {0};

        for (int i = 0; i < FRAME_LEN; i++) {
            real[i] = ((float)pcm_data[frame * FRAME_STEP + i] / 32768.0f) * s_window[i];
        }

        fft_radix2(real, imag, FFT_SIZE);

        for (int m = 0; m < NUM_MEL_FILTERS; m++) {
            float mel_power = 0.0f;
            for (int k = 0; k <= FFT_SIZE / 2; k++) {
                float power = real[k] * real[k] + imag[k] * imag[k];
                mel_power += power * s_mel_filterbank[m][k];
            }
            s_mel_db[frame][m] = 10.0f * log10f(fmaxf(mel_power, 1.0e-10f));
            if (s_mel_db[frame][m] > max_mel_db) {
                max_mel_db = s_mel_db[frame][m];
            }
        }
    }

    // Torchaudio MFCC uses AmplitudeToDB(top_db=80), whose threshold is based on
    // the maximum Mel power in the complete input clip.
    float mel_floor_db = max_mel_db - 80.0f;
    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        for (int coefficient = 0; coefficient < NUM_MFCC_COEFS; coefficient++) {
            float sum = 0.0f;
            for (int mel = 0; mel < NUM_MEL_FILTERS; mel++) {
                sum += fmaxf(s_mel_db[frame][mel], mel_floor_db) * s_dct_matrix[coefficient][mel];
            }
            out_features[coefficient * NUM_FRAMES + frame] = sum;
        }
    }
}