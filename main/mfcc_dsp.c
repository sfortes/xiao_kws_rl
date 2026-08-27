#include "mfcc_dsp.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#include "esp_attr.h"

EXT_RAM_BSS_ATTR static float hamming_window[FRAME_LEN];
EXT_RAM_BSS_ATTR static float mel_filters[NUM_MELS][FFT_SIZE / 2 + 1];
EXT_RAM_BSS_ATTR static float dct_matrix[NUM_MFCC][NUM_MELS];

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void mfcc_init(void) {
    // 1. Initialize Hamming Window
    for (int i = 0; i < FRAME_LEN; i++) {
        hamming_window[i] = 0.54f - 0.46f * cosf((2.0f * M_PI * i) / (FRAME_LEN - 1));
    }

    // 2. Build Triangular Mel Filterbank
    float min_mel = hz_to_mel(0.0f);
    float max_mel = hz_to_mel(SAMPLE_RATE / 2.0f);
    float mel_points[NUM_MELS + 2];
    int bin_points[NUM_MELS + 2];

    for (int i = 0; i < NUM_MELS + 2; i++) {
        mel_points[i] = min_mel + i * (max_mel - min_mel) / (NUM_MELS + 1);
        float hz = mel_to_hz(mel_points[i]);
        bin_points[i] = (int)floorf((FFT_SIZE + 1) * hz / SAMPLE_RATE);
    }

    memset(mel_filters, 0, sizeof(mel_filters));
    for (int m = 1; m <= NUM_MELS; m++) {
        for (int k = bin_points[m - 1]; k < bin_points[m]; k++) {
            mel_filters[m - 1][k] = (float)(k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1]);
        }
        for (int k = bin_points[m]; k < bin_points[m + 1]; k++) {
            mel_filters[m - 1][k] = (float)(bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m]);
        }
    }

    // 3. Compute Discrete Cosine Transform (DCT-II) Matrix
    for (int i = 0; i < NUM_MFCC; i++) {
        for (int j = 0; j < NUM_MELS; j++) {
            dct_matrix[i][j] = cosf((M_PI * i * (2 * j + 1)) / (2.0f * NUM_MELS));
        }
    }
}

// Discrete Fourier Transform implementation (Magnitude Spectrum)
static void compute_fft_mag(const float* frame, float* mag) {
    for (int k = 0; k <= FFT_SIZE / 2; k++) {
        float real = 0.0f;
        float imag = 0.0f;
        for (int n = 0; n < FRAME_LEN; n++) {
            float angle = 2.0f * M_PI * k * n / FFT_SIZE;
            real += frame[n] * cosf(angle);
            imag -= frame[n] * sinf(angle);
        }
        mag[k] = sqrtf(real * real + imag * imag);
    }
}

void mfcc_compute(const int16_t* pcm_data, float* out_features) {
    float frame_buf[FRAME_LEN];
    float mag_spec[FFT_SIZE / 2 + 1];
    float mel_energy[NUM_MELS];

    for (int f = 0; f < NUM_FRAMES; f++) {
        int sample_offset = f * FRAME_STEP;

        // Apply Windowing
        for (int i = 0; i < FRAME_LEN; i++) {
            frame_buf[i] = ((float)pcm_data[sample_offset + i]) * hamming_window[i];
        }

        // FFT Magnitude Spectrum
        compute_fft_mag(frame_buf, mag_spec);

        // Filterbank Energies & Log
        for (int m = 0; m < NUM_MELS; m++) {
            float sum = 1e-6f; // Floor epsilon prevents log(0)
            for (int k = 0; k <= FFT_SIZE / 2; k++) {
                sum += mag_spec[k] * mel_filters[m][k];
            }
            mel_energy[m] = logf(sum);
        }

        // DCT-II Conversion to MFCC Coefficients
        for (int i = 0; i < NUM_MFCC; i++) {
            float sum = 0.0f;
            for (int j = 0; j < NUM_MELS; j++) {
                sum += mel_energy[j] * dct_matrix[i][j];
            }
            // Layout: [13 MFCCs, 97 Frames] (Row-major matching PyTorch input tensor)
            out_features[i * NUM_FRAMES + f] = sum;
        }
    }
}