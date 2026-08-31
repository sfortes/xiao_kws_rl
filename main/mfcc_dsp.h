#ifndef MFCC_DSP_H
#define MFCC_DSP_H

#include <stdint.h>

#define SAMPLE_RATE 16000
#define AUDIO_LEN_SAMPLES 16000
#define FFT_SIZE 512
#define FRAME_LEN 400   // 25ms
#define FRAME_STEP 160  // 10ms
#define NUM_FRAMES 97
#define NUM_MELS 32
#define NUM_MFCC 13

void mfcc_dsp_init(void);
void mfcc_compute(const int16_t* pcm_data, float* out_features);

#endif