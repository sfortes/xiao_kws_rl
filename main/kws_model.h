#ifndef KWS_MODEL_H
#define KWS_MODEL_H

#include <stdint.h>

#define MODEL_INPUT_SIZE 1261
#define MODEL_HIDDEN_SIZE 64
#define MODEL_OUTPUT_SIZE 6

// Mutable dynamic weights residing in SRAM for Core 1 RL training updates
extern float fc1_w[MODEL_HIDDEN_SIZE][MODEL_INPUT_SIZE];
extern float fc1_b[MODEL_HIDDEN_SIZE];
extern float fc2_w[MODEL_OUTPUT_SIZE][MODEL_HIDDEN_SIZE];
extern float fc2_b[MODEL_OUTPUT_SIZE];

void kws_model_init(void);
void model_inference(const float* input, float* output);
#endif
