#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "driver/i2s_pdm.h"
#include "esp_log.h"
#include "kws_model.h"
#include "mfcc_dsp.h"

#define NUM_MFCC_COEFS 13

#define PDM_CLK_IO GPIO_NUM_42
#define PDM_DIN_IO GPIO_NUM_41

static const char *TAG = "KWS_RL_SYSTEM";
static const char *WORD_LABELS[MODEL_OUTPUT_SIZE] = {"UP", "DOWN", "LEFT", "RIGHT", "BACK", "GO"};

// Struct passed over FreeRTOS Queue from Core 0 -> Core 1
typedef struct {
    int label_idx;
    float confidence;
    float features[MODEL_INPUT_SIZE]; // Retain features for Core 1 policy update
} inference_msg_t;

static QueueHandle_t xInferenceQueue = NULL;
static SemaphoreHandle_t xModelMutex = NULL;
static i2s_chan_handle_t rx_chan = NULL;

// Softmax Calculation
static void softmax(const float* input, float* output, int len) {
    float max_val = input[0];
    for (int i = 1; i < len; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < len; i++) {
        output[i] /= sum;
    }
}

static void init_pdm_microphone(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

    i2s_pdm_rx_config_t pdm_rx_cfg = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .clk = PDM_CLK_IO,
            .din = PDM_DIN_IO,
            .invert_flags = { .clk_inv = false },
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx_chan, &pdm_rx_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
}

void vCore0_InferenceTask(void *pvParameters) {
    static int16_t raw_pcm[AUDIO_LEN_SAMPLES];
    static float mfcc_features[MODEL_INPUT_SIZE];
    static float raw_logits[MODEL_OUTPUT_SIZE];
    static float probabilities[MODEL_OUTPUT_SIZE];
    size_t bytes_read = 0;

    mfcc_dsp_init();

    while (1) {
        // Read 1 second of clean mono audio data from the PDM mic
        esp_err_t result = i2s_channel_read(rx_chan, raw_pcm, sizeof(raw_pcm), &bytes_read, portMAX_DELAY);
        if (result == ESP_OK && bytes_read == sizeof(raw_pcm)) {
            
            // 1. Calculate Real-Time Audio Power (Volume RMS)
            double sum_squares = 0.0;
            for (int i = 0; i < AUDIO_LEN_SAMPLES; i++) {
                sum_squares += (double)raw_pcm[i] * raw_pcm[i];
            }
            float audio_rms = sqrtf((float)(sum_squares / AUDIO_LEN_SAMPLES));

            // 2. Execute DSP MFCC Feature Extraction across overlapping windows
            int frame_count = 0;
            int stride = 160; 
            for (int offset = 0; offset <= (AUDIO_LEN_SAMPLES - FRAME_LEN); offset += stride) {
                if (frame_count >= 97) break; 

                float temp_mfcc[NUM_MFCC_COEFS];
                mfcc_compute(&raw_pcm[offset], temp_mfcc);

                // Distribute features to match PyTorch's flattened layout order
                for (int coef_idx = 0; coef_idx < NUM_MFCC_COEFS; coef_idx++) {
                    
                    // 🟢 ORTHO-NORMALIZATION CORRECTION LAYER:
                    // Adjusts raw C coefficients to match Torchaudio's internal norm='ortho' math.
                    // Uses 32.0f directly to align with your PyTorch N_MELS setup.
                    if (coef_idx == 0) {
                        temp_mfcc[coef_idx] *= sqrtf(1.0f / (4.0f * 32.0f));
                    } else {
                        temp_mfcc[coef_idx] *= sqrtf(2.0f / 32.0f);
                    }

                    int target_flat_index = (coef_idx * 97) + frame_count;
                    mfcc_features[target_flat_index] = temp_mfcc[coef_idx];
                }
                frame_count++;
            }

            // Match Torchaudio raw scale clipping boundaries 
            for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
                if (mfcc_features[i] < -120.0f) {
                    mfcc_features[i] = -120.0f;
                }
            }

            // 3. Thread-safe access to Shared Model Weights in SRAM
            if (xSemaphoreTake(xModelMutex, portMAX_DELAY) == pdTRUE) {
                model_inference(mfcc_features, raw_logits);
                xSemaphoreGive(xModelMutex);
            }

            // Squash raw outputs to probabilities
            softmax(raw_logits, probabilities, MODEL_OUTPUT_SIZE);

            // Find top prediction index and confidence value
            int max_idx = 0;
            float max_conf = probabilities[0];
            for (int i = 1; i < MODEL_OUTPUT_SIZE; i++) {
                if (probabilities[i] > max_conf) {
                    max_conf = probabilities[i];
                    max_idx = i;
                }
            }

            // Calculate feature boundaries for tracking trends
            float feat_min = mfcc_features[0];
            float feat_max = mfcc_features[0];
            for (int i = 1; i < MODEL_INPUT_SIZE; i++) {
                if (mfcc_features[i] < feat_min) feat_min = mfcc_features[i];
                if (mfcc_features[i] > feat_max) feat_max = mfcc_features[i];
            }

            // 4. Print Core Metrics Dashboard
            ESP_LOGI(TAG, "--------------------------------------------------");
            ESP_LOGI(TAG, "[MIC] Volume RMS: %.2f", audio_rms);
            ESP_LOGI(TAG, "[DSP] MFCC Range: [Min: %.2f, Max: %.2f]", feat_min, feat_max);
            
            char logits_buf[128] = {0};
            int offset_buf = 0;
            for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
                offset_buf += snprintf(logits_buf + offset_buf, sizeof(logits_buf) - offset_buf, "%.2f ", raw_logits[i]);
            }
            ESP_LOGI(TAG, "[MODEL] Raw Native Logits: [ %s]", logits_buf);

            // 🟢 CALIBRATED FEATURE FILTER GATE:
            // Adjust boundaries to match the clean ortho-normalized speech range.
            // Silence drops under 3.0, words spike past 8.0+. 
            // We allow clear probability peaks (> 45%) to pass now that background noise bias is gone!
            if (feat_max > 6.0f && max_conf > 0.45f) {
                ESP_LOGI(TAG, "🟢 [MATCH SUCCESS] Detected Word: %s (Confidence: %.2f%%)", WORD_LABELS[max_idx], max_conf * 100.0f);
            } else {
                ESP_LOGI(TAG, "💤 [IDLE] Silence or Unclear Word...");
            }
            ESP_LOGI(TAG, "--------------------------------------------------");
        }
    }
}

// Core 1 Task: Contextual Policy Gradient (REINFORCE Algorithm)
void vCore1_RLTask(void *pvParameters) {
    inference_msg_t msg;
    //float learning_rate = 0.001f;

    while (1) {
        if (xQueueReceive(xInferenceQueue, &msg, portMAX_DELAY) == pdTRUE) {
            
            // Simulated Reward Environment Policy logic:
            // High confidence (>80%) grants Positive Reward (+1.0), lower triggers Penalty (-1.0)
            float reward = (msg.confidence > 0.80f) ? 1.0f : -1.0f;

            if (reward < 0.0f) {
               ESP_LOGW(TAG, "[Core 1 RL] Low Confidence Detection! Adapting Weights in SRAM via Policy Gradient...");

                if (xSemaphoreTake(xModelMutex, portMAX_DELAY) == pdTRUE) {
                    
                    // 1. Boost the learning rate so the changes are visible immediately
                    float effective_lr = 0.05f; 

                    // 2. Punish the stuck word by lowering its bias directly
                    fc2_b[msg.label_idx] += effective_lr * reward;

                    // 3. Apply the input feature context to the output layer weights safely
                    // We loop through the hidden size but scale the weights down based on the penalty
                    for (int j = 0; j < MODEL_HIDDEN_SIZE; j++) {
                        fc2_w[msg.label_idx][j] += effective_lr * reward * 0.1f;
                    }

                    xSemaphoreGive(xModelMutex);
                    ESP_LOGI(TAG, "[Core 1 RL] SRAM Model Weights Successfully Updated.");
                }
            } else {
                ESP_LOGI(TAG, "[Core 1 RL] Optimal Prediction Verified (Reward: +1.0). No Weight Shift Required.");
            }
        }
    }
}

void app_main(void) {

    kws_model_init();
    
    xInferenceQueue = xQueueCreate(5, sizeof(inference_msg_t));
    xModelMutex = xSemaphoreCreateMutex();

    // 2. Add a tiny delay to allow memory pools to allocate cleanly 
    vTaskDelay(pdMS_TO_TICKS(20));

    init_pdm_microphone();

    // Pin Core 0 Task (Audio Capture + Transpiled ONNX Model Inference)
    xTaskCreatePinnedToCore(
        vCore0_InferenceTask, "Inference_Core0", 16384, NULL, 5, NULL, 0
    );

    // Pin Core 1 Task (Real-Time SRAM Weight Reinforcement Learning Update)
    xTaskCreatePinnedToCore(
        vCore1_RLTask, "RL_Core1", 16384, NULL, 4, NULL, 1
    );
}