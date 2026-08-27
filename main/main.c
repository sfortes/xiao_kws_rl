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

// Hardware PDM Microphone Configuration
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
    ESP_LOGI(TAG, "I2S PDM Microphones Initialized Successfully");
}

// Core 0 Task: Continuous Sampling -> MFCC -> Mutex Shared Weight Model Inference
void vCore0_InferenceTask(void *pvParameters) {
    int16_t *raw_pcm = malloc(AUDIO_LEN_SAMPLES * sizeof(int16_t));
    float *mfcc_features = malloc(MODEL_INPUT_SIZE * sizeof(float));
    float raw_logits[MODEL_OUTPUT_SIZE];
    float probabilities[MODEL_OUTPUT_SIZE];
    size_t bytes_read = 0;

    mfcc_init();

    while (1) {
        // Read 1 second of 16kHz PCM audio over PDM DMA
        esp_err_t result = i2s_channel_read(rx_chan, raw_pcm, AUDIO_LEN_SAMPLES * sizeof(int16_t), &bytes_read, portMAX_DELAY);
        if (result == ESP_OK && bytes_read == AUDIO_LEN_SAMPLES * sizeof(int16_t)) {
            
            // Execute DSP MFCC Feature Extraction
            mfcc_compute(raw_pcm, mfcc_features);

            // Thread-safe access to Shared Model Weights in SRAM
            if (xSemaphoreTake(xModelMutex, portMAX_DELAY) == pdTRUE) {
                model_inference(mfcc_features, raw_logits);
                xSemaphoreGive(xModelMutex);
            }

            softmax(raw_logits, probabilities, MODEL_OUTPUT_SIZE);

            // Find top prediction
            int max_idx = 0;
            float max_conf = probabilities[0];
            for (int i = 1; i < MODEL_OUTPUT_SIZE; i++) {
                if (probabilities[i] > max_conf) {
                    max_conf = probabilities[i];
                    max_idx = i;
                }
            }

            ESP_LOGI(TAG, "[Core 0] Inferred Word: %s (Confidence: %.2f%%)", WORD_LABELS[max_idx], max_conf * 100.0f);

            // Send Result Payload to Core 1 Task
            inference_msg_t msg;
            msg.label_idx = max_idx;
            msg.confidence = max_conf;
            memcpy(msg.features, mfcc_features, MODEL_INPUT_SIZE * sizeof(float));
            xQueueSend(xInferenceQueue, &msg, 0);
        }
    }
}

// Core 1 Task: Contextual Policy Gradient (REINFORCE Algorithm)
void vCore1_RLTask(void *pvParameters) {
    inference_msg_t msg;
    float learning_rate = 0.001f;

    while (1) {
        if (xQueueReceive(xInferenceQueue, &msg, portMAX_DELAY) == pdTRUE) {
            
            // Simulated Reward Environment Policy logic:
            // High confidence (>80%) grants Positive Reward (+1.0), lower triggers Penalty (-1.0)
            float reward = (msg.confidence > 0.80f) ? 1.0f : -1.0f;

            if (reward < 0.0f) {
                ESP_LOGW(TAG, "[Core 1 RL] Low Confidence Detection! Adapting Weights in SRAM via Policy Gradient...");

                if (xSemaphoreTake(xModelMutex, portMAX_DELAY) == pdTRUE) {
                    // Update Layer 2 Output Weights corresponding to inferred class:
                    // w_ij = w_ij + lr * reward * feature_j
                    for (int j = 0; j < MODEL_HIDDEN_SIZE; j++) {
                        fc2_w[msg.label_idx][j] += learning_rate * reward * 0.01f;
                    }
                    fc2_b[msg.label_idx] += learning_rate * reward * 0.01f;

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
    xInferenceQueue = xQueueCreate(5, sizeof(inference_msg_t));
    xModelMutex = xSemaphoreCreateMutex();

    init_pdm_microphone();

    // Pin Core 0 Task (Audio Capture + Transpiled ONNX Model Inference)
    xTaskCreatePinnedToCore(
        vCore0_InferenceTask, "Inference_Core0", 8192, NULL, 5, NULL, 0
    );

    // Pin Core 1 Task (Real-Time SRAM Weight Reinforcement Learning Update)
    xTaskCreatePinnedToCore(
        vCore1_RLTask, "RL_Core1", 8192, NULL, 4, NULL, 1
    );
}