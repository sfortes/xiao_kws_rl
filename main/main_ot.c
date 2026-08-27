#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/i2s_pdm.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "kws_model.h"
#include "mfcc_dsp.h"

#define PDM_CLK_IO          GPIO_NUM_42
#define PDM_DIN_IO          GPIO_NUM_41
#define AUDIO_LEN_SAMPLES   16000

static const char *TAG = "KWS_RL_SYSTEM";
static const char *WORD_LABELS[MODEL_OUTPUT_SIZE] = {"UP", "DOWN", "LEFT", "RIGHT", "BACK", "GO"};

// Pointer-based message passing to avoid copying 5KB structs over FreeRTOS Queues
typedef struct {
    int label_idx;
    float confidence;
    float *features; // Heap-allocated pointer managed by Core 1
} inference_msg_t;

// Double-buffered weights for lock-free Core 0 inference
static kws_model_weights_t weight_buffers[2] KWS_ALIGN32;
static volatile uint8_t active_weight_idx = 0; // Read index for Core 0
static portMUX_TYPE weight_spinlock = portMUX_INITIALIZER_UNLOCKED;

static QueueHandle_t xInferenceQueue = NULL;
static i2s_chan_handle_t rx_chan = NULL;

// Softmax Implementation
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
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .clk = PDM_CLK_IO,
            .din = PDM_DIN_IO,
            .invert_flags = { .clk_inv = false },
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx_chan, &pdm_rx_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
    ESP_LOGI(TAG, "I2S PDM Microphones Initialized");
}

// Core 0 Task: Continuous PDM DMA Sampling -> Feature Extraction -> Non-Blocking Inference
void vCore0_InferenceTask(void *pvParameters) {
    int16_t *raw_pcm = heap_caps_malloc(AUDIO_LEN_SAMPLES * sizeof(int16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    float raw_logits[MODEL_OUTPUT_SIZE];
    float probabilities[MODEL_OUTPUT_SIZE];
    size_t bytes_read = 0;

    mfcc_init();

    while (1) {
        esp_err_t result = i2s_channel_read(rx_chan, raw_pcm, AUDIO_LEN_SAMPLES * sizeof(int16_t), &bytes_read, portMAX_DELAY);
        if (result == ESP_OK && bytes_read == AUDIO_LEN_SAMPLES * sizeof(int16_t)) {
            
            // Allocate heap memory for features sent to Core 1
            float *mfcc_features = heap_caps_malloc(MODEL_INPUT_SIZE * sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!mfcc_features) {
                ESP_LOGE(TAG, "Heap allocation failed for MFCC buffer");
                continue;
            }

            mfcc_compute(raw_pcm, mfcc_features);

            // Zero-Lock Core 0 Inference reading from active buffer index
            uint8_t read_idx;
            taskENTER_CRITICAL(&weight_spinlock);
            read_idx = active_weight_idx;
            taskEXIT_CRITICAL(&weight_spinlock);

            // Model inference executes on active buffer without mutex contention
            model_inference_with_weights(&weight_buffers[read_idx], mfcc_features, raw_logits);
            softmax(raw_logits, probabilities, MODEL_OUTPUT_SIZE);

            int max_idx = 0;
            float max_conf = probabilities[0];
            for (int i = 1; i < MODEL_OUTPUT_SIZE; i++) {
                if (probabilities[i] > max_conf) {
                    max_conf = probabilities[i];
                    max_idx = i;
                }
            }

            ESP_LOGI(TAG, "[Core 0] Inferred: %s (Conf: %.2f%%)", WORD_LABELS[max_idx], max_conf * 100.0f);

            // Dispatch pointer payload to Core 1 (4 bytes enqueue instead of 5KB)
            inference_msg_t msg = {
                .label_idx = max_idx,
                .confidence = max_conf,
                .features = mfcc_features
            };

            if (xQueueSend(xInferenceQueue, &msg, 0) != pdTRUE) {
                // Queue full: free buffer immediately to prevent leaks
                free(mfcc_features);
            }
        }
    }
}

// Core 1 Task: Policy Gradient RL Weight Adaptation on Inactive Buffer
void vCore1_RLTask(void *pvParameters) {
    inference_msg_t msg;
    const float learning_rate = 0.001f;

    while (1) {
        if (xQueueReceive(xInferenceQueue, &msg, portMAX_DELAY) == pdTRUE) {
            
            float reward = (msg.confidence > 0.80f) ? 1.0f : -1.0f;

            if (reward < 0.0f) {
                ESP_LOGW(TAG, "[Core 1 RL] Low Confidence! Updating Inactive SRAM Buffer...");

                uint8_t active_idx, update_idx;
                taskENTER_CRITICAL(&weight_spinlock);
                active_idx = active_weight_idx;
                update_idx = 1 - active_idx; // Compute target buffer
                taskEXIT_CRITICAL(&weight_spinlock);

                // Copy active weights to target inactive buffer
                memcpy(&weight_buffers[update_idx], &weight_buffers[active_idx], sizeof(kws_model_weights_t));

                // Apply Policy Gradient Updates on layer 2
                for (int j = 0; j < MODEL_HIDDEN_SIZE; j++) {
                    weight_buffers[update_idx].fc2_w[msg.label_idx][j] += learning_rate * reward * 0.01f;
                }
                weight_buffers[update_idx].fc2_b[msg.label_idx] += learning_rate * reward * 0.01f;

                // Atomic Pointer Swap: Core 0 instantly points to newly updated weights
                taskENTER_CRITICAL(&weight_spinlock);
                active_weight_idx = update_idx;
                taskEXIT_CRITICAL(&weight_spinlock);

                ESP_LOGI(TAG, "[Core 1 RL] Atomic Weight Swap Complete. Active Index: %d", update_idx);
            }

            // Free feature buffer allocated by Core 0
            free(msg.features);
        }
    }
}

void app_main(void) {
    // Zero-initialize dynamic weight buffers in SRAM
    memset(&weight_buffers[0], 0, sizeof(kws_model_weights_t));
    memset(&weight_buffers[1], 0, sizeof(kws_model_weights_t));
    kws_model_init(); // Initialize baseline weights into weight_buffers[0]

    xInferenceQueue = xQueueCreate(4, sizeof(inference_msg_t));

    init_pdm_microphone();

    // Pin Inference Pipeline to Core 0 (DSP + Matrix Ops)
    xTaskCreatePinnedToCore(
        vCore0_InferenceTask, "Inference_Core0", 8192, NULL, 5, NULL, 0
    );

    // Pin Weight Reinforcement Engine to Core 1
    xTaskCreatePinnedToCore(
        vCore1_RLTask, "RL_Core1", 6144, NULL, 4, NULL, 1
    );
}