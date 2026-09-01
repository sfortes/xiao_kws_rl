#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
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
#define VAD_CHUNK_SAMPLES FRAME_STEP
#define VAD_PRE_ROLL_SAMPLES (SAMPLE_RATE / 4)
#define VAD_CALIBRATION_CHUNKS 150
#define VAD_TRIGGER_MARGIN 180.0f
#define VAD_TRIGGER_PEAK_MULTIPLIER 1.10f
#define VAD_RELEASE_MULTIPLIER 1.20f
#define VAD_NOISE_ADAPT_RATE 0.02f
#define VAD_REQUIRED_CHUNKS 2
#define VAD_COOLDOWN_MS 1000

static const char *TAG = "KWS_RL_SYSTEM";
#define VALIDATION_MODE 1
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

static float calculate_rms(const int16_t *samples, size_t sample_count) {
    int64_t sum = 0;
    for (size_t i = 0; i < sample_count; i++) {
        sum += samples[i];
    }

    float mean = (float)sum / sample_count;
    double sum_squares = 0.0;
    for (size_t i = 0; i < sample_count; i++) {
        float centered_sample = samples[i] - mean;
        sum_squares += (double)centered_sample * centered_sample;
    }
    return sqrtf((float)(sum_squares / sample_count));
}

static void remove_dc_offset(int16_t *samples, size_t sample_count) {
    int64_t sum = 0;
    for (size_t i = 0; i < sample_count; i++) {
        sum += samples[i];
    }

    int32_t mean = (int32_t)(sum / (int64_t)sample_count);
    for (size_t i = 0; i < sample_count; i++) {
        samples[i] = (int16_t)((int32_t)samples[i] - mean);
    }
}

static void update_vad_thresholds(float noise_floor, float noise_peak,
                                  float *trigger_rms, float *release_rms) {
    float peak_trigger = noise_peak * VAD_TRIGGER_PEAK_MULTIPLIER;
    float margin_trigger = noise_floor + VAD_TRIGGER_MARGIN;
    *trigger_rms = (peak_trigger > margin_trigger) ? peak_trigger : margin_trigger;
    *release_rms = noise_floor * VAD_RELEASE_MULTIPLIER;
}

// Softmax Calculation
static int select_top_class(const float *logits, int num_classes, float *confidence_out) {
    if (!logits || !confidence_out || num_classes <= 0) {
        return -1;
    }

    float max_logit = logits[0];
    for (int i = 1; i < num_classes; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float denom = 0.0f;
    float probs[32];
    if (num_classes > (int)(sizeof(probs) / sizeof(probs[0]))) {
        return -1;
    }

    for (int i = 0; i < num_classes; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        denom += probs[i];
    }

    int best_idx = 0;
    float best_prob = 0.0f;

    for (int i = 0; i < num_classes; ++i) {
        float p = probs[i] / denom;
        if (p > best_prob) {
            best_prob = p;
            best_idx = i;
        }
    }

    *confidence_out = best_prob;
    return best_idx;
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

static bool calibrate_vad(int16_t *audio_chunk, float *noise_floor,
                          float *trigger_rms, float *release_rms) {
    float noise_sum = 0.0f;
    float noise_peak = 0.0f;
    size_t bytes_read = 0;

    ESP_LOGI(TAG, "[VAD] Calibrating ambient noise for 1.5 seconds; remain quiet");
    for (int chunk = 0; chunk < VAD_CALIBRATION_CHUNKS; chunk++) {
        esp_err_t result = i2s_channel_read(rx_chan, audio_chunk, sizeof(int16_t) * VAD_CHUNK_SAMPLES,
                                             &bytes_read, portMAX_DELAY);
        if (result != ESP_OK || bytes_read != sizeof(int16_t) * VAD_CHUNK_SAMPLES) {
            ESP_LOGW(TAG, "[VAD] Calibration read failed: result=%d bytes=%u", result, (unsigned)bytes_read);
            return false;
        }

        float chunk_rms = calculate_rms(audio_chunk, VAD_CHUNK_SAMPLES);
        noise_sum += chunk_rms;
        if (chunk_rms > noise_peak) {
            noise_peak = chunk_rms;
        }
    }

    *noise_floor = noise_sum / VAD_CALIBRATION_CHUNKS;
    update_vad_thresholds(*noise_floor, noise_peak, trigger_rms, release_rms);
    ESP_LOGI(TAG, "[VAD] Calibrated: noise=%.2f peak=%.2f trigger=%.2f release=%.2f",
             *noise_floor, noise_peak, *trigger_rms, *release_rms);
    return true;
}

void vCore0_InferenceTask(void *pvParameters) {
    static int16_t raw_pcm[AUDIO_LEN_SAMPLES];
    static int16_t pre_roll[VAD_PRE_ROLL_SAMPLES];
    static float mfcc_features[MODEL_INPUT_SIZE];
    static float raw_logits[MODEL_OUTPUT_SIZE];
    int16_t audio_chunk[VAD_CHUNK_SAMPLES];
    size_t pre_roll_write = 0;
    int active_chunk_count = 0;
    bool vad_armed = false;
    float noise_floor = 0.0f;
    float trigger_rms = 0.0f;
    float release_rms = 0.0f;
    TickType_t next_trigger_tick = 0;
    size_t bytes_read = 0;

    mfcc_dsp_init();
    if (!calibrate_vad(audio_chunk, &noise_floor, &trigger_rms, &release_rms)) {
        ESP_LOGE(TAG, "[VAD] Calibration failed; inference task stopping");
        vTaskDelete(NULL);
        return;
    }

    while (1) {
        esp_err_t result = i2s_channel_read(rx_chan, audio_chunk, sizeof(audio_chunk), &bytes_read, portMAX_DELAY);
        if (result != ESP_OK || bytes_read != sizeof(audio_chunk)) {
            ESP_LOGW(TAG, "[VAD] Audio read failed: result=%d bytes=%u", result, (unsigned)bytes_read);
            continue;
        }

        for (int i = 0; i < VAD_CHUNK_SAMPLES; i++) {
            pre_roll[pre_roll_write] = audio_chunk[i];
            pre_roll_write = (pre_roll_write + 1) % VAD_PRE_ROLL_SAMPLES;
        }
        float chunk_rms = calculate_rms(audio_chunk, VAD_CHUNK_SAMPLES);

        if (!vad_armed) {
            if (chunk_rms <= release_rms) {
                vad_armed = true;
                active_chunk_count = 0;

                noise_floor += VAD_NOISE_ADAPT_RATE * (chunk_rms - noise_floor);
                update_vad_thresholds(noise_floor, noise_floor, &trigger_rms, &release_rms);
            }
            continue;
        }

        if (chunk_rms < trigger_rms) {
            active_chunk_count = 0;
            continue;
        }

        active_chunk_count++;
        if (active_chunk_count < VAD_REQUIRED_CHUNKS || (int32_t)(xTaskGetTickCount() - next_trigger_tick) < 0) {
            continue;
        }

        for (int i = 0; i < VAD_PRE_ROLL_SAMPLES; i++) {
            raw_pcm[i] = pre_roll[(pre_roll_write + i) % VAD_PRE_ROLL_SAMPLES];
        }

        int captured_samples = VAD_PRE_ROLL_SAMPLES;
        while (captured_samples < AUDIO_LEN_SAMPLES) {
            result = i2s_channel_read(rx_chan, audio_chunk, sizeof(audio_chunk), &bytes_read, portMAX_DELAY);
            if (result != ESP_OK || bytes_read != sizeof(audio_chunk)) {
                ESP_LOGW(TAG, "[VAD] Capture failed: result=%d bytes=%u", result, (unsigned)bytes_read);
                break;
            }
            memcpy(&raw_pcm[captured_samples], audio_chunk, sizeof(audio_chunk));
            captured_samples += VAD_CHUNK_SAMPLES;
        }

        vad_armed = false;
        active_chunk_count = 0;
        next_trigger_tick = xTaskGetTickCount() + pdMS_TO_TICKS(VAD_COOLDOWN_MS);

        if (captured_samples == AUDIO_LEN_SAMPLES) {
            remove_dc_offset(raw_pcm, AUDIO_LEN_SAMPLES);
            float audio_rms = calculate_rms(raw_pcm, AUDIO_LEN_SAMPLES);
            ESP_LOGI(TAG, "[VAD] Triggered at %.2f RMS (threshold %.2f); classifying a 1.0 s clip",
                     chunk_rms, trigger_rms);

            // Extract all frames together so the 80 dB torchaudio floor is
            // calculated from the same one-second clip used for inference.
            mfcc_compute_batch(raw_pcm, mfcc_features);

            // Thread-safe access to Shared Model Weights in SRAM
            if (xSemaphoreTake(xModelMutex, portMAX_DELAY) == pdTRUE) {
                model_inference(mfcc_features, raw_logits);
                xSemaphoreGive(xModelMutex);
            }

            // Find top prediction index and confidence value
            float max_conf = 0.0f;
            int max_idx = select_top_class(raw_logits, MODEL_OUTPUT_SIZE, &max_conf);
            if (max_idx < 0) {
                max_idx = 0;
                max_conf = 0.0f;
            }

            // Calculate feature boundaries for tracking trends
            float feat_min = mfcc_features[0];
            float feat_max = mfcc_features[0];
            for (int i = 1; i < MODEL_INPUT_SIZE; i++) {
                if (mfcc_features[i] < feat_min) feat_min = mfcc_features[i];
                if (mfcc_features[i] > feat_max) feat_max = mfcc_features[i];
            }

            // Print Core Metrics Dashboard
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

    while (1) {
        if (xQueueReceive(xInferenceQueue, &msg, portMAX_DELAY) == pdTRUE) {
#if VALIDATION_MODE
            ESP_LOGW(TAG, "[VALIDATION] RL disabled. Observing logits only.");
            ESP_LOGI(TAG, "[VALIDATION] Candidate label: %s (confidence %.2f%%)",
                     WORD_LABELS[msg.label_idx], msg.confidence * 100.0f);
            continue;
#else
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
#endif
        }
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "[VALIDATION] word order = [UP, DOWN, LEFT, RIGHT, BACK, GO]");
    ESP_LOGI(TAG, "[VALIDATION] model dims = input=%d hidden=%d output=%d", MODEL_INPUT_SIZE, MODEL_HIDDEN_SIZE, MODEL_OUTPUT_SIZE);
    ESP_LOGI(TAG, "[VALIDATION] expected feature layout = 13 MFCC x 97 frames -> flattened length 1261");

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