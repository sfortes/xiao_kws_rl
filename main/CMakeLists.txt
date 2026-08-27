idf_component_register(SRCS "main.c" "mfcc_dsp.c" "kws_model.c"
                       INCLUDE_DIRS "."
                       REQUIRES driver esp_ringbuf esp_driver_i2s)
