idf_component_register(
    SRCS 
        "main.c"
        "mfcc_dsp.c"
        "kws_model.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES 
        driver
        esp_driver_gpio
        esp_driver_i2s
        esp_ringbuf
        freertos
)