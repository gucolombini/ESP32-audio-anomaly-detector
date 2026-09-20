#pragma once

#define SAMPLE_RATE_HZ 16000
#define AUDIO_SAMPLES 16000
#define FRAME_LENGTH 480
#define FRAME_STEP 320
#define FFT_SIZE 512
#define MEL_BINS 26
#define MFCC_COUNT 13
#define MFCC_FRAMES 49

// Ajuste estes pinos à sua montagem.
#define I2S_PORT I2S_NUM_0
#define I2S_BCLK_GPIO 26
#define I2S_WS_GPIO 25
#define I2S_DATA_GPIO 33
#define ALERT_LED_GPIO 2

#define COUGH_THRESHOLD 0.50f
#define ALERT_HOLD_MS 500
#define TENSOR_ARENA_BYTES (70 * 1024)
