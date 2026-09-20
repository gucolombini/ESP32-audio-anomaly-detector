#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "driver/uart_vfs.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "audio_features.h"
#include "app_config.h"
#include "model_data.h"

static const char *TAG="cough_detector";
struct AudioBlock { int16_t samples[AUDIO_SAMPLES]; int64_t started_us; int64_t captured_us; float rms; int peak; };
struct FeatureBlock { float mfcc[MFCC_FRAMES*MFCC_COUNT]; int64_t started_us; int64_t captured_us; int64_t features_us; float rms; int peak; };
static AudioBlock audio_blocks[2];
static FeatureBlock extracted_features;
static FeatureBlock inference_features;
static int32_t i2s_raw_samples[256];
static QueueHandle_t free_audio_q, ready_audio_q, feature_q;
static i2s_chan_handle_t rx_channel;

static void capture_task(void *) {
    int32_t *raw=i2s_raw_samples;
#ifdef AUDIO_COLLECTION_MODE
    uint32_t sequence=0;
#endif
    while(true) {
        AudioBlock *b=nullptr; xQueueReceive(free_audio_q,&b,portMAX_DELAY);
        b->started_us=esp_timer_get_time(); int n=0;
        while(n<AUDIO_SAMPLES) {
            size_t bytes=0; int want=AUDIO_SAMPLES-n; if(want>256) want=256;
            esp_err_t err=i2s_channel_read(rx_channel,raw,want*sizeof(int32_t),&bytes,portMAX_DELAY);
            if(err==ESP_ERR_TIMEOUT) continue;
            ESP_ERROR_CHECK(err);
            int got=bytes/sizeof(int32_t);
            for(int i=0;i<got;++i) {
                // INMP441: amostra de 24 bits alinhada à esquerda; reduz para PCM16.
                int32_t s=raw[i]>>14; if(s>32767)s=32767; if(s<-32768)s=-32768;
                b->samples[n++]=(int16_t)s;
            }
        }
        b->captured_us=esp_timer_get_time();
        double energy=0.0; int peak=0;
        for(int i=0;i<AUDIO_SAMPLES;++i) {
            int sample=b->samples[i]; int magnitude=sample<0 ? -sample : sample;
            if(magnitude>peak) peak=magnitude;
            energy+=(double)sample*sample;
        }
        b->rms=(float)(sqrt(energy/AUDIO_SAMPLES)/32768.0);
        b->peak=peak;
#ifdef AUDIO_COLLECTION_MODE
        struct __attribute__((packed)) FrameHeader {
            char marker[4]; uint32_t size; uint32_t sequence; uint32_t checksum;
        } header={{'W','A','V','1'},sizeof(b->samples),sequence++,0};
        const uint8_t *pcm=reinterpret_cast<const uint8_t *>(b->samples);
        for(size_t i=0;i<sizeof(b->samples);++i) header.checksum+=pcm[i];
        fwrite(&header,1,sizeof(header),stdout);
        fwrite(pcm,1,sizeof(b->samples),stdout);
        fflush(stdout);
        xQueueSend(free_audio_q,&b,portMAX_DELAY);
#else
        xQueueSend(ready_audio_q,&b,portMAX_DELAY);
#endif
    }
}

static void feature_task(void *) {
    while(true) {
        AudioBlock *a=nullptr; xQueueReceive(ready_audio_q,&a,portMAX_DELAY);
        FeatureBlock &f=extracted_features;
        f.started_us=a->started_us; f.captured_us=a->captured_us;
        f.rms=a->rms; f.peak=a->peak;
        extract_mfcc(a->samples,f.mfcc); f.features_us=esp_timer_get_time();
        xQueueSend(free_audio_q,&a,portMAX_DELAY);
        // Fila de tamanho 1: mantém o resultado mais recente sob sobrecarga.
        xQueueOverwrite(feature_q,&f);
    }
}

static void detection_task(void *) {
    if(g_model_data_len==0) { ESP_LOGE(TAG,"Modelo ausente. Execute tools/train.py e tools/export_model.py"); vTaskDelete(nullptr); }
    const tflite::Model *model=tflite::GetModel(g_model_data);
    if(model->version()!=TFLITE_SCHEMA_VERSION) { ESP_LOGE(TAG,"Schema TFLite incompatível"); vTaskDelete(nullptr); }
    static tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddFullyConnected(); resolver.AddSoftmax();
    static uint8_t arena[TENSOR_ARENA_BYTES];
    static tflite::MicroInterpreter interpreter(model,resolver,arena,sizeof(arena));
    if(interpreter.AllocateTensors()!=kTfLiteOk) {
        ESP_LOGE(TAG,"AllocateTensors falhou: confira operadores registrados e tamanho da arena");
        vTaskDelete(nullptr);
    }
    TfLiteTensor *input=interpreter.input(0), *output=interpreter.output(0);
    ESP_LOGI(TAG,"csv,capture_ms,mfcc_ms,inference_ms,total_ms,audio_rms,audio_peak,q_clipped,cough_probability,alert");
    while(true) {
        FeatureBlock &f=inference_features;
        xQueueReceive(feature_q,&f,portMAX_DELAY);
        int clipped=0;
        for(int i=0;i<MFCC_FRAMES*MFCC_COUNT;++i) {
            int q=(int)lrintf(f.mfcc[i]/input->params.scale)+input->params.zero_point;
            if(q>127) { q=127; ++clipped; }
            if(q<-128) { q=-128; ++clipped; }
            input->data.int8[i]=(int8_t)q;
        }
        int64_t t0=esp_timer_get_time();
        if(interpreter.Invoke()!=kTfLiteOk) { ESP_LOGE(TAG,"Falha na inferência"); continue; }
        int64_t done=esp_timer_get_time();
        float p=(output->data.int8[1]-output->params.zero_point)*output->params.scale;
        bool alert=p>=COUGH_THRESHOLD; gpio_set_level((gpio_num_t)ALERT_LED_GPIO,alert);
        if(alert) vTaskDelay(pdMS_TO_TICKS(ALERT_HOLD_MS));
        gpio_set_level((gpio_num_t)ALERT_LED_GPIO,0);
        ESP_LOGI(TAG,"csv,%.2f,%.2f,%.2f,%.2f,%.5f,%d,%d,%.4f,%d",
            (f.captured_us-f.started_us)/1000.0,(f.features_us-f.captured_us)/1000.0,
            (done-t0)/1000.0,(done-f.started_us)/1000.0,f.rms,f.peak,clipped,p,alert);
    }
}

extern "C" void app_main() {
#ifdef AUDIO_COLLECTION_MODE
    // O console usa CRLF por padrão. PCM é binário e não pode sofrer LF->CRLF.
    uart_vfs_dev_port_set_tx_line_endings(CONFIG_ESP_CONSOLE_UART_NUM,ESP_LINE_ENDINGS_LF);
    setvbuf(stdout,nullptr,_IONBF,0);
#endif
    gpio_set_direction((gpio_num_t)ALERT_LED_GPIO,GPIO_MODE_OUTPUT);
    i2s_chan_config_t channel_cfg=I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT,I2S_ROLE_MASTER);
    channel_cfg.dma_desc_num=8; channel_cfg.dma_frame_num=256;
    ESP_ERROR_CHECK(i2s_new_channel(&channel_cfg,nullptr,&rx_channel));
    i2s_std_config_t std_cfg={
        .clk_cfg=I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ),
        .slot_cfg=I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,I2S_SLOT_MODE_MONO),
        .gpio_cfg={
            .mclk=I2S_GPIO_UNUSED,
            .bclk=(gpio_num_t)I2S_BCLK_GPIO,
            .ws=(gpio_num_t)I2S_WS_GPIO,
            .dout=I2S_GPIO_UNUSED,
            .din=(gpio_num_t)I2S_DATA_GPIO,
            .invert_flags={.mclk_inv=false,.bclk_inv=false,.ws_inv=false},
        },
    };
    std_cfg.slot_cfg.slot_mask=I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_channel,&std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_channel));
    free_audio_q=xQueueCreate(2,sizeof(AudioBlock*)); ready_audio_q=xQueueCreate(2,sizeof(AudioBlock*));
    feature_q=xQueueCreate(1,sizeof(FeatureBlock));
    for(auto &b:audio_blocks){AudioBlock *p=&b;xQueueSend(free_audio_q,&p,0);}
    xTaskCreatePinnedToCore(capture_task,"audio_capture",4096,nullptr,configMAX_PRIORITIES-2,nullptr,0);
#ifndef AUDIO_COLLECTION_MODE
    xTaskCreatePinnedToCore(feature_task,"mfcc",8192,nullptr,configMAX_PRIORITIES-4,nullptr,1);
    xTaskCreatePinnedToCore(detection_task,"detect",8192,nullptr,configMAX_PRIORITIES-6,nullptr,1);
#else
    ESP_LOGW(TAG,"Modo coleta: transmitindo janelas PCM16 pela serial");
#endif
}
