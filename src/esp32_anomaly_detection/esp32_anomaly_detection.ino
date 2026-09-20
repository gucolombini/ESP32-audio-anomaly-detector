#include <Arduino.h>
#include "driver/i2s.h"
#include <arduinoFFT.h>

// ============================================================
// CONFIGURAÇÕES
// ============================================================

#define I2S_PORT I2S_NUM_0

#define I2S_WS   25
#define I2S_SCK  26
#define I2S_SD   33

#define LED_PIN 2
#define LED_ON_TIME 500

#define SAMPLE_RATE 16000
#define BUFFER_SIZE 1024

#define NUM_AUDIO_BUFFERS 4

#define FFT_SIZE 1024

volatile uint32_t ledOffTime = 0;

// ============================================================
// BUFFER DE ÁUDIO
// ============================================================

int32_t audioBuffers[NUM_AUDIO_BUFFERS][BUFFER_SIZE];

volatile int writeIndex = 0;
volatile int readIndex = 0;

// ============================================================
// METADADOS DE LATÊNCIA
// ============================================================

// Para cada buffer guardamos os tempos da captura.

uint32_t captureStartUs[NUM_AUDIO_BUFFERS];
uint32_t captureEndUs[NUM_AUDIO_BUFFERS];
uint32_t captureDurationUs[NUM_AUDIO_BUFFERS];

// ============================================================
// ESTRUTURA DE FEATURES
// ============================================================

struct AudioFeatures {

  float rms;
  float peak;

  float spectralCentroid;

  uint32_t captureEndUs;
  uint32_t featureStartUs;
  uint32_t featureEndUs;

  uint32_t captureDurationUs;
};

// ============================================================
// SINCRONIZAÇÃO
// ============================================================

SemaphoreHandle_t bufferReadySemaphore;
SemaphoreHandle_t bufferMutex;

QueueHandle_t featureQueue;

// ============================================================
// ESTATÍSTICAS
// ============================================================

volatile uint32_t buffersCaptured = 0;
volatile uint32_t buffersProcessed = 0;
volatile uint32_t detectionsPerformed = 0;

// Estatísticas acumuladas

uint64_t totalCaptureUs = 0;
uint64_t totalFeatureUs = 0;
uint64_t totalFeatureWaitUs = 0;
uint64_t totalDetectionWaitUs = 0;
uint64_t totalDetectionUs = 0;
uint64_t totalEndToEndUs = 0;

uint32_t maxCaptureUs = 0;
uint32_t maxFeatureUs = 0;
uint32_t maxFeatureWaitUs = 0;
uint32_t maxDetectionWaitUs = 0;
uint32_t maxDetectionUs = 0;
uint32_t maxEndToEndUs = 0;

// ============================================================
// CONFIGURAÇÃO DO I2S
// ============================================================

void setupI2S() {

  i2s_config_t i2s_config = {

    .mode = (i2s_mode_t)(
      I2S_MODE_MASTER |
      I2S_MODE_RX
    ),

    .sample_rate = SAMPLE_RATE,

    .bits_per_sample =
      I2S_BITS_PER_SAMPLE_32BIT,

    .channel_format =
      I2S_CHANNEL_FMT_ONLY_LEFT,

    .communication_format =
      I2S_COMM_FORMAT_I2S,

    .intr_alloc_flags =
      ESP_INTR_FLAG_LEVEL1,

    .dma_buf_count = 8,

    .dma_buf_len = 256,

    .use_apll = false,

    .tx_desc_auto_clear = false,

    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {

    .bck_io_num = I2S_SCK,

    .ws_io_num = I2S_WS,

    .data_out_num =
      I2S_PIN_NO_CHANGE,

    .data_in_num = I2S_SD
  };

  i2s_driver_install(
    I2S_PORT,
    &i2s_config,
    0,
    NULL
  );

  i2s_set_pin(
    I2S_PORT,
    &pin_config
  );

  i2s_zero_dma_buffer(I2S_PORT);

  Serial.println("I2S configurado!");
}

// ============================================================
// TASK 1 — CAPTURA
// ============================================================

void TaskAudioCapture(void *parameter) {

  Serial.println(
    "TaskAudioCapture iniciada"
  );

  while (true) {

    // Verifica se o buffer circular está cheio

    if (
      (writeIndex + 1) % NUM_AUDIO_BUFFERS
      == readIndex
    ) {

      vTaskDelay(
        pdMS_TO_TICKS(1)
      );

      continue;
    }

    int currentBuffer;

    // Protege acesso ao índice

    xSemaphoreTake(
      bufferMutex,
      portMAX_DELAY
    );

    currentBuffer = writeIndex;

    xSemaphoreGive(
      bufferMutex
    );

    // --------------------------------------------------------
    // INÍCIO DA CAPTURA
    // --------------------------------------------------------

    uint32_t startUs = micros();

    captureStartUs[currentBuffer] =
      startUs;

    size_t bytesRead = 0;

    esp_err_t result = i2s_read(

      I2S_PORT,

      audioBuffers[currentBuffer],

      sizeof(audioBuffers[currentBuffer]),

      &bytesRead,

      portMAX_DELAY
    );

    // --------------------------------------------------------
    // FIM DA CAPTURA
    // --------------------------------------------------------

    uint32_t endUs = micros();

    if (result == ESP_OK) {

      captureEndUs[currentBuffer] =
        endUs;

      captureDurationUs[currentBuffer] =
        endUs - startUs;

      // Avança buffer

      xSemaphoreTake(
        bufferMutex,
        portMAX_DELAY
      );

      writeIndex =
        (writeIndex + 1)
        % NUM_AUDIO_BUFFERS;

      buffersCaptured++;

      xSemaphoreGive(
        bufferMutex
      );

      // Informa Task 2

      xSemaphoreGive(
        bufferReadySemaphore
      );
    }
  }
}

// ============================================================
// EXTRAÇÃO DE FEATURES
// ============================================================

double vReal[FFT_SIZE];
double vImag[FFT_SIZE];

ArduinoFFT<double> FFT = ArduinoFFT<double>(
  vReal,
  vImag,
  FFT_SIZE,
  SAMPLE_RATE
);

AudioFeatures calculateFeatures(int32_t *buffer) {

  AudioFeatures features;

  double sumSquares = 0.0;
  float peak = 0.0;

  // ============================================================
  // PREPARAÇÃO DOS DADOS
  // ============================================================

  for (int i = 0; i < BUFFER_SIZE; i++) {

    float sample =
      (float)(buffer[i] >> 8);

    sample /= 8388608.0f;

    // RMS
    sumSquares += sample * sample;

    // Peak
    float absoluteSample = fabs(sample);

    if (absoluteSample > peak) {
      peak = absoluteSample;
    }

    // FFT
    vReal[i] = sample;
    vImag[i] = 0.0;
  }

  // ============================================================
  // RMS
  // ============================================================

  features.rms =
    sqrt(sumSquares / BUFFER_SIZE);

  // ============================================================
  // PEAK
  // ============================================================

  features.peak = peak;

  // ============================================================
  // FFT
  // ============================================================

  FFT.windowing(
    FFTWindow::Hamming,
    FFTDirection::Forward
  );

  FFT.compute(
    FFTDirection::Forward
  );

  FFT.complexToMagnitude();

  // ============================================================
  // SPECTRAL CENTROID
  // ============================================================

  double weightedSum = 0.0;
  double magnitudeSum = 0.0;

  for (int i = 1; i < FFT_SIZE / 2; i++) {

    double frequency =
      ((double)i * SAMPLE_RATE) /
      FFT_SIZE;

    double magnitude =
      vReal[i];

    weightedSum +=
      frequency * magnitude;

    magnitudeSum +=
      magnitude;
  }

  if (magnitudeSum > 0.0) {

    features.spectralCentroid =
      weightedSum / magnitudeSum;

  } else {

    features.spectralCentroid = 0.0;
  }

  return features;
}

// ============================================================
// TASK 2 — FEATURES
// ============================================================

void TaskFeatureExtraction(
  void *parameter
) {

  Serial.println(
    "TaskFeatureExtraction iniciada"
  );

  while (true) {

    // Espera um buffer ficar pronto

    if (
      xSemaphoreTake(
        bufferReadySemaphore,
        portMAX_DELAY
      )
      == pdTRUE
    ) {

      // ------------------------------------------------------
      // MOMENTO EM QUE COMEÇAMOS O PROCESSAMENTO
      // ------------------------------------------------------

      uint32_t featureStartUs =
        micros();

      int currentBuffer;

      xSemaphoreTake(
        bufferMutex,
        portMAX_DELAY
      );

      currentBuffer =
        readIndex;

      xSemaphoreGive(
        bufferMutex
      );

      // ------------------------------------------------------
      // CALCULA FEATURES
      // ------------------------------------------------------

      AudioFeatures features =
        calculateFeatures(
          audioBuffers[currentBuffer]
        );

      uint32_t featureEndUs =
        micros();

      // ------------------------------------------------------
      // PREENCHE METADADOS
      // ------------------------------------------------------

      features.captureEndUs =
        captureEndUs[currentBuffer];

      features.captureDurationUs =
        captureDurationUs[currentBuffer];

      features.featureStartUs =
        featureStartUs;

      features.featureEndUs =
        featureEndUs;

      // ------------------------------------------------------
      // ATUALIZA ESTATÍSTICAS
      // ------------------------------------------------------

      uint32_t featureWait =
        featureStartUs -
        captureEndUs[currentBuffer];

      uint32_t featureDuration =
        featureEndUs -
        featureStartUs;

      totalFeatureWaitUs +=
        featureWait;

      totalFeatureUs +=
        featureDuration;

      if (
        featureWait > maxFeatureWaitUs
      ) {

        maxFeatureWaitUs =
          featureWait;
      }

      if (
        featureDuration > maxFeatureUs
      ) {

        maxFeatureUs =
          featureDuration;
      }

      // ------------------------------------------------------
      // AVANÇA BUFFER
      // ------------------------------------------------------

      xSemaphoreTake(
        bufferMutex,
        portMAX_DELAY
      );

      readIndex =
        (readIndex + 1)
        % NUM_AUDIO_BUFFERS;

      buffersProcessed++;

      xSemaphoreGive(
        bufferMutex
      );

      // ------------------------------------------------------
      // ENVIA FEATURES
      // ------------------------------------------------------

      xQueueSend(
        featureQueue,
        &features,
        portMAX_DELAY
      );
    }
  }
}

// ============================================================
// TASK 3 — DETECÇÃO
// ============================================================

void TaskAnomalyDetection(
  void *parameter
) {

  Serial.println(
    "TaskAnomalyDetection iniciada"
  );

  AudioFeatures features;

  while (true) {

    if (
      digitalRead(LED_PIN) == HIGH &&
      millis() >= ledOffTime
    ) {

      digitalWrite(
        LED_PIN,
        LOW
      );
    }

    if (
      xQueueReceive(
        featureQueue,
        &features,
        portMAX_DELAY
      )
      == pdTRUE
    ) {

      // ------------------------------------------------------
      // MOMENTO EM QUE A DETECÇÃO COMEÇA
      // ------------------------------------------------------

      uint32_t detectionStartUs =
        micros();

      // Tempo esperando na Queue

      uint32_t detectionWait =
        detectionStartUs -
        features.featureEndUs;

      // ------------------------------------------------------
      // CLASSIFICADOR PROVISÓRIO
      // ------------------------------------------------------

      const float ANOMALY_THRESHOLD =
        0.08f;

      bool anomaly =
        features.rms >
        ANOMALY_THRESHOLD;

      // ------------------------------------------------------
      // FIM DA DETECÇÃO
      // ------------------------------------------------------

      uint32_t detectionEndUs =
        micros();

      uint32_t detectionDuration =
        detectionEndUs -
        detectionStartUs;

      // ------------------------------------------------------
      // LATÊNCIA TOTAL
      // ------------------------------------------------------

      uint32_t totalLatency =
        detectionEndUs -
        (features.featureStartUs -
         (features.featureStartUs -
          features.captureEndUs));

      /*
       * Para simplificar:
       * a latência total relevante aqui será
       * desde o fim da captura até a decisão.
       */

      uint32_t endToEnd =
        detectionEndUs -
        features.captureEndUs;

      // ------------------------------------------------------
      // ESTATÍSTICAS
      // ------------------------------------------------------

      totalDetectionWaitUs +=
        detectionWait;

      totalDetectionUs +=
        detectionDuration;

      totalEndToEndUs +=
        endToEnd;

      if (
        detectionWait >
        maxDetectionWaitUs
      ) {

        maxDetectionWaitUs =
          detectionWait;
      }

      if (
        detectionDuration >
        maxDetectionUs
      ) {

        maxDetectionUs =
          detectionDuration;
      }

      if (
        endToEnd >
        maxEndToEndUs
      ) {

        maxEndToEndUs =
          endToEnd;
      }

      detectionsPerformed++;

      // ------------------------------------------------------
      // ALERTA
      // ------------------------------------------------------

      if (anomaly) {

        digitalWrite(
          LED_PIN,
          HIGH
        );

        ledOffTime =
          millis() + LED_ON_TIME;

        Serial.println(
          ">>> POSSIVEL ANOMALIA <<<"
        );

      } else {

        Serial.println(
          "Som normal"
        );
      }

      // ------------------------------------------------------
      // MOSTRA LATÊNCIAS
      // ------------------------------------------------------

      Serial.print(
        "RMS: "
      );

      Serial.print(
        features.rms,
        5
      );

      Serial.print(
        " | Centroid: "
      );

      Serial.print(
        features.spectralCentroid,
        2
      );

      Serial.print(
        " Hz"
      );

      Serial.print(
        " | Captura: "
      );

      Serial.print(
        features.captureDurationUs
        / 1000.0,
        2
      );

      Serial.print(
        " ms | Feature wait: "
      );

      Serial.print(
        detectionWait / 1000.0,
        2
      );

      Serial.print(
        " ms | Feature: "
      );

      Serial.print(
        (features.featureEndUs -
         features.featureStartUs)
        / 1000.0,
        2
      );

      Serial.print(
        " ms | Detection: "
      );

      Serial.print(
        detectionDuration / 1000.0,
        2
      );

      Serial.print(
        " ms | Total pós-captura: "
      );

      Serial.print(
        endToEnd / 1000.0,
        2
      );

      Serial.println(
        " ms"
      );
    }
  }
}

// ============================================================
// TASK MONITOR
// ============================================================

void TaskMonitor(
  void *parameter
) {

  while (true) {

    Serial.println();
    Serial.println(
      "========== PERFORMANCE =========="
    );

    uint32_t count =
      detectionsPerformed;

    if (count > 0) {

      Serial.print(
        "Media espera feature: "
      );

      Serial.print(
        (totalFeatureWaitUs /
         count) / 1000.0,
        3
      );

      Serial.println(
        " ms"
      );

      Serial.print(
        "Media processamento feature: "
      );

      Serial.print(
        (totalFeatureUs /
         count) / 1000.0,
        3
      );

      Serial.println(
        " ms"
      );

      Serial.print(
        "Media espera detection: "
      );

      Serial.print(
        (totalDetectionWaitUs /
         count) / 1000.0,
        3
      );

      Serial.println(
        " ms"
      );

      Serial.print(
        "Media detection: "
      );

      Serial.print(
        (totalDetectionUs /
         count) / 1000.0,
        3
      );

      Serial.println(
        " ms"
      );

      Serial.print(
        "Media total pos-captura: "
      );

      Serial.print(
        (totalEndToEndUs /
         count) / 1000.0,
        3
      );

      Serial.println(
        " ms"
      );

      Serial.println();

      Serial.print(
        "Max feature: "
      );

      Serial.print(
        maxFeatureUs / 1000.0,
        3
      );

      Serial.println(
        " ms"
      );

      Serial.print(
        "Max detection: "
      );

      Serial.print(
        maxDetectionUs / 1000.0,
        3
      );

      Serial.println(
        " ms"
      );

      Serial.print(
        "Max total pos-captura: "
      );

      Serial.print(
        maxEndToEndUs / 1000.0,
        3
      );

      Serial.println(
        " ms"
      );
    }

    Serial.println(
      "================================="
    );

    Serial.print(
      "Buffers capturados: "
    );

    Serial.println(
      buffersCaptured
    );

    Serial.print(
      "Buffers processados: "
    );

    Serial.println(
      buffersProcessed
    );

    Serial.print(
      "Heap livre: "
    );

    Serial.println(
      ESP.getFreeHeap()
    );

    vTaskDelay(
      pdMS_TO_TICKS(5000)
    );
  }
}

// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println(
    "========================================"
  );

  Serial.println(
    " DETECTOR DE ANOMALIAS ACUSTICAS"
  );

  Serial.println(
    " ESP32 + INMP441 + FreeRTOS"
  );

  Serial.println(
    " Versao com medicao de latencia"
  );

  Serial.println(
    "========================================"
  );

  pinMode(
    LED_PIN,
    OUTPUT
  );

  digitalWrite(
    LED_PIN,
    LOW
  );

  // I2S

  setupI2S();

  // Mutex

  bufferMutex =
    xSemaphoreCreateMutex();

  // Semáforo

  bufferReadySemaphore =
    xSemaphoreCreateCounting(
      NUM_AUDIO_BUFFERS,
      0
    );

  // Queue

  featureQueue =
    xQueueCreate(
      8,
      sizeof(AudioFeatures)
    );

  if (
    bufferMutex == NULL ||
    bufferReadySemaphore == NULL ||
    featureQueue == NULL
  ) {

    Serial.println(
      "ERRO CRITICO: "
      "falha ao criar sincronizacao!"
    );

    while (true) {
      delay(1000);
    }
  }

  // ==========================================================
  // TASKS
  // ==========================================================

  xTaskCreate(
    TaskAudioCapture,
    "AudioCapture",
    4096,
    NULL,
    3,
    NULL
  );

  xTaskCreate(
    TaskFeatureExtraction,
    "FeatureExtraction",
    4096,
    NULL,
    2,
    NULL
  );

  xTaskCreate(
    TaskAnomalyDetection,
    "AnomalyDetection",
    4096,
    NULL,
    1,
    NULL
  );

  xTaskCreate(
    TaskMonitor,
    "Monitor",
    2048,
    NULL,
    1,
    NULL
  );

  Serial.println(
    "Todas as tasks foram criadas!"
  );
}

// ============================================================
// LOOP
// ============================================================

void loop() {

  vTaskDelay(
    pdMS_TO_TICKS(1000)
  );
}