# Detector embarcado de tosse — ESP32 + INMP441

O projeto detecta tosse em janelas de áudio de 1 s. É um caso de uso de monitoramento não médico (por exemplo, conforto ambiental ou contagem exploratória); o resultado **não é diagnóstico clínico**.

## Arquitetura

O firmware contém três tarefas FreeRTOS com prioridades distintas:

1. `audio_capture` (alta, core 0): recebe PCM do INMP441 por I2S/DMA e preenche um dos dois buffers.
2. `mfcc` (média, core 1): consome o buffer pronto e extrai 49×13 MFCCs.
3. `detect` (baixa, core 1): quantiza as features, executa a MLP int8 no TFLite Micro e aciona o LED.

As filas `free_audio_q` e `ready_audio_q` transferem a propriedade dos buffers; portanto captura e MFCC nunca acessam o mesmo buffer ao mesmo tempo. `feature_q`, de comprimento 1, usa overwrite para evitar backlog e privilegiar a janela mais recente. Veja [docs/rtos_tasks.svg](docs/rtos_tasks.svg).

## 1. Ligação

| INMP441 | ESP32 padrão |
|---|---:|
| VDD | 3V3 |
| GND | GND |
| SCK/BCLK | GPIO 26 |
| WS/LRCL | GPIO 25 |
| SD | GPIO 33 |
| L/R | GND (canal esquerdo) |

O LED está no GPIO 2. Altere os pinos em `esp32-anomaly-detector/include/app_config.h` se necessário.

## 2. Coleta com o mesmo microfone

Use Python 3.12 (TensorFlow 2.18 não suporta Python 3.14):

```bash
pyenv local 3.12.12
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cd esp32-anomaly-detector
pio run -e esp32dev-collect -t upload
cd ..
python tools/collect_serial.py background --port /dev/ttyUSB0 --count 100
python tools/collect_serial.py cough --port /dev/ttyUSB0 --count 100
```

Grave em várias distâncias, orientações, horários e níveis de ruído. Para `background`, inclua fala, palmas, passos, porta, música e silêncio; exemplos difíceis reduzem falsos positivos. Não grave apenas janelas perfeitamente centralizadas. Obtenha consentimento e evite armazenar conversas.

## 3. Treino, exportação e teste

Execute a partir da raiz:

```bash
python tools/train.py
python tools/export_model.py
python tools/evaluate_tflite.py
pytest -q
```

São gerados `models/cough_detector.onnx` (entregável), `models/cough_detector_int8.tflite`, `models/metrics.json` e o array C++ em `esp32-anomaly-detector/src/model_data.cpp`. O split atual é por arquivo. Para uma avaliação honesta, recomenda-se coletar sessões/pessoas diferentes e reservar uma sessão inteira para teste, evitando que trechos vizinhos apareçam em treino e teste.

## 4. Firmware final e latência

```bash
cd esp32-anomaly-detector
pio run -e esp32dev -t upload
pio device monitor
```

Cada inferência escreve uma linha com latência, nível do áudio, saturação da entrada quantizada, probabilidade e alerta. Para reproduzir as 100 medições do relatório:

```bash
cd ..
python tools/measure_latency.py --port /dev/ttyUSB0 --count 100
```

O arquivo é salvo em `models/latency.csv`. A latência total inclui a janela de aproximadamente 1 s; não deve ser confundida com os cerca de 3,7 ms gastos apenas na inferência.

## Resultados atuais

- dados: 221 janelas de background e 47 de tosse;
- teste independente por arquivo: 82,9% de acurácia e 85,7% de recall de tosse;
- modelo TFLite int8: 23.408 bytes, 20.482 parâmetros;
- latência média no ESP32: 400,30 ms para MFCC, 3,66 ms para inferência e 1.384,29 ms ponta a ponta;
- firmware: 48,6% da RAM e 13,8% da partição de aplicação.

As métricas completas, matriz de confusão, limitações do split e discussão estão em [docs/REPORT.md](docs/REPORT.md).

## Critérios atendidos

- captura contínua I2S/DMA do INMP441;
- três tarefas sincronizadas com filas e propriedade explícita de buffers;
- MFCC no edge e modelo pré-treinado TFLite Micro int8;
- exportação ONNX;
- medição por etapa e ponta a ponta;
- alerta no LED e limiar configurável;
- teste host de features e avaliação do artefato TFLite.

O firmware propositalmente recusa inferência enquanto `model_data.cpp` for o placeholder. Isso impede uma demonstração enganosa antes de o modelo ser treinado com os dados reais.
