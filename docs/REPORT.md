# Relatório técnico — Detector embarcado de tosse

## 1. Objetivo e aplicação

O projeto implementa no ESP32 um detector binário de tosse para monitoramento exploratório de ambientes. Todo o áudio é processado localmente: o dispositivo captura uma janela, extrai MFCCs, executa uma rede neural quantizada e acende o LED quando `p(tosse) ≥ 0,50`. A aplicação não é um dispositivo médico e não deve ser usada para diagnóstico ou tratamento.

## 2. Hardware e aquisição

Foram usados ESP32, microfone digital INMP441 e o LED do GPIO 2. O INMP441 opera em I2S mono, 16 kHz, com palavras de 32 bits contendo amostras de 24 bits alinhadas à esquerda. O firmware desloca cada palavra em 14 bits e produz PCM16. Ligações: BCLK GPIO 26, WS GPIO 25, SD GPIO 33, L/R em GND, alimentação em 3,3 V.

O conjunto atual contém 268 janelas WAV de um segundo, todas PCM16 mono a 16 kHz:

- 221 exemplos de `background`, incluindo silêncio, ruído ambiente, voz e conversa;
- 47 exemplos de `cough`;
- coleta realizada com o mesmo microfone e cadeia de conversão empregados na inferência.

Os dados são desbalanceados e foram obtidos predominantemente no mesmo contexto de aquisição. O split estratificado por arquivo contém 187 amostras de treino, 40 de validação e 41 de teste. Trechos próximos de uma mesma sessão podem ser correlacionados; portanto o teste por arquivo não demonstra generalização para pessoas ou ambientes inéditos. Uma avaliação futura deve reservar sessões e participantes inteiros.

O dataset utilizado pode ser acessado [neste link do Google Drive](https://drive.google.com/file/d/1k2aA7Mzl4r3rgRwgUTb_eFnZMuIT-xe_/view?usp=sharing).

## 3. Processamento de sinal e modelo

Cada janela de 16.000 amostras gera 49 quadros de 30 ms, com passo de 20 ms. Em cada quadro são aplicados janela de Hann, FFT de 512 pontos, 26 filtros Mel, logaritmo das energias e DCT para obter 13 MFCCs. O vetor final possui 637 valores (`49 × 13`).

A rede selecionada é `637 → Dense(32, ReLU) → Dense(2, Softmax)`, com 20.482 parâmetros. Durante o treino as features são padronizadas usando média e desvio calculados exclusivamente no treino. Essa transformação foi incorporada aos pesos e bias da primeira camada, evitando novo operador e custo adicional no firmware. Foram usados class weights, dropout de 25% e early stopping.

Uma rede maior `637 → 64 → 32 → 2` foi comparada, mas não foi adotada. Em validação cruzada estratificada de cinco folds, a rede menor obteve 88,2% de especificidade e 40,4% de recall de tosse; a maior manteve 88,2% de especificidade e reduziu o recall para 36,2%. O gargalo observado é diversidade dos dados, não capacidade da rede.

O modelo foi quantizado integralmente para int8 com dataset representativo. O `.tflite` possui 23.408 bytes e utiliza somente `FullyConnected` e `Softmax`; a arena do TFLite Micro reserva 70 KiB. A versão ONNX exigida acompanha o repositório em `models/cough_detector.onnx`.

## 4. Arquitetura FreeRTOS e concorrência

O diagrama está em [rtos_tasks.svg](rtos_tasks.svg). O firmware contém três tarefas concorrentes:

| Tarefa | Core | Prioridade relativa | Responsabilidade |
|---|---:|---:|---|
| `audio_capture` | 0 | alta | Leitura contínua I2S/DMA e montagem da janela PCM16 |
| `mfcc` | 1 | média | Extração dos 49 × 13 MFCCs |
| `detect` | 1 | baixa | Quantização, inferência TFLM, telemetria e LED |

`free_audio_q` e `ready_audio_q` implementam um pool de dois buffers. Uma fila transfere o ponteiro e, consequentemente, a propriedade exclusiva do bloco entre captura e MFCC. Nenhuma tarefa lê um buffer enquanto outra o escreve, eliminando data race sem mutex no caminho crítico. `feature_q` tem comprimento um e usa `xQueueOverwrite`: sob sobrecarga, a detecção recebe a feature mais recente em vez de acumular atraso. As esperas bloqueantes das filas também evitam polling e inversão de prioridade.

Os buffers grandes de FFT, áudio e features ficam em memória estática. Essa decisão evita estouro das pilhas das tarefas; a proteção de stack por canário do FreeRTOS permanece habilitada.

## 5. Resultados de detecção

Resultados do conjunto independente de teste por arquivo, com limiar 0,50:

| Métrica | Resultado |
|---|---:|
| Amostras de teste | 41 (34 background, 7 tosses) |
| Acurácia | 82,93% |
| Precisão da classe tosse | 50,00% |
| Recall da classe tosse | 85,71% |
| F1 da classe tosse | 63,16% |
| Especificidade | 82,35% |
| Matriz de confusão `[[TN, FP], [FN, TP]]` | `[[28, 6], [1, 6]]` |

A avaliação do artefato TFLite sobre os 268 arquivos produziu 92,9% de acurácia, 89,4% de recall de tosse e 93,7% de especificidade. Como essa execução inclui arquivos usados no treino, ela valida principalmente a conversão/quantização e não substitui o teste independente.

## 6. Latência e uso de recursos

Foram coletadas 100 inferências reais pela UART com `tools/measure_latency.py`; os dados brutos estão em `models/latency.csv`.

| Etapa | Média (ms) | p95 (ms) | Máximo (ms) |
|---|---:|---:|---:|
| Captura da janela | 979,70 | 987,44 | 1.007,91 |
| Extração de MFCC | 400,30 | 400,31 | 400,57 |
| Inferência TFLM | 3,66 | 3,69 | 4,73 |
| Ponta a ponta | 1.384,29 | 1.392,06 | 1.413,86 |

A captura e o MFCC são executados em pipeline em buffers diferentes. A latência ponta a ponta de um evento ainda inclui a espera pela janela de aproximadamente um segundo. O build final ocupa 159.392 bytes de RAM estática (48,6% de 320 KiB) e 272.032 bytes de flash (13,8% da partição de 1.920 KiB).

## 7. Validação e reprodução

O repositório inclui:

- `tools/evaluate_tflite.py`: simula o fluxo MFCC → int8 → inferência e mede desempenho;
- `tools/measure_latency.py`: coleta e resume latências reais da UART;
- `tests/test_features.py`: verifica dimensões e sensibilidade das features;
- `models/metrics.json`: métricas do teste independente;
- `models/latency.csv`: 100 medições no dispositivo;
- `models/cough_detector.onnx` e `models/cough_detector_int8.tflite`;
- `esp32-anomaly-detector/src/model_data.cpp`: TFLite incorporado ao firmware.

Comandos de reprodução estão no `README.md`. Na validação final, os dois testes host passaram, a avaliação TFLite foi concluída e o firmware final compilou e foi gravado com sucesso.

## 8. Limitações e melhorias

Voz e tosse compartilham energia e componentes espectrais; por isso exemplos de conversa foram adicionados como negativos difíceis. Ainda há poucos exemplos de tosse e a validação cruzada apresentou variação relevante entre folds. As prioridades seguintes são coletar tosses de mais pessoas e sessões, reservar grupos completos para teste e incluir mais fala, pigarro, risada e ruídos impulsivos no background. Aumentar a rede antes de ampliar o conjunto não apresentou benefício.

O sistema usa janelas não sobrepostas de um segundo. Uma tosse na fronteira pode ser dividida entre duas janelas; sobreposição de 50% pode melhorar recall ao custo de mais MFCCs. O limiar de 0,50 privilegia recall; ele pode ser elevado se a aplicação tolerar mais falsos negativos em troca de menos alertas indevidos.

## 9. Uso de inteligência artificial

O agente Codex foi utilizado como ferramenta de apoio durante o desenvolvimento da atividade, auxiliando na implementação, revisão, documentação e validação do projeto. As decisões técnicas, os testes e a responsabilidade pelo conteúdo final permaneceram sob supervisão do autor.
