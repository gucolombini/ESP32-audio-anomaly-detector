#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys
import numpy as np
import tensorflow as tf
from features import load_wav, mfcc

saved="models/saved_model"; out=Path("models"); out.mkdir(exist_ok=True)
samples=[mfcc(load_wav(p)).reshape(1,-1) for p in Path("data").glob("*/*.wav")]
def representative():
    for x in samples[:100]: yield [x.astype(np.float32)]
converter=tf.lite.TFLiteConverter.from_saved_model(saved)
converter.optimizations=[tf.lite.Optimize.DEFAULT]; converter.representative_dataset=representative
converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type=tf.int8; converter.inference_output_type=tf.int8
tflite=converter.convert(); (out/"cough_detector_int8.tflite").write_bytes(tflite)
subprocess.run([sys.executable,"-m","tf2onnx.convert","--saved-model",saved,
                "--output",str(out/"cough_detector.onnx"),"--opset","13"],check=True)
lines=[]
for i in range(0,len(tflite),12): lines.append("  "+", ".join(f"0x{x:02x}" for x in tflite[i:i+12])+",")
header='#include "model_data.h"\nconst unsigned char g_model_data[] = {\n'+"\n".join(lines)+'\n};\nconst size_t g_model_data_len = sizeof(g_model_data);\n'
Path("esp32-anomaly-detector/src/model_data.cpp").write_text(header,encoding="utf-8")
print(f"TFLite: {len(tflite)} bytes; ONNX e header gerados")
