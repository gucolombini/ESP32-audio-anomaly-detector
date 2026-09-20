#!/usr/bin/env python3
"""Código de teste: simula o fluxo, mede latência e valida o TFLite int8."""
import argparse, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from features import load_wav, mfcc

p=argparse.ArgumentParser(); p.add_argument("--model",default="models/cough_detector_int8.tflite"); p.add_argument("--threshold",type=float,default=.5)
a=p.parse_args(); interp=tf.lite.Interpreter(model_path=a.model); interp.allocate_tensors()
inp,out=interp.get_input_details()[0],interp.get_output_details()[0]; ok=total=0; times=[]; truth=[]; predicted=[]
for label,name in enumerate(("background","cough")):
    for wav in (Path("data")/name).glob("*.wav"):
        t=time.perf_counter(); x=mfcc(load_wav(wav)).reshape(-1); scale,zero=inp["quantization"]
        q=np.clip(np.rint(x/scale+zero),-128,127).astype(np.int8)[None]
        interp.set_tensor(inp["index"],q); interp.invoke(); raw=interp.get_tensor(out["index"])[0]
        probability=(float(raw[1])-out["quantization"][1])*out["quantization"][0]; pred=int(probability>=a.threshold)
        times.append((time.perf_counter()-t)*1000); ok+=pred==label; total+=1; truth.append(label); predicted.append(pred)
print(f"amostras={total} acuracia={ok/max(total,1):.3f} latencia_host_ms_media={np.mean(times):.2f} p95={np.percentile(times,95):.2f}")
from sklearn.metrics import confusion_matrix, classification_report
print("matriz_confusao=",confusion_matrix(truth,predicted).tolist())
print(classification_report(truth,predicted,target_names=["background","cough"],digits=3))
