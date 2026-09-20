#!/usr/bin/env python3
"""Coleta a telemetria CSV do firmware e resume a latência no ESP32."""
import argparse, csv, re, time
from pathlib import Path
import numpy as np
import serial

p=argparse.ArgumentParser()
p.add_argument("--port",required=True); p.add_argument("--baud",type=int,default=115200)
p.add_argument("--count",type=int,default=100); p.add_argument("--timeout",type=float,default=180)
p.add_argument("--output",type=Path,default=Path("models/latency.csv"))
a=p.parse_args()
pattern=re.compile(r"csv,([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+),(\d+),(\d+),([0-9.]+),([01])")
rows=[]; deadline=time.monotonic()+a.timeout
with serial.Serial(a.port,a.baud,timeout=.25) as device:
    device.dtr=False; device.rts=False
    while len(rows)<a.count and time.monotonic()<deadline:
        line=device.readline().decode("utf-8","ignore")
        match=pattern.search(line)
        if match: rows.append([float(x) for x in match.groups()])
if len(rows)<a.count: raise SystemExit(f"Somente {len(rows)}/{a.count} amostras recebidas")
a.output.parent.mkdir(parents=True,exist_ok=True)
header=["capture_ms","mfcc_ms","inference_ms","total_ms","audio_rms","audio_peak","q_clipped","cough_probability","alert"]
with a.output.open("w",newline="") as f: csv.writer(f).writerows([header,*rows])
print(f"amostras={len(rows)} arquivo={a.output}")
for index,name in enumerate(header[:4]):
    values=np.asarray([row[index] for row in rows])
    print(f"{name}: media={values.mean():.2f} p95={np.percentile(values,95):.2f} max={values.max():.2f}")
