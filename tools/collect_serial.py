#!/usr/bin/env python3
import argparse, re, struct, wave
from pathlib import Path
import serial

p=argparse.ArgumentParser(description="Salva janelas PCM16 validadas enviadas pelo ESP32")
p.add_argument("label",choices=["cough","background"]); p.add_argument("--port",required=True)
p.add_argument("--count",type=int,default=20); p.add_argument("--baud",type=int,default=115200)
a=p.parse_args(); out=Path("data")/a.label; out.mkdir(parents=True,exist_ok=True)
# Continua após o maior índice existente. Arquivos apagados não fazem a
# numeração voltar e uma nova coleta nunca sobrescreve dados anteriores.
indices=[]
for path in out.glob(f"{a.label}_*.wav"):
    match=re.fullmatch(rf"{re.escape(a.label)}_(\d+)",path.stem)
    if match: indices.append(int(match.group(1)))
next_index=max(indices,default=-1)+1
s=serial.Serial(a.port,a.baud,timeout=5); buf=bytearray(); saved=0; last_sequence=None
while saved<a.count:
    buf.extend(s.read(4096)); pos=buf.find(b"WAV1")
    if pos<0: buf[:]=buf[-3:]; continue
    if len(buf)<pos+16: continue
    size,sequence,checksum=struct.unpack_from("<III",buf,pos+4)
    if size!=32000: del buf[:pos+4]; continue
    if len(buf)<pos+16+size: continue
    pcm=bytes(buf[pos+16:pos+16+size]); del buf[:pos+16+size]
    if sum(pcm)&0xffffffff != checksum:
        print(f"quadro {sequence} corrompido; descartado")
        continue
    if last_sequence is not None and sequence != last_sequence+1:
        print(f"aviso: sequência saltou de {last_sequence} para {sequence}")
    last_sequence=sequence
    path=out/f"{a.label}_{next_index+saved:04d}.wav"
    with wave.open(str(path),"wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)
    saved+=1; print(f"[{saved}/{a.count}] {path}")
