"""MFCC equivalente ao firmware (sem dependência de librosa)."""
from pathlib import Path
import wave
import numpy as np

SR, N, FRAME, STEP, FFT, MELS, COEFFS = 16000, 16000, 480, 320, 512, 26, 13

def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        if width != 2: raise ValueError(f"{path}: somente PCM16 é aceito")
        x=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").astype(np.float32)
    if channels>1: x=x.reshape(-1,channels).mean(axis=1)
    if rate!=SR:
        from scipy.signal import resample_poly
        x=resample_poly(x,SR,rate)
    if len(x)<N: x=np.pad(x,(0,N-len(x)))
    return x[:N]/32768.0

def mfcc(x: np.ndarray) -> np.ndarray:
    mel=lambda h:2595*np.log10(1+h/700)
    hz=lambda m:700*(10**(m/2595)-1)
    bins=np.floor((FFT+1)*hz(np.linspace(0,mel(SR/2),MELS+2))/SR).astype(int).clip(0,FFT//2)
    bank=np.zeros((MELS,FFT//2+1),np.float32)
    for m in range(1,MELS+1):
        for k in range(bins[m-1],bins[m]):
            if bins[m]>bins[m-1]: bank[m-1,k]=(k-bins[m-1])/(bins[m]-bins[m-1])
        for k in range(bins[m],bins[m+1]):
            if bins[m+1]>bins[m]: bank[m-1,k]=(bins[m+1]-k)/(bins[m+1]-bins[m])
    dct=np.cos(np.pi*np.arange(COEFFS)[:,None]*(np.arange(MELS)+.5)/MELS).astype(np.float32)
    out=[]
    for start in range(0,N-FRAME+1,STEP):
        frame=x[start:start+FRAME]*np.hanning(FRAME)
        power=(np.abs(np.fft.rfft(frame,n=FFT))**2/FFT).astype(np.float32)
        out.append(dct@np.log(bank@power+1e-6))
    return np.asarray(out,dtype=np.float32) # (49, 13)
