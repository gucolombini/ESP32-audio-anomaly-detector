#include "audio_features.h"
#include <math.h>
#include <string.h>

static float hz_to_mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

// A extração é executada por uma única tarefa. Manter estes buffers em RAM
// estática evita consumir mais de 5 KiB da pilha da tarefa a cada chamada.
static float fft_real[FFT_SIZE];
static float fft_imag[FFT_SIZE];
static float power_spectrum[FFT_SIZE / 2 + 1];
static float mel_energies[MEL_BINS];

static void fft(float *re, float *im) {
    for (int i = 1, j = 0; i < FFT_SIZE; ++i) {
        int bit = FFT_SIZE >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t=re[i]; re[i]=re[j]; re[j]=t; t=im[i]; im[i]=im[j]; im[j]=t; }
    }
    for (int len = 2; len <= FFT_SIZE; len <<= 1) {
        const float angle = -2.0f * (float)M_PI / len;
        for (int i = 0; i < FFT_SIZE; i += len) {
            for (int j = 0; j < len / 2; ++j) {
                float c=cosf(angle*j), s=sinf(angle*j);
                float vr=re[i+j+len/2]*c-im[i+j+len/2]*s;
                float vi=re[i+j+len/2]*s+im[i+j+len/2]*c;
                float ur=re[i+j], ui=im[i+j];
                re[i+j]=ur+vr; im[i+j]=ui+vi;
                re[i+j+len/2]=ur-vr; im[i+j+len/2]=ui-vi;
            }
        }
    }
}

void extract_mfcc(const int16_t *audio, float *output) {
    float *re=fft_real, *im=fft_imag, *power=power_spectrum;
    int bins[MEL_BINS+2];
    const float mel_max=hz_to_mel(SAMPLE_RATE_HZ/2.0f);
    for (int i=0; i<MEL_BINS+2; ++i) {
        float hz=mel_to_hz(mel_max*i/(MEL_BINS+1));
        bins[i]=(int)floorf((FFT_SIZE+1)*hz/SAMPLE_RATE_HZ);
        if (bins[i] > FFT_SIZE/2) bins[i]=FFT_SIZE/2;
    }
    for (int f=0; f<MFCC_FRAMES; ++f) {
        memset(re,0,sizeof(fft_real));
        memset(im,0,sizeof(fft_imag));
        for (int i=0; i<FRAME_LENGTH; ++i) {
            float w=0.5f-0.5f*cosf(2.0f*(float)M_PI*i/(FRAME_LENGTH-1));
            re[i]=(audio[f*FRAME_STEP+i]/32768.0f)*w;
        }
        fft(re,im);
        for (int k=0;k<=FFT_SIZE/2;++k) power[k]=(re[k]*re[k]+im[k]*im[k])/FFT_SIZE;
        float *logmel=mel_energies;
        for (int m=1;m<=MEL_BINS;++m) {
            float e=0.0f;
            int den1=bins[m]-bins[m-1], den2=bins[m+1]-bins[m];
            for(int k=bins[m-1];k<bins[m];++k) if(den1) e+=power[k]*(k-bins[m-1])/den1;
            for(int k=bins[m];k<bins[m+1];++k) if(den2) e+=power[k]*(bins[m+1]-k)/den2;
            logmel[m-1]=logf(e+1e-6f);
        }
        for(int c=0;c<MFCC_COUNT;++c) {
            float v=0.0f;
            for(int m=0;m<MEL_BINS;++m) v+=logmel[m]*cosf((float)M_PI*c*(m+0.5f)/MEL_BINS);
            output[f*MFCC_COUNT+c]=v;
        }
    }
}
