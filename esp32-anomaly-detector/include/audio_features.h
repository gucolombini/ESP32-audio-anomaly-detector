#pragma once
#include <stdint.h>
#include "app_config.h"

// Implementação de referência compartilhada conceitualmente com tools/features.py.
void extract_mfcc(const int16_t *audio, float *output);
