#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>

int predict_digit(const float *input_image, float *out_confidence);

#endif // INFERENCE_H