#include <stdio.h>
#include <math.h>
#include "mnist_weights.h"

float relu(float x) {
    return x > 0 ? x : 0;
}

void linear_relu(const float *input, const float *weights, float *output, int input_dim, int output_dim) {
    for (int i = 0; i < output_dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < input_dim; ++j) {
            sum += input[j] * weights[j * output_dim + i];
        }
        output[i] = relu(sum);
    }
}

void linear(const float *input, const float *weights, float *output, int input_dim, int output_dim) {
    for (int i = 0; i < output_dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < input_dim; ++j) {
            sum += input[j] * weights[j * output_dim + i];
        }
        output[i] = sum;
    }
}

void softmax(const float *input, float *output, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; ++i)
        if (input[i] > max_val) max_val = input[i];

    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < length; ++i) {
        output[i] /= sum;
    }
}

int predict_digit(const float *input_image, float *out_confidence) {
    float layer1_out[L1_DIM];
    float layer2_out[L2_DIM];
    float layer3_out[OUTPUT_DIM];
    float probs[OUTPUT_DIM];

    linear_relu(input_image, &nn_weights[L1_WEIGHT_OFFSET], layer1_out, INPUT_DIM, L1_DIM);
    linear_relu(layer1_out, &nn_weights[L2_WEIGHT_OFFSET], layer2_out, L1_DIM, L2_DIM);
    linear(layer2_out, &nn_weights[L3_WEIGHT_OFFSET], layer3_out, L2_DIM, OUTPUT_DIM);
    softmax(layer3_out, probs, OUTPUT_DIM);

    int predicted = 0;
    float max_prob = probs[0];
    for (int i = 1; i < OUTPUT_DIM; ++i) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            predicted = i;
        }
    }

    if (out_confidence != NULL) {
        *out_confidence = max_prob;
    }

    return predicted;
}
