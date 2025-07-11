#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "mnist_weights.h"

#define INPUT_SIZE 100  // 10x10 image

// ReLU activation
float relu(float x) {
    return x > 0 ? x : 0;
}

// Fully connected layer: y = ReLU(Wx)
void linear_relu(const float *input, const float *weights, float *output,
                 int input_dim, int output_dim) {
    for (int i = 0; i < output_dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < input_dim; ++j) {
            sum += input[j] * weights[j * output_dim + i];
        }
        output[i] = relu(sum);
    }
}

// Final layer: no ReLU
void linear(const float *input, const float *weights, float *output,
            int input_dim, int output_dim) {
    for (int i = 0; i < output_dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < input_dim; ++j) {
            sum += input[j] * weights[j * output_dim + i];
        }
        output[i] = sum;
    }
}

// Softmax to get probabilities (optional, for classification)
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

int predict_digit(const float *input_image) {
    float layer1_out[L1_DIM];
    float layer2_out[L2_DIM];
    float layer3_out[OUTPUT_DIM];
    float probs[OUTPUT_DIM];

    linear_relu(input_image, &nn_weights[L1_WEIGHT_OFFSET], layer1_out, INPUT_DIM, L1_DIM);
    linear_relu(layer1_out, &nn_weights[L2_WEIGHT_OFFSET], layer2_out, L1_DIM, L2_DIM);
    linear(layer2_out, &nn_weights[L3_WEIGHT_OFFSET], layer3_out, L2_DIM, OUTPUT_DIM);
    softmax(layer3_out, probs, OUTPUT_DIM);

    // Get the index of the highest probability
    int predicted = 0;
    float max_prob = probs[0];
    for (int i = 1; i < OUTPUT_DIM; ++i) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            predicted = i;
        }
    }
    return predicted;
}

// Simple bilinear resize to 10x10
void resize_to_10x10(unsigned char *src, int w, int h, float *dest) {
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            float gx = (x + 0.5f) * w / 10 - 0.5f;
            float gy = (y + 0.5f) * h / 10 - 0.5f;
            int gxi = (int)gx;
            int gyi = (int)gy;
            float c00 = src[gyi * w + gxi];
            float c10 = (gxi + 1 < w) ? src[gyi * w + gxi + 1] : c00;
            float c01 = (gyi + 1 < h) ? src[(gyi + 1) * w + gxi] : c00;
            float c11 = (gxi + 1 < w && gyi + 1 < h) ? src[(gyi + 1) * w + gxi + 1] : c00;

            float dx = gx - gxi;
            float dy = gy - gyi;

            float val = (1 - dx) * (1 - dy) * c00 +
                        dx * (1 - dy) * c10 +
                        (1 - dx) * dy * c01 +
                        dx * dy * c11;

            dest[y * 10 + x] = val / 255.0f; // Normalize
        }
    }
}

int main() {
    int width, height, channels;
    unsigned char *img = stbi_load("0.jpg", &width, &height, &channels, 1);
    if (!img) {
        printf("Failed to load image");
        printf("stbi_failure_reason: %s\n", stbi_failure_reason());
        return 1;
    }

    float input[100]; // 10x10 flattened
    resize_to_10x10(img, width, height, input);
    stbi_image_free(img);

    int predicted = predict_digit(input);
    printf("Predicted digit: %d\n", predicted);

    return 0;
}
