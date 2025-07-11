# CNN-Based Handwritten Digit Recognition on ESP32

This project demonstrates a lightweight, embedded neural network inference system that recognizes handwritten digits (0–9) on an **ESP32** microcontroller using data sent via **serial (UART)**. It includes:

- A PyTorch-trained digit classifier
- ESP32 firmware that runs inference on 100 float inputs and display on LCD 16x2

## 0. Requirement and Setup

In this experiment, I have used ESP-IDF version 5.4.1 and Python 3.9.21 with following package:

```
numpy==1.26.4
opencv-python==4.7.0
pyserial==3.5
torch==2.0.1
torchvision==0.15.2
```


## 1. System flow diagram

The following image illustrate the overall flow and procedure.

![alt text](assets/flow_diagram.png)

1. First setup the python enviroment and relevant packages.
2. Prepare training parameter and dataset, then train the model based on PyTorch framework as shown in [training code](mnist_train/train.py).
3. Extract the weight in C programming form as header file, then load the header file in [testing code](mnist_test/test.c) to evaluate the result.
4. Deploy the weight file into ESP-IDF project workspace as shown in [this directory](esp32_deploy/mnist_esp32).