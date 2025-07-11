| Supported Targets | ESP32 | ESP32-C2 | ESP32-C3 | ESP32-C6 | ESP32-H2 | ESP32-P4 | ESP32-S2 | ESP32-S3 |
| ----------------- | ----- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |

# _ESP32 CNN_

Copy weight file (.h) into main folder:

```
mnist_esp32/
|- main/
  |- CMakeLists.txt
  |- main.c
  |- inference.c
  |- inference.h
  |- mnist_weights.h 
...

```

According to ESP32 datasheet, I2C and UART can be map to any pins. Hence in this case I am using:

1. I2C Interface for PCF8574
- GPIO 13 (SDA)
- GPIO 12 (SCL)

2. UART Interface for input
- GPIO 27 (RXD)
- GPIO 26 (TXD)

To flash the overall code into ESP32, just type:

```
idf.py -p COMx flash monitor
```