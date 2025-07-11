#include <stdio.h>
#include <string.h>
#include "driver/i2c.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"

#include "driver/gpio.h"
#include "driver/uart.h"

#include "inference.h"

#define I2C_MASTER_NUM I2C_NUM_0
#define I2C_MASTER_SCL_IO 12
#define I2C_MASTER_SDA_IO 13
#define I2C_MASTER_FREQ_HZ 100000

#define FLOAT_ARRAY_SIZE 100
#define FLOAT_BYTES (sizeof(float) * FLOAT_ARRAY_SIZE)

#define LCD_ADDR 0x27 // or 0x3F depending on your module
#define LCD_BACKLIGHT 0x08
#define LCD_ENABLE 0x04
#define LCD_RW 0x00
#define LCD_RS 0x01

void i2c_master_init() {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };
    i2c_param_config(I2C_MASTER_NUM, &conf);
    i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);
}

void lcd_write_nibble(uint8_t nibble, uint8_t mode) {
    uint8_t data = (nibble & 0xF0) | LCD_BACKLIGHT | mode;
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (LCD_ADDR << 1) | I2C_MASTER_WRITE, true);
    
    // pulse enable
    i2c_master_write_byte(cmd, data, true);
    i2c_master_write_byte(cmd, data | LCD_ENABLE, true);
    i2c_master_write_byte(cmd, data & ~LCD_ENABLE, true);

    i2c_master_stop(cmd);
    i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    vTaskDelay(pdMS_TO_TICKS(2));
}

void lcd_send_byte(uint8_t byte, uint8_t mode) {
    lcd_write_nibble(byte & 0xF0, mode);
    lcd_write_nibble((byte << 4) & 0xF0, mode);
}

void lcd_send_cmd(uint8_t cmd) {
    lcd_send_byte(cmd, 0);
}

void lcd_send_data(uint8_t data) {
    lcd_send_byte(data, LCD_RS);
}

void lcd_init() {
    vTaskDelay(pdMS_TO_TICKS(50));
    lcd_write_nibble(0x30, 0);
    vTaskDelay(pdMS_TO_TICKS(5));
    lcd_write_nibble(0x30, 0);
    vTaskDelay(pdMS_TO_TICKS(1));
    lcd_write_nibble(0x30, 0);
    vTaskDelay(pdMS_TO_TICKS(10));
    lcd_write_nibble(0x20, 0); // 4-bit mode

    lcd_send_cmd(0x28); // 4-bit, 2-line, 5x8 dots
    lcd_send_cmd(0x0C); // display ON, cursor OFF
    lcd_send_cmd(0x06); // entry mode
    lcd_send_cmd(0x01); // clear display
    vTaskDelay(pdMS_TO_TICKS(2));
}

void lcd_set_cursor(uint8_t col, uint8_t row) {
    const uint8_t offsets[] = {0x00, 0x40};
    lcd_send_cmd(0x80 | (col + offsets[row]));
}

void lcd_print(const char *str) {
    while (*str) {
        lcd_send_data((uint8_t)(*str++));
    }
}

void lcd_clear() {
    lcd_send_cmd(0x01);  // Clear display
    vTaskDelay(pdMS_TO_TICKS(2));  // This command needs >1.5ms to complete
}

void uart_init() {
    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    uart_driver_install(UART_NUM_1, 256, 0, 0, NULL, 0);
    uart_param_config(UART_NUM_1, &uart_config);
    uart_set_pin(UART_NUM_1, GPIO_NUM_27, GPIO_NUM_26, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
}

void app_main() {
    i2c_master_init();
    lcd_init();
    uart_init();
    lcd_clear();
    printf("Waiting for 100 floats (400 bytes) from UART1...\n");
    lcd_set_cursor(0, 0);
    lcd_print("Waiting for data...");

    float input[FLOAT_ARRAY_SIZE];
    uint8_t raw[FLOAT_BYTES];
    int received = 0;

    // Wait (blocking) until exactly 400 bytes (100 floats) are received
    while (received < FLOAT_BYTES) {
        int len = uart_read_bytes(UART_NUM_1, raw + received, FLOAT_BYTES - received, portMAX_DELAY);
        if (len > 0) {
            received += len;
        }
    }

    memcpy(input, raw, FLOAT_BYTES);  // binary float array

    // Start counting time for prediction

    int64_t start_time = esp_timer_get_time();


    float confidence;
    int predicted = predict_digit(input, &confidence);

    int64_t end_time = esp_timer_get_time();
    float elapsed_ms = (end_time - start_time) / 1000.0f;

    int conf_int = (int)(confidence * 100);
    int time_int = (int)(elapsed_ms * 1000);  // ms to microseconds

    // End complete prediction time

    printf("Prediction complete.\nPredicted digit: %d\n", predicted);
    printf("Confidence: %.2f\n", confidence);
    printf("Time taken for prediction: %.2f ms\n", elapsed_ms);


    lcd_clear();
    lcd_set_cursor(0, 0);
    char buffer[16];
    char parameter[16];
    snprintf(buffer, sizeof(buffer), "Predicted: %d", predicted);
    snprintf(parameter, sizeof(parameter), "C:%d%%, T:%d us", conf_int, time_int);
    lcd_print(buffer);
    lcd_set_cursor(0, 1);
    lcd_print(parameter);
}
