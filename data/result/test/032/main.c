/* main_template.c */
#include "iec_std_lib.h"
#include "iec_types_all.h"
#include "POUS.h"
#include "Config0.h"

/* 这些函数由 matiec 生成 */
extern void config_run__(unsigned long tick);
extern void config_init__(void);

/* 定义全局时间变量 (MatIEC 需要) */
TIME __CURRENT_TIME;
BOOL __DEBUG = 0; // 如果不需要调试模式设为0

/* 简单的时钟模拟 */
void wait_next_cycle() {
    __CURRENT_TIME.tv_nsec += 20 * 1000 * 1000; // 增加 20ms
    if (__CURRENT_TIME.tv_nsec >= 1000000000) {
        __CURRENT_TIME.tv_nsec -= 1000000000;
        __CURRENT_TIME.tv_sec += 1;
    }
}

int main(int argc, char **argv) {
    printf("--- Starting PLC Simulation ---\n");
    
    config_init__();
    
    // 运行 100 个周期
    for(int i=0; i<100; i++) {
        config_run__(0);
        wait_next_cycle();
        // 注意：这里无法直接看到 LLM 块内部的变量，
        // 除非你在 ST 中将变量映射到 %Q (Output) 地址，
        // 或者解析 generated_plc.st 对应的结构体。
    }
    
    printf("--- Simulation Finished ---\n");
    return 0;
}