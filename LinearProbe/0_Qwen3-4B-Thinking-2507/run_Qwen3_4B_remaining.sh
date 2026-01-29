#!/bin/bash

# 定义要跑的文件列表 (文件名)
EXPERIMENTS=(
"linear_probe_qwen_thinking_near_5.py"
"linear_probe_qwen_thinking_near_10.py"
)

# 循环执行
for script in "${EXPERIMENTS[@]}"; do
    echo "=================================================="
    echo "Starting: $script"
    echo "Time: $(date)"
    echo "=================================================="

    # 执行 Python 脚本
    # 2>&1 | tee ... : 既在屏幕上显示，又保存到单独的日志文件里
    log_file="${script%.py}.log"
    python "$script" 2>&1 | tee "$log_file"

    echo ""
    echo "Finished: $script"
    echo "Logs saved to: $log_file"
    echo "Cleaning up..."
    sleep 3  # 给系统几秒钟喘息，确保显存彻底释放
done

echo "=================================================="
echo "All Qwen3-4B experiments completed!"
