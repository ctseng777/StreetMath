#!/bin/bash

# 定义要跑的文件列表 (文件名)
EXPERIMENTS=(
"linear_probe_falcon_34b_near_2.py"
"linear_probe_falcon_34b_near_3.py"
"linear_probe_falcon_34b_near_6.py"
"linear_probe_falcon_34b_near_9.py"
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
echo "All Falcon-34B experiments completed!"
