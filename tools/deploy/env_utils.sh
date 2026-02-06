#!/bin/bash

# Detect hardware platform silently
detect_device() {
    if command -v npu-smi &> /dev/null || [ -c /dev/davinci0 ]; then
        echo "npu"
    elif command -v nvidia-smi &> /dev/null || [ -c /dev/nvidia0 ]; then
        echo "gpu"
    else
        echo "cpu"
    fi
}

# Export environment variables based on device type
setup_env() {
    local device_type=$1
    local devices=$2

    if [[ "$device_type" == "npu" ]]; then
        export ASCEND_RT_VISIBLE_DEVICES=$devices
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    elif [[ "$device_type" == "gpu" ]]; then
        export CUDA_VISIBLE_DEVICES=$devices
    fi
}