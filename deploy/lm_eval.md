```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
lm_eval --model vllm \
    --model_args pretrained="/home/lichangcai/projects/llm/npuslim/outputs/qwen3-32b_int8_dyn",tensor_parallel_size=2,gpu_memory_utilization=0.8,max_model_len=4096,trust_remote_code=True,quantization=ascend \
    --tasks mmlu \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path /home/lichangcai/projects/llm/npuslim/outputs/eval/mmlu_results.json
```