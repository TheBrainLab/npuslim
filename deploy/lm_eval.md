```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
lm_eval --model vllm \
    --model_args pretrained="outputs/qwen3-30b_a3b_int8_dyn",tensor_parallel_size=2,gpu_memory_utilization=0.8,max_model_len=4096,trust_remote_code=True,quantization=ascend \
    --tasks mmlu \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path outputs/eval/qwen3-30b_a3b_int8_dyn/mmlu_results.json

lm_eval --model vllm \
    --model_args pretrained="outputs/qwen3-30b_a3b_int4_gptq",tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096,trust_remote_code=True \
    --tasks mmlu \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path outputs/eval/qwen3-30b_a3b_int4_gptq/mmlu_results.json

lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path ./mmlu_results_raw.json
```