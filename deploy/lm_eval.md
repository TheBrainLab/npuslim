```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
lm_eval --model vllm \
    --model_args pretrained="outputs/qwen3-30b_a3b_int8_dyn",tensor_parallel_size=2,gpu_memory_utilization=0.8,max_model_len=4096,trust_remote_code=True,quantization=ascend \
    --tasks mmlu \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path outputs/eval/qwen3-30b_a3b_int8_dyn/mmlu_results.json


CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained="outputs/new/qwen3_0_6b-int4_gptq",tensor_parallel_size=1,gpu_memory_utilization=0.6,max_model_len=4096,trust_remote_code=True \
    --tasks mmlu \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path outputs/eval/qwen3_8b-int4_gptq/mmlu_results.json

lm_eval --model hf \
    --model_args pretrained=outputs/qwen3-4b_int4_gptq \
    --tasks arc_easy,arc_challenge,boolq,headqa_en,openbookqa,hellaswag,piqa,winogrande \
    --device cuda:1 \
    --batch_size auto


CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained=/data16t/MODELSCOPE/models/Qwen/Qwen3-30B-A3B-Instruct-2507,max_model_len=4096,tensor_parallel_size=2,gpu_memory_utilization=0.8,trust_remote_code=True \
    --tasks arc_easy,arc_challenge,boolq,headqa_en,openbookqa,hellaswag,piqa,winogrande \
    --batch_size auto
```