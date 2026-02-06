```bash
evalscope perf \
  --parallel 1 10 50 100 200 \
  --number 10 20 100 200 400 \
  --model outputs/qwen3-4b_int8_dyn \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 1024 \
  --min-tokens 1024 \
  --prefix-length 0 \
  --min-prompt-length 1024 \
  --max-prompt-length 1024 \
  --tokenizer-path outputs/qwen3-4b_int8_dyn \
  --extra-args '{"ignore_eos": true}'

evalscope eval \
 --model /home/lichangcai/projects/llm/npuslim/outputs/qwen3-32b_int8_dyn \
 --api-url http://127.0.0.1:8080/v1 \
 --api-key EMPTY \
 --eval-type openai_api \
 --datasets mmlu \
#  --limit 10
```