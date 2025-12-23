## 安装
在之前安装的vllm环境上直接安装：
```
conda activate vllm
pip install -e . -v
```
确保`torch`是`2.7.1`以及`vllm`是`0.11`版本（版本太老不支持量化部署）。

## 量化
目前仅实现了`qwen3`的`INT8 dynamic`量化。

检查配置文件`config/qwen3-0_6b_int8_dyn.yaml`中的`model_path`, 修改为本地自己的模型路径。
执行：
```bash
python tools/run_quant.py -c config/qwen3-0_6b_int8_dyn.yaml
```

## vLLM部署
- 完成量化后应该可以直接部署了，因为底层的算子之类的东西`vLLM-ascend`已经实现好了，可以直接通过插件的方式调用它实现的`W8A8 Dynamic`方法。（前面的`pip install -e . -v`已经注册插件了，当然后续还要扩充。）
    ```bash
    bash deploy/run_vllm.sh --model-path outputs/qwen3-0_6b_int8_dyn -d 4,5 -t 2 -q
    ```
    其他同学的vllm服务可能会占用端口，换一个即可。每个参数的含义可以查看启动脚本的内容，比如`-q`是指采用量化。

- `deploy/evalscope.md`和`deploy/lm_eval.md`中提供了压测和精度评估的命令，改一下即可。

- 运行`python deploy/chat.py`可以尝试与服务进行对话测试。