from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="",
        model="glm52-bf16-eval",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="10.1.0.106",
        host_port=9081,
        url="",
        max_out_len=32768,
        batch_size=8,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.6,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
