from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="",
        model="front_w8a8",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="HOST_IP_PLACEHOLDER",
        host_port=8000,
        url="",
        max_out_len=32768,
        batch_size=5,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.6,
            ignore_eos=False,
            chat_template_kwargs=dict(reasoning_effort="high"),
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
