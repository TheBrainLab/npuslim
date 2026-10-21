from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="",
        model="front_l0",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="HOST_IP_PLACEHOLDER",
        host_port=8000,
        url="",
        max_out_len=65536,
        batch_size=8,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.0,
            ignore_eos=False,
            chat_template_kwargs=dict(reasoning_effort="high"),
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
