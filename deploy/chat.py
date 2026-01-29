from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:52065/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model = "outputs/qwen3-0_6b_int4_gptq"

chat_response = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello? Who are you?"},
           
        ],
    }],
)
print("Chat completion output:", chat_response.choices[0].message.content)

