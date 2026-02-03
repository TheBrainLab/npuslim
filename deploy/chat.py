from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model = "outputs/qwen3-30b_a3b_int8_dyn-skip_down"

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

