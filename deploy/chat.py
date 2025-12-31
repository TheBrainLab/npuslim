from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model = "outputs/qwen3-0_6b_sparsegpt_2of4"

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

