from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://0.0.0.0:8082/v1",
)

models = client.models.list()
model = models.data[0].id

print(f"Using model: {model}")

try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
        stream=False
    )
    print("Response:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
