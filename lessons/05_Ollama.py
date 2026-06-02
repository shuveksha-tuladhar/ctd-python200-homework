from dotenv import load_dotenv
from ollama import Client

load_dotenv()

client = Client()

messages = [
    {
        "role": "system",
        "content": "You are a witty assistant who gives short, clever answers."
    },
    {
        "role": "user",
        "content": "Explain why Python is popular."
    }
]

response = client.chat(
    model="qwen3:0.6b",  # or "llama3"
    messages=messages
)

print(response["message"]["content"])