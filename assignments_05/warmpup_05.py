# Part 1: Warmup Exercises
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# --- The Chat Completions API ---

# API Question 1 - Set up your OpenAI client and make your first chat completion call. Use the model "gpt-4o-mini" and send this prompt: "What is one thing that makes Python a good language for beginners?"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("Response Text:", response.choices[0].message.content)
print("Model Used:", response.model)
print("Total Tokens Used:", response.usage.total_tokens)

