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

# API Question 2 - Run the same prompt three times with three different temperature settings: 0, 0.7, and 1.5. Print each response, labeled with its temperature.

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temp,
        messages=[{"role": "user", "content": prompt}]
    )
    
    print(f"Temperature {temp}:")
    print(response.choices[0].message.content)
    print("-" * 40)
    
# Observation:
# At temperature 0, the model returned a single, short, and consistent name ("DataForge Solutions").
# At temperature 0.7, it generated a list of multiple creative suggestions with a conversational tone.
# At temperature 1.5, it also produced a list, but with slightly more varied and less predictable names.
# This shows that higher temperatures increase creativity and variation, while lower temperatures are more focused and consistent. For consistent and reproducible output, I would use temperature = 0.