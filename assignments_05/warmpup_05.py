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

# API Question 3 - Use n=3 with temperature=1.0 to get three different completions in a single API call. Print all three.

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices, start=1):
    print(f"Completion {i}:")
    print(choice.message.content)
    
# API Question 4 - Set max_tokens=15 and send a prompt that would normally produce a long response (for example, "Explain how neural networks work."). Print the result. 

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

print(response.choices[0].message.content)

# Observation:
# The response is cut off after about 15 tokens, so the explanation is incomplete. This happens because max_tokens limits how much the model is allowed to generate, regardless of whether the answer is finished.
# In real applications, max_tokens is useful to control cost (fewer tokens = cheaper API usage) and limit response length for UI constraints. This would also prevent overly long or verbose outputs

# --- System Messages and Personas ---
# System Question 1 - Use a system message to give the model a personality, then ask it a question. Print the response.
messages_a = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response_a = client.chat.completions.create(model='gpt-4o-mini', 
                                          messages=messages_a)


print("Tutor personality:")
print(response_a.choices[0].message.content)
print("-" * 40)

messages_b = [
    {
        "role": "system",
        "content": "You are a strict senior software engineer. You are concise, direct, and do not add encouragement or fluff."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response_b = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_b
)

print("Strict engineer personality:")
print(response_b.choices[0].message.content)

# Observation:
# Changing the system message completely changes the tone, structure, and style of the answer. The first response is friendly and encouraging with simple explanations, while the second is concise, technical, and removes emotional support. This shows that system messages strongly control the model's "personality" and behavior.


# System Question 2 - The completions API is stateless — it has no memory of previous calls. The way to give a model context is to pass the conversation history yourself as a list of messages. Build the following conversation manually (no loop, no user input — just construct the list) and send it in a single API call:

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(response.choices[0].message.content)

# Observation:
# The model knows Jordan's name because the full conversation history was included in the API request. Even though the model is stateless (it does not remember past calls), it can "appear" to have memory when you resend previous messages as context.

