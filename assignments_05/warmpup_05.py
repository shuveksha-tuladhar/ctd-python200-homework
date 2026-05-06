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

# --- Prompt Engineering ---
# Prompt Question 1 — Zero-Shot - Ask the model to classify the sentiment of each review below as positive, negative, or mixed. Give it no examples — just the task description and the reviews. Print each result labeled with the review number.
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]


prompt = (
    "Classify the sentiment of each review as positive, negative, or mixed. "
    "Return only the label for each review."
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a sentiment classification assistant."},
        {"role": "user", "content": prompt + "\n\n" + "\n".join(reviews)}
    ]
)

results = response.choices[0].message.content.strip().split("\n")

for i, label in enumerate(results, start=1):
    print(f"Review {i}: {label}")

# Observation:
# The model outputs sentiment labels based on overall tone.
# Review 1 is positive which is about smooth onboarding and welcoming team,
# Review 2 is negative which is about crashes and no support),
# Review 3 is mixed which is about good price but poor documentation).
# This shows the model can infer sentiment without examples (zero-shot classification).

# Prompt Question 2 — One-Shot - Repeat the same task, but this time add one example before the reviews to show the model the format. Print the results.

prompt = """
Classify the sentiment of each review as positive, negative, or mixed.
Example:Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify the following reviews. 
Return only the sentiment label for each review, one per line:
""" + "\n".join(reviews)
response = client.chat.completions.create(    
                                          model="gpt-4o-mini",    
                                          messages=[        
                                                    {"role": "system", "content": "You are a sentiment classification assistant."},        
                                                    {"role": "user", "content": prompt}    ])
results = response.choices[0].message.content.strip().split("\n")

for i, label in enumerate(results, start=1):    
    print(f"Review {i}: {label}")

# Observation:
# Adding a single example (one-shot) acts as a template.  While the sentiment accuracy remains high, the formatting becomes more predictable. The review sentiment is same as Q1.

# Prompt Question 3 — Few-Shot - Repeat the task again, this time with three examples. At least one example should be positive, one negative, and one mixed. Print the results.
prompt = """
Classify the sentiment of each review as positive, negative, or mixed. 

Example 1:
Review: "The app is incredibly fast and intuitive."
Sentiment: positive

Example 2:
Review: "I hate the new update; it removed my favorite features."
Sentiment: negative

Example 3:
Review: "The hardware feels premium, but the software is buggy."
Sentiment: mixed

Now classify the following reviews. Return only the sentiment label for each review:
""" + "\n".join(reviews)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a sentiment classification assistant."},
        {"role": "user", "content": prompt}
    ]
)

results = response.choices[0].message.content.strip().split("\n")

for i, label in enumerate(results, start=1):
    print(f"Review {i}: {label}")

# Observation:
# Zero-shot relies only on instructions, more variation in formatting and interpretation.
# One-shot improves format consistency by showing a single example.
# Few-shot gives the most stable and reliable output, with clearer labeling and structure.

# Zero-shot is used for simple tasks, fast prototyping, when format doesn't matter much.
# One-shot is used when we need basic structure guidance with minimal tokens.
# Few-shotis used when output consistency, formatting, or accuracy is important in production use cases.

# Prompt Question 4 — Chain of Thought - Ask the model to solve the following problem, but instruct it to show its reasoning step by step before giving a final answer. Label the final answer. Print the full response including the reasoning
prompt = """
Solve the following problem step by step and clearly label the final answer.

A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.

What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a careful math assistant that explains reasoning clearly step by step."},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

# Observation:
# Asking the model to break a problem into steps improves accuracy because it makes the model do each part one at a time. This helps avoid mistakes where the model skips steps or mixes things up. It is especially useful for multi-step math problems like percentages, salary changes, or unit conversions, where small errors in early steps can lead to wrong final answers.

# Prompt Question 5 — Structured Output
import json

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""
Analyze the review and return ONLY valid JSON with the following keys:
- sentiment (positive, negative, or mixed)
- confidence (a float from 0 to 1)
- reason (one sentence explanation)

Review:
{review}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You return only valid JSON. No extra text."},
        {"role": "user", "content": prompt}
    ]
)

raw_output = response.choices[0].message.content

print("Raw response:")
print(raw_output)

try:
    data = json.loads(raw_output)

    print("\nParsed Output:")
    print("Sentiment:", data["sentiment"])
    print("Confidence:", data["confidence"])
    print("Reason:", data["reason"])

except json.JSONDecodeError:
    print("\nFailed to parse JSON. Raw output shown for debugging:")
    print(raw_output)
    
# Prompt Question 6 — Delimiters

# Case 1: Instructions present
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt_1 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response_1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt_1}]
)

print("Case 1 Output:")
print(response_1.choices[0].message.content)

print("\n" + "-" * 40 + "\n")

# Case 2: No instructions
non_instruction_text = "The kitchen smelled like fresh herbs and garlic as the evening light faded through the window."

prompt_2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{non_instruction_text}```
"""

response_2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt_2}]
)

print("Case 2 Output:")
print(response_2.choices[0].message.content)

# Observation:
# Delimiters like triple backticks clearly separate user content from instructions. This prevents the model from from mixing up the content with the task. It also helps prevent the model from being confused or tricked by instructions inside the text.