from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# 1. Start with the system message that defines the bot's personality
messages = [
    {"role": "system", 
    "content": "You are a friendly, competent sales assistant who helps customers find shoes that fit their needs. Ask questions to learn their shoe size, style preferences, and use case before recommending options."}
]

print("SalesBot: Hi there! I'm here to help you find the perfect pair of shoes.")
print("Please let me know what kind of shoes you're looking for -- running, casual, hiking, etc.")
print("(Type 'quit' anytime to end the chat.)")

# 2. Begin the conversation loop
exchange_count = 0
max_exchanges = 10

while exchange_count < max_exchanges:
    user_input = input("You: ")

    if user_input.lower() in {"quit", "exit"}:
        print("SalesBot: Thanks for visiting! Have a great day.")
        break

    # Add the user's message to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Send the full conversation so far to the API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_completion_tokens=300
    )

    # Extract and print the assistant's reply
    reply = response.choices[0].message.content
    print(f"SalesBot: {reply}")
    print("(Type 'quit' anytime to end the chat.)\n")

    # Add the assistant's reply to the conversation history
    messages.append({"role": "assistant", "content": reply})

    exchange_count += 1

if exchange_count >= max_exchanges:
    print("SalesBot: It was great chatting with you! Let's continue another time.")