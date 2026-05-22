# --- AI_agents ---
# --- Tool Definitions and the ReAct Loop ---
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import json
import os

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')

# Q1
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

tools = [
    {
        'type': 'function',
        'function': {
            "name": "celsius_to_fahrenheit",
            "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "celsius": {
                        "type": "number",
                        "description": "Temperature in degrees Celsius"
                    }
                },
                "required": ["celsius"]
            }
        },
    }
]

print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# --- Q2 ---

# Prediction:
# No, the agent will not use a tool. Because the agent only knows the get_current_time tool. It does not know the celsius_to_fahrenheit tool.
# So the model will answer by itself instead of calling a tool.
# Number of API calls: Only 1 API call will happen. Because no tool is used, so there is no second API call.

def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current local time as a formatted string.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

# ReAct-style agent loop
def run_agent(user_input: str) -> str:

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input},
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
    )

    message = response.choices[0].message

    if message.tool_calls:

        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name

        if tool_name == "get_current_time":

            result = get_current_time()

            messages.append(message)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

            second_response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                tools=tools,
            )

            return second_response.choices[0].message.content

    return message.content

result = run_agent("Convert 100 degrees Celsius to Fahrenheit")

print(result)

# Result analysis:
# The prediction was correct. The agent did not call a tool. This happened because the agent only had access to the get_current_time tool. It did not have access to the celsius_to_fahrenheit tool.
# The model answered the question by itself. It returned a response similar to: "100 degrees Celsius is 212 degrees Fahrenheit."
# Only one API call was made because no tool call was needed.

# --- Q3 --- 

def run_agent(user_input: str) -> str:

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input},
    ]

    # First API call
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
    )

    message = response.choices[0].message

    if message.tool_calls:

        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name

        arguments = json.loads(tool_call.function.arguments)

        if tool_name == "get_current_time":
            result = get_current_time()

        elif tool_name == "celsius_to_fahrenheit":
            result = celsius_to_fahrenheit(arguments["celsius"])

        else:
            result = "Unknown tool."

        messages.append(message)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

        # Second API call
        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools,
        )

        return second_response.choices[0].message.content

    # No tool call
    return message.content


response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)

# A tool was called here. The model used the celsius_to_fahrenheit tool because the question asked for a temperature conversion.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)

# No tool was needed here. The model already knows that water boils at about 100 degrees Celsius (212 degrees Fahrenheit), so it answered directly without calling a tool.