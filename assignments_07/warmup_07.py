# --- AI_agents ---
# --- Tool Definitions and the ReAct Loop ---
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import json
import os
import pandas as pd
from smolagents import ToolCallingAgent, CodeAgent, tool
from smolagents.models import OpenAIServerModel

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

# --- Multi-Tool Agent ---
# --- Q4 ---
class CsvManager:

    def __init__(self):
        self.df = None

    def load_csv(self, file_path: str):
        """
        Load a CSV file into a pandas DataFrame.
        """
        try:
            self.df = pd.read_csv(file_path)

            return {
                "status": "success",
                "columns": list(self.df.columns),
                "rows": len(self.df)
            }

        except Exception as e:
            return {"error": str(e)}

    def preview_data(self, n: int = 5):
        """
        Preview the first n rows of the DataFrame.
        """
        if self.df is None:
            return {"error": "No CSV loaded."}

        return self.df.head(n).to_dict(orient="records")

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """

        if self.df is None:
            return {"error": "No CSV loaded."}

        if col1 not in self.df.columns:
            return {"error": f"Column '{col1}' not found."}

        if col2 not in self.df.columns:
            return {"error": f"Column '{col2}' not found."}

        try:
            r, p = pearsonr(self.df[col1], self.df[col2])

            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 4)
            }

        except Exception as e:
            return {"error": str(e)}


csv_manager = CsvManager()

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file into a pandas DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    }
                },
                "required": ["file_path"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "preview_data",
            "description": "Preview the first n rows of the DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of rows to preview"
                    }
                }
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation between two columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "First column name"
                    },
                    "col2": {
                        "type": "string",
                        "description": "Second column name"
                    }
                },
                "required": ["col1", "col2"]
            }
        }
    }
]

# Tool Dispatcher
node_tools = {
    "load_csv": csv_manager.load_csv,
    "preview_data": csv_manager.preview_data,
    "compute_correlation": csv_manager.compute_correlation
}

# System Prompt
SYSTEM_PROMPT = """
You are a CSV data assistant.

You can:
- load CSV files
- preview datasets
- compute correlations between columns

Use tools whenever needed.
"""

# ReAct Agent Loop

def run_agent_cycle(messages, user_input, max_rounds=5):

    messages.append({
        "role": "user",
        "content": user_input
    })

    for i in range(max_rounds):

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema
        )

        message = response.choices[0].message

        messages.append(message)

        # If no tool call, return final answer
        if not message.tool_calls:
            return message.content

        # Handle tool calls
        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            tool_function = node_tools[tool_name]

            tool_result = tool_function(**tool_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result)
            })

    return "Tool round limit reached."

# --- Q5 ---

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)

print(result)

# --- Q6 ---

# Role explanations:
# system: Gives the agent its instructions and behavior rules.
# user: Contains the user's request or question.
# assistant: Contains the model's reasoning, responses, and tool-call decisions.
# tool: Contains the output returned by a tool function.
# The assistant reads these results before continuing.

print(json.dumps(messages, indent=2, default=str))

# --- smolagents ---
# --- Q7 ---

@tool
def compute_correlation(col1: str, col2: str):
    """
    Compute the Pearson correlation between two columns
    in the loaded DataFrame.
    """

    return csv_manager.compute_correlation(col1, col2)


print(compute_correlation.description)

# Comparison:
# In Q4, I had to manually write the JSON schema with:
# - name
# - description
# - parameter types
# - required fields

# smolagents generates this information automatically from the function name, type hints, and docstring.

# To generate a good tool description, smolagents needs:
# - a clear function name
# - type annotations
# - a good docstring that explains what the tool does

def load_csv(file_path: str):
    """
    Load a CSV file into a pandas DataFrame.
    """

    return csv_manager.load_csv(file_path)

@tool
def preview_data(n: int = 5):
    """
    Preview the first rows of the loaded dataset.
    """

    return csv_manager.preview_data(n)

# Shared Model
model = OpenAIServerModel(
    model_id="gpt-4.1-mini"
)

# Shared TOOLS list
TOOLS = [
    load_csv,
    preview_data,
    compute_correlation
]

# --- Q8 ---

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model
)

prompt = """
Load bike_commute.csv.
Plot avg_heart_rate vs duration_min
as a scatter plot with green dots.
"""

response_tool = tool_agent.run(prompt)

response_code = code_agent.run(
    prompt,
    additional_args={"csv_manager": csv_manager}
)

print("ToolCallingAgent Response:")
print(response_tool)

print("\nCodeAgent Response:")
print(response_code)


# Analysis:
# The ToolCallingAgent mainly uses predefined tools. It can load data and preview it, but it usually cannot generate custom plotting code unless a plotting tool exists.
# The ToolCallingAgent probably did not change the dot color to green because no plotting tool with color control was provided.
# The CodeAgent can write and execute Python code directly. It can create the scatter plot and customize details like green dots.

# This shows that ToolCallingAgents are better when:
# - tasks are predictable
# - actions are limited and controlled
# - safety and reliability are important

# CodeAgents are better when:
# - tasks are flexible
# - custom analysis or plotting is needed
# - the agent must combine many operations dynamically

# --- Q9 ---
# A ToolCallingAgent is better for tasks with a small, well-defined set of actions.
# A customer-support chatbot that can:
# - check order status
# - cancel orders
# - issue refunds

# This works well because the allowed actions are limited, structured, and predictable.
# A tool-based approach is safer because the agent canonly use approved tools.

# One important risk of a CodeAgent is that it generates and runs Python code dynamically.
# Bad or incorrect code could:
# - delete files
# - expose data
# - crash the program
# - use too much memory or CPU
# This risk does not apply as strongly to ToolCallingAgents because they can only call predefined tools instead of executing arbitrary code.