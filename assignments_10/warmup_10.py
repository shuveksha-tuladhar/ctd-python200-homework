# --- LLMs as Transform ---

# Q1

# Parse the string "Jan 5th, 2024" into an ISO date format like "2024-01-05".
# I would use deterministic code because date parsing follows a predictable format.

# Classify a customer support ticket -- "my card was charged twice" -- into one of: billing, technical, or general.
# I would use an LLM because it can understand the meaning of freeform text and classify it correctly.

# Calculate the average of a list of numbers.
# I would use deterministic code because mathematical calculations should be exact.

# Extract the company name from a freeform job title like "Sr. Data Eng @ Acme Corp (contract)".
# I would use an LLM because the text format may vary and requires language understanding.

# Determine whether a product review is more than 100 words long.
# I would use deterministic code because counting words is a straightforward rule-based task.


# Q2

# The prompt creates a problem because the output format is not defined, so the summary could vary in length and structure, making it difficult to parse and store consistently in a pipeline.

# Improved prompt:
# system = """
# Summarize the product review and return only valid JSON in this format:
# {"summary": "<summary text>"}
# """

# Q3

# If each of the 50,000 calls takes 1 second, sequential processing would take about 50,000 seconds, which is approximately 13.9 hours.
# One practical strategy is to process requests concurrently in batches so that multiple API calls run at the same time instead of one after another.

# --- Azure OpenAI ---

# Q1

# Two reasons an organization might use Azure OpenAI:
# 1. It integrates with existing Azure services, security controls, and identity management.
# 2. It helps organizations meet compliance, governance, and data residency requirements.

# Q2

# The three Azure-specific parameters are:
# azure_endpoint: he URL of the Azure OpenAI resource.
# api_version: The Azure OpenAI API version to use for requests.
# model (deployment name): The name of the deployed model configured in the Azure OpenAI resource.

# Q3

# When using AzureOpenAI, the model parameter takes the deployment name rather than a model name such as "gpt-4o-mini".
# The deployment name is defined when the model is deployed in Azure OpenAI and can be found in the Azure Portal under the Azure OpenAI resource's deployments.
