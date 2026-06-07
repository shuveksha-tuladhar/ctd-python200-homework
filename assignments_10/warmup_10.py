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
