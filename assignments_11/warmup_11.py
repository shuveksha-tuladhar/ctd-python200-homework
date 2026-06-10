# --- Part 1: Warmup ---
# --- Prefect Orchestration ---
# Q1

# A @task represents an individual unit of work in a Prefect workflow. Tasks can be tracked, retried, logged, and monitored independently. 
# A @flow is the orchestration layer that coordinates one or more tasks. Flows define the overall workflow and execution order. It is the workflow that coordinates and runs tasks. 
# I would not decorate a helper function that converts Celsius to Fahrenheit with @task because it is a simple in-memory calculation with no I/O, retries, logging, or orchestration needs. It is better kept as a regular Python function.

# Q2
# @task(retries=3, retry_delay_seconds=30)

# Q3
# I would open the failed flow run in the Prefect UI and look at the failed transform task. 
# The task run details and logs would show the error message, stack trace, retry attempts, timestamps, and any task-specific information that explains why the task failed. 
# This helps identify the root cause and understand why the load task never ran.