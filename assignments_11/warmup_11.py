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

# --- Production Patterns ---

# Q1
# raise_for_status() automatically raises an exception when an HTTP request returns an error status such as 4xx or 5xx. 
# This is better than printing an error message because it causes the task to fail immediately, making the problem visible in the pipeline and preventing downstream tasks from running.
# If the API returns a 500 error and we only print an error message, the pipeline may continue running with bad or missing data. 
# With raise_for_status(), the task fails and downstream tasks do not run.

# Q2
# overwrite=True allows the pipeline to safely replace an existing blob when it is re-run after a failure. 
# If the pipeline crashes during the transform step, we can fix the bug and run it again without worrying about an existing output file causing an error. 
# Without overwrite=True, the upload could fail because a blob with the same name already exists.

# Q3

from prefect import task

@task
def log_loaded_records(records, blob_path):
    logger = get_run_logger()
    logger.info(
        f"Loaded {len(records)} records from {blob_path}"
    )