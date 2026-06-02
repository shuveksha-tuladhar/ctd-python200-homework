from prefect import task, flow

# Define tasks
@task
def say_hello(name):
    print(f"Hello, {name}!")

@task
def say_goodbye(name):
    print(f"Goodbye, {name}!")

# Define a flow that uses the tasks
@flow
def chatty_pipeline(name):
    say_hello(name)
    say_goodbye(name)

# To run the flow
if __name__ == "__main__":  
    chatty_pipeline("Code the Dream")