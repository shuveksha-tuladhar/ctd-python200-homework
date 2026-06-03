# --- Azure Authentication ---
# --- Q1 ---

# When we run a Python script locally that uses DefaultAzureCredential, it usually relies on our Azure CLI login credentials for authentication.
# Before running the script, we must run "az login" to sign in to Azure.
# DefaultAzureCredential automatically checks for available credentials, including Azure CLI credentials, and uses them if they are available.

# --- Q2 --- 
# A deployed pipeline running on an Azure VM or container cannot use "az login" because there is no user available to interactively sign in.
# Instead, it typically uses a Managed Identity assigned to the resource.
# DefaultAzureCredential can automatically detect and use the Managed Identity, which allows the same Python code to work locally and in Azure  without requiring code changes.

# Q3
# If a script immediately gets an AuthenticationError, one likely cause is that we have not run "az login" locally, so no Azure credentials are available.
# We can diagnose this by running "az account show" to verify that we are logged in.
# Another likely cause is that the account or Managed Identity does not have permission to access the Azure resource being requested. 
# We can diagnose this by checking the error message and verifying the assigned roles and permissions in Azure.