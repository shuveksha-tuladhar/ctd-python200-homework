from azure.identity import DefaultAzureCredential
from azure.mgmt.subscription import SubscriptionClient

credential = DefaultAzureCredential()
client = SubscriptionClient(credential)

for sub in client.subscriptions.list():
    print(sub.display_name)

