## Cloud Concepts
### Cloud Concepts Question 1
The core economic model of cloud computing is pay for what we use. Instead of buying and maintaining our own servers, we rent computing resources from a cloud provider and only pay when we use them.

### Cloud Concepts Question 2
- Vertical scaling means making one machine bigger by adding more CPU, RAM, or storage. Example: Upgrading a database server with more memory because it is running slowly.
- Horizontal scaling means adding more machines and spreading the work across them. Example: Adding more web servers when a website gets a lot more traffic.

**Scenarios**:
- Web app grows from 1,000 to 100,000 users: Horizontal scaling because the traffic can be spread across multiple servers.
- Model training needs faster GPU and more RAM: Vertical scaling because a more powerful machine is needed.
- Data pipeline grows from 10 files to 10,000 files: Horizontal scaling because the files can be processed by many machines at the same time.

### Cloud Concepts Question 3
Classification
- **Gmail**: SaaS because users access a complete application without managing infrastructure.
- **Azure Virtual Machines**: IaaS because we get virtual servers and manage the operating system and software ourself.
- **Azure App Service**: PaaS because Azure manages the infrastructure while we deploy our application.
- **AWS S3**: IaaS because it provides storage infrastructure that we use and manage.
- **GitHub Codespaces**: PaaS because GitHub manages the development environment infrastructure.
- **Snowflake**: SaaS because it is a fully managed data platform that users access as a service.

**IaaS (Infrastructure as a Service)**: The cloud provider gives the computing resources like servers, storage, and networking. We manage the operating system and applications. Example: Azure Virtual Machines.

**PaaS (Platform as a Service)**: The cloud provider manages the infrastructure and platform, and user focus on the application code. Example: Azure App Service.

**SaaS (Software as a Service)**: The provider delivers a complete application that users access through the web. We only use the software. Example: Gmail.

### Cloud Concepts Question 4
A managed data platform like Databricks or Snowflake provides ready-to-use tools for data storage, processing, and analytics. Compared to using Azure directly, we gain convenience and less infrastructure management, but we give up some control and customization.

### Cloud Concepts Question 5
The cloud is probably not the right choice when:

- We have a dataset and workload that can easily run on a single machine, because local processing is often faster and cheaper.
- We are building an initial prototype and do not need large-scale cloud resources, since cloud infrastructure can add extra complexity and costs.

## Azure Basics

### Azure Basics Question 1
An Azure subscription is the main account that owns and pays for all Azure resources. A resource group is a container that organizes related resources for a project. The subscription is shared by CTD, while we each have our own resource group.

### Azure Basics Question 2
Cloud Shell is ephemeral by default, which means files and folders are deleted when the session ends. Our course setup uses a storage account and file share to make our Cloud Shell home directory persistent, so our files stay available between sessions.

### Azure Basics Question 3
The SSH private key is secret and stays on our machine. The SSH public key is shared with remote systems we want to connect to. The public key gets uploaded because it can be used to verify our identity, but it cannot be used to recreate or access our private key.

### Azure Basics Question 4
Without ``--output table``, the command ``az account show`` displays the information in JSON format, which contains all the details in a structured text format. When we add ``az account show --output table``, the same information is shown in a simpler table format that is easier for people to read.

```json
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#tuladhar.shuveksha@gmail.com",
    "type": "user"
  }
}
```