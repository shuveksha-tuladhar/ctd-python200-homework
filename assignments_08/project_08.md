## Part 2: Project -- Intro and Cost Analysis

### 1. Portal Walkthrough
This is my Azure portal walkthrough video.
https://youtu.be/sbacXOc2SIU

### 2. Cost Analysis
This is video showing low and high end estimates for the infrasture cost using Azure Pricing Calculator.
https://youtu.be/OSAQKqSmhIo

**Scenario A – Lightweight Compute**

For Scenario A, we estimated the cost of running a Standard_B1s virtual machine for about 160 hours per month. The total monthly cost was approximately $7.59/month. This scenario showed how affordable cloud computing can be when resources are only used during working hours and shut down when not needed.

**Scenario B – Heavy Analytics Workload**

For Scenario B, we estimated the cost of a Standard_NC6s_v3 GPU-enabled virtual machine running 24/7, along with an Azure SQL Database and 1 TB of Blob Storage. The total monthly cost was approximately $2398.16/month. Most of the cost came from the GPU virtual machine, while storage was relatively inexpensive in comparison.

**Reflection**

The most surprising thing I found was how quickly costs increase when using GPU-enabled resources. A small VM used part-time costs very little, but running high-performance infrastructure continuously can become expensive. It was also interesting to see that storage costs were much lower than compute costs. This exercise showed why monitoring cloud spending and shutting down unused resources is important when working in the cloud.

