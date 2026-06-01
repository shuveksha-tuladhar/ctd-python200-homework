# project_08.py
# Run this in Azure Cloud Shell after completing the Cost Analysis above.

# Fill in the hourly rates from your two Pricing Calculator estimates.
rate_a = 1.66    # Standard_B1s hourly rate (Scenario A)
rate_b = 657    # Standard_NC6s_v3 hourly rate (Scenario B, VM only)

hours_a = 160   # Scenario A: 8h/day, 5 days/week, ~4 weeks
hours_b = 730   # Scenario B: always on

cost_a = rate_a * hours_a
cost_b = rate_b * hours_b

print("=== Monthly Cost Estimates ===")
print(f"Scenario A (lightweight):       ${cost_a:.2f}")
print(f"Scenario B (GPU VM only):       ${cost_b:.2f}")

if cost_a > 0:
    print(f"Scenario B VM costs {cost_b / cost_a:.1f}x more than Scenario A")
    
# The Python script calculated a monthly cost of $265.60 for Scenario A and $479,610.00 for Scenario B. 
# It also showed that the GPU VM in Scenario B costs about 1,805.8 times more than the lightweight VM in Scenario A.
# The results demonstrated how quickly cloud costs can increase when using high-performance resources such as GPU-enabled virtual machines. 
# The difference between the two scenarios was much larger than I expected.