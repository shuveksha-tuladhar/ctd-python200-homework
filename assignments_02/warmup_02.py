import numpy as np
from sklearn.linear_model import LinearRegression

# --- scikit-learn API--- 
# scikit-learn Q1 : Create a LinearRegression model, fit it to this data, and then predict the salary for someone with 4 years of experience and someone with 8 years. Print the slope (model.coef_[0]), the intercept (model.intercept_), and the two predictions. Label each printed value.
import numpy as np
from sklearn.linear_model import LinearRegression

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
new_years = np.array([4,8]).reshape(-1,1)

model = LinearRegression() 
model.fit(years, salary)
salary_predicted = model.predict(new_years)

print("Slope (coef_):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 4 years:", salary_predicted[0])
print("Predicted salary for 8 years:", salary_predicted[1])
 
