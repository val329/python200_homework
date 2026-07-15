import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


'''
# The scikit-learn API

# 1. scikit-learn Question 1
# Create a LinearRegression model, fit it to this data, and then predict the salary for someone 
# with 4 years of experience and someone with 8 years. Print the slope (model.coef_[0]), the intercept 
# (model.intercept_), and the two predictions. Label each printed value.

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# years of work experience vs annual salary
x_values = years
print(x_values.shape, x_values.ndim)
y_values = salary
new_x = np.array([4,8]).reshape(-1, 1)

model = LinearRegression()                    # 1. create model
model.fit(x_values, y_values)                 # 2. fit model to data (learn)
y_predicted = model.predict(new_x)            # 3. predict with new data
print(f'Slope: {model.coef_[0]}, intercept {model.intercept_}, predicted values: {y_predicted}')


# 2. scikit-learn Question 2
# scikit-learn requires the feature array X to be 2D even when you only have one feature. Start with this 1D array:
# Print its shape. Use .reshape() to convert it to a 2D array and print the new shape. Add a comment explaining, i
# n your own words, why scikit-learn needs X to be 2D.

x = np.array([10, 20, 30, 40, 50])
print(x.shape)
x = x.reshape(-1,1)
print(x.shape, x.ndim)
# scikit-learn is optimized to work with matrices (rows of data and at least one feature), so the minimum dimensions is 2


# 3. scikit-learn Question 3
# K-Means is an unsupervised algorithm that follows the same create → fit → predict pattern as everything else 
# in scikit-learn. Use the code below to generate a synthetic dataset with three natural clusters:
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
print(X_clusters.shape)

# Create a KMeans model with n_clusters=3 and random_state=42, fit it to X_clusters, and predict a cluster label for each point. 
# Print the cluster centers (kmeans.cluster_centers_) and how many points fell into each cluster using np.bincount(labels).
# Then create a scatter plot coloring each point by its cluster label, plot the cluster centers as black X's, 
# add a title and axis labels. Save the figure to outputs/kmeans_clusters.png.

# fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42)            # 1. Create the model
kmeans.fit(X_clusters)                                    # 2. Fit -- find cluster centers
labels = kmeans.predict(X_clusters)                       # 3. Predict a label for each point
centers = kmeans.cluster_centers_

print(f"Cluster centers: {centers}, points: {np.bincount(labels)}")

# plot data points
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)

# plot cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.9, marker='X', label='Centroids')

plt.title("Clusters found by K-Means")
plt.savefig("./outputs/kmeans_clusters.png")
'''


# Linear Regression

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# 1. Linear Regression Question 1
# Before fitting anything, look at the data. Create a scatter plot of age on the x-axis and cost on the y-axis. 
# Color the points by smoker status by passing c=smoker and cmap="coolwarm" to plt.scatter(). Add a title "Medical Cost vs Age", 
# label both axes, and save to outputs/cost_vs_age.png.
# Add a comment describing what you see. Are there two distinct groups visible? What does that suggest about the smoker variable?
plt.scatter(age, cost, c=smoker, cmap='coolwarm')
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical cost")
plt.savefig("./outputs/cost_vs_age.png")
# two distinct groups visible, seem to be categorized by smoker variable. One group tends to have lower medical costs than the other


# 2. Linear Regression Question 2
# Split the data into training and test sets using age as the only feature, an 80/20 split, and random_state=42. 
# Reshape age to a 2D array before using it as X. Print the shapes of all four arrays.
X = age.reshape(-1,1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# 3. Linear Regression Question 3
# Fit a LinearRegression model to your training data from Question 2. Print the slope and intercept. 
# Then predict on the test set and print:
# RMSE: np.sqrt(np.mean((y_pred - y_test) ** 2))
# R² on the test set: model.score(X_test, y_test)
# Add a comment interpreting the slope in plain English -- what does it mean for medical costs?

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

RMSE = np.sqrt(np.mean((y_pred - y_test) ** 2))
R2 = model.score(X_test, y_test)

print(f'Slope: {model.coef_[0]}, intercept {model.intercept_}')
print(f'RMSE: {RMSE}, R2: {R2}')

# Slope describes the relationship between age and medical costs. Since slope is positive, the relationship is positive


# 4. Linear Regression Question 4
# Now add smoker as a second feature and fit a new model.
X_full = np.column_stack([age, smoker])

# Split, fit, and print the test R². Compare it to the R² from Question 3 -- does adding the smoker flag help? 
# Print both coefficients

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)
y_pred = model_full.predict(X_test)

RMSE = np.sqrt(np.mean((y_pred - y_test) ** 2))
R2 = model_full.score(X_test, y_test)
print(f'RMSE: {RMSE}, R2: {R2}')

print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])
# Add a comment interpreting the smoker coefficient: what does it represent in practical terms?
# The R2 is higher with two feature model, which means that the smoker feature is relevant. Smoker coefficient
# tells how much medical cost increases for every age year holding smoker constant. 


# 5. Linear Regression Question 5
# A predicted vs actual plot is a standard tool for evaluating regression models. Each test observation becomes a dot: 
# the model's prediction goes on the x-axis, the true value goes on the y-axis. A perfect model would place every point 
# on the diagonal line where predicted equals actual.
# Using the two-feature model from Linear Regression Question 4, create this plot for the test set. Add a diagonal reference line, 
# a title "Predicted vs Actual", labeled axes, and save to outputs/predicted_vs_actual.png.
# Add a comment: what does it mean when a point falls above the diagonal? What about below?

plt.scatter(X_test[:, 0], y_test, c=X_test[:, 1], cmap='coolwarm')

plt.title("Predicted vs Actual")
plt.xlabel("Age")
plt.ylabel("Medical cost")
plt.savefig("./outputs/predicted_vs_actual.png")
