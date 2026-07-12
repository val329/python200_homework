import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
# ---------------- Pandas ---------------- #

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# 1
print(f"first 3 rows: {df.head(3)}")
print(f"shape: {df.shape}")
print(f"data types: {df.dtypes}")

# 2
# Filter the rows to show only students who passed and have a grade above 80
print(df[(df['passed']) & (df['grade'] > 80)])

# 3
# Add a new column called "grade_curved" that adds 5 points to each student's grade. 
df['grade_curved'] = df['grade'] + 5
print(df)

# 4
# Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor
df['name_upper'] = df['name'].str.upper()
print(df[['name', 'name_upper']])

# 5
# Group the DataFrame by "city" and compute the mean grade for each city. Print the result.
print(df.groupby('city')['grade'].mean())

# 6
# Replace the value "Austin" in the "city" column with "Houston". Print the "name" and "city" columns to confirm the change.
df['city'] = df['city'].str.replace('Austin', 'Houston')
print(df[['name', 'city']])

# 7
# Sort the DataFrame by "grade" in descending order and print the top 3 rows.
df.sort_values(by="grade", ascending=False, inplace=True)
print(df.head(3))



# ---------------- NumPy ---------------- #

# 1
# Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.

arr = np.array([10, 20, 30, 40, 50])
print(arr.shape, arr.dtype, arr.ndim)

# 2
# Create the following 2D array and print its shape and size (total number of elements).

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(arr.shape, arr.size)

# 3 
# Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]].
print(arr[0:2,0:2])

# 4
# Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both.
arr0 = np.zeros((3,4))
arr1 = np.ones((2,5))
print(arr0, arr1)

# 5
# Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like. 
# Then, print the array, its shape, mean, sum, and standard deviation.
arr = np.arange(0,50,5)
print(arr, arr.shape, arr.mean(), arr.sum(), arr.std())

# 6
# Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 
# (use np.random.normal()). Print the mean and standard deviation of the result.
arr = np.random.normal(loc=0.0, scale=1.0, size=(200,))
print(arr.mean(), arr.std())


# ---------------- Matplotlib ---------------- #

# 1
# Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y, marker='o', linestyle='-', color='blue')
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# 2
# Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.bar(subjects, scores, color=['green', 'orange'])
plt.title("Subject scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()


# 3
# Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, 
# add a legend, and label both axes.
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color="green")
plt.scatter(x2, y2, color="red")
plt.title("x and y")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# 4
# Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. 
# In the left subplot, plot x vs y from Q1 as a line. 
# In the right subplot, plot the subjects and scores from Q2 as a bar plot. 


# Subplot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left plot: Bar Plot
axes[0].plot(x, y, marker='o', linestyle='-', color='blue')
axes[0].set_title("Squares")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# Right plot: Histogram
axes[1].bar(subjects, scores, color=['green', 'orange'])
axes[1].set_title("Subject scores")
axes[1].set_xlabel("subjects")
axes[1].set_ylabel("scores")

plt.tight_layout()  # Adjust layout for better spacing
plt.show()

'''




# ---------------- Descriptive Statistics ---------------- #




# ---------------- Hypothesis Testing ---------------- #



# ---------------- Correlation ---------------- #




# ---------------- Pipeline ---------------- #



