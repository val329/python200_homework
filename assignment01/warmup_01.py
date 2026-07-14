import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns



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


# ---------------- Descriptive Statistics ---------------- #

# 1
# Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. 
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print("mean: ", np.mean(data), "\nmedian ", np.median(data), "\nstd: ", np.std(data))

# 2
# Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 
# (use np.random.normal(65, 10, 500)). Plot a histogram with 20 bins. 
# Add a title "Distribution of Scores" and label both axes.
data = np.random.normal(loc=65, scale=10, size=500)
plt.hist(data, bins=20, color='purple', alpha=0.7)
plt.title("Distribution of Scores")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 3 
# Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison".
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score comparison")
plt.show()

# 4
# Create side-by-side boxplots comparing the two distributions. Label each boxplot appropriately 
# ("Normal" and "Exponential") and add a title "Distribution Comparison".
# Then, add a comment in your code briefly noting which distribution is more skewed, and which 
# descriptive statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution.

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.boxplot([normal_data, skewed_data], labels=["Normal", "Skewed"])
plt.title("Distribution comparison")
plt.ylabel("Value")
plt.show()

# Exponential distribution is more skewed since ot has one of the whiskers longer. Median is an appropriate measure
# For normal distribution mean is an appropriate measure since it doesn't have many outliers

# 5
# Print the mean, median, and mode of the following:
# Why are the median and mean so different for data2? Add your answer as a comment in the code.

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print("Data1 : ", np.mean(data1), np.median(data1), stats.mode(data1))
print("Data2 : ", np.mean(data2), np.median(data2), stats.mode(data2))

# Data 2 has an outlier that skews the mean


# ---------------- Hypothesis Testing ---------------- #
# 1
# Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# independent samples t-test
t_stat, p_val = stats.ttest_ind(group_a, group_b)
print("t-statistic:", t_stat)
print("p-value:", p_val)

# 2
# Using the p-value from Q1, write an if/else statement that prints whether the result is statistically 
# significant at alpha = 0.05.
if p_val < 0.05:
    print("The difference is statistically significant.")
else:
    print("No statistically significant difference detected.")

# 3
# Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_val = stats.ttest_rel(before, after)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}")

# 4
# Run a one-sample t-test to check whether the mean of scores is significantly different from a national 
# benchmark of 70. Print the t-statistic and p-value.
scores = [72, 68, 75, 70, 69, 74, 71, 73]

t_stat, p_val = stats.ttest_1samp(scores, 70)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}")

# 5
# Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. 
# Print the resulting p-value. Use the alternative parameter.
t_stat, p_val = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}")

# 6
# Write a plain-language conclusion for the result of Q1 
# Your conclusion should mention the direction of the difference and whether it is likely due to chance.
print("The two-tailed t-test conducted on Q1 dataset shows the p-value was less than the significance level, " \
"which means the result is unlikely due to chance")



# ---------------- Correlation ---------------- #
# 1
# Compute the Pearson correlation between x and y below using np.corrcoef(). Print the full correlation matrix, 
# then print just the correlation coefficient (the value at position [0, 1]).
# What do you expect the correlation to be, and why? Add your answer as a comment in the code.

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print(corr_matrix, corr_matrix[0,1])
# The correlation is very strong because all elements of dataset are multipliers of another dataset

# 2
# Use pearsonr() from scipy.to compute the correlation between x and y below. 
# Print both the correlation coefficient and the p-value.
from scipy.stats import pearsonr

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

r, p = pearsonr(x, y)
print("Correlation:", round(r, 2))
print("p-value:", round(p, 4))

# 3
# Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result.
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
print(df.corr())

# 4
# Create a scatter plot of x and y below, which have a negative relationship. 
# Add a title "Negative Correlation" and label both axes.
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y, color="green")
plt.title("Negative correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 5
# Using the correlation matrix from Q3, create a heatmap with sns.heatmap(). Pass annot=True so the correlation 
# values appear in each cell, and add a title "Correlation Heatmap".
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()



# ---------------- Pipeline ---------------- #

# 1
# A data pipeline is a sequence of processing steps where each step takes in data, transforms it, and passes the result to the next. You don't need a special framework to build one -- chaining plain functions together is often enough.
# Given the array below, which contains some missing values scattered throughout:

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


def create_series(arr): 
    # takes a NumPy array and returns a pandas Series with the name "values"
    return pd.Series(arr, name='values')

def clean_data(series):
    # takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series.
    return series.dropna()

def summarize_data(series):
    # takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode"
    stats = {
        'mean': np.mean(series),
        'median': np.median(series),
        'std': np.std(series),
        'mode': series.mode()[0]
    }
    return stats

def data_pipeline(arr):
    #calls the cleaning and summarizing functions in sequence and returns the summary dictionary
    values = create_series(arr)
    values = clean_data(values)
    values = summarize_data(values)
    return values

for key, value in data_pipeline(arr).items():
    print(f"{key}: {value}")


