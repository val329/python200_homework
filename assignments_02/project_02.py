

# Part 2: Mini-Project -- Predicting Student Math Performance

# Task 1: Load and Explore
# Load the dataset with the correct separator. Print the shape, the first five rows, and the data types of all columns.

# Then plot a histogram of G3 with 21 bins (one per possible value, 0-20). Add a title "Distribution of Final Math Grades", label both axes, and save to outputs/g3_distribution.png. You should see a cluster of zeros sitting apart from the main distribution. They represent the students who didn't take the final exam.




# Task 2: Preprocess the Data
# Handle the G3=0 rows first. Filter them out and save the result to a new DataFrame. Print the shape before and after to confirm how many rows were removed. Add a comment explaining your reasoning -- why would keeping these rows distort the model?

# Then convert the yes/no columns to 1/0 and the sex column to 0/1.

# Now check something interesting before moving on. Compute the Pearson correlation between absences and G3 on both the original dataset and the filtered one, and print both values. The difference is striking. Add a comment explaining why filtering changes the result: what were students with G3=0 doing in the original data that made absences look like a weak predictor? You might want to explore scatter plots to help understand this.



# Task 3: Exploratory Data Analysis
# Compute the Pearson correlation between each numeric feature and G3 on the filtered dataset, and print them sorted from most negative to most positive. Which feature has the strongest relationship with G3? Are any results surprising?

# Then create at least two visualizations of your own choosing and save them to outputs/. Use your judgment from previous weeks of data engineering to guide your use of plots. Use the correlation results to guide you -- what relationships seem worth a closer look? Add a comment for each plot describing what you see.



# Task 4: Baseline Model
# Build the simplest possible model: use failures alone to predict G3. Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, and print the slope, RMSE, and R² on the test set.
# Add a comment: given that grades are on a 0-20 scale, what do the slopes and RMSE tell you in plain English? Is R² better or worse than you expected from exploratory data analysis?





# Task 5: Build the Full Model