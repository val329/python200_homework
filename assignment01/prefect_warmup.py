import pandas as pd
import numpy as np
from prefect import task, flow


# ---------------- Pipeline ---------------- #

# 2
# Rebuild the pipeline from Q1 using Prefect. Copy your three functions from Pipeline Question 1 
# (create_series, clean_data, summarize_data) into this file and turn them into Prefect tasks using @task.

# Define tasks
@task
def create_series(arr): 
    # takes a NumPy array and returns a pandas Series with the name "values"
    return pd.Series(arr, name='values')

@task
def clean_data(series):
    # takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series.
    return series.dropna()

@task
def summarize_data(series):
    # takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode"
    stats = {
        'mean': np.mean(series),
        'median': np.median(series),
        'std': np.std(series),
        'mode': series.mode()[0]
    }
    return stats

# Define a flow that uses the tasks
@flow
def data_pipeline(arr):
    #calls the cleaning and summarizing functions in sequence and returns the summary dictionary
    values = create_series(arr)
    values = clean_data(values)
    values = summarize_data(values)
    return values


arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# To run the flow
if __name__ == "__main__":  
    print(data_pipeline(arr))


# Why might Prefect be more overhead than it is worth here?
# prefect loads extra packages and takes up resources. Since there's no additional requirements besides a simple print,
# it might not be worth using it in this case

# Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline 
# logic itself stays simple like in this case.
# In case if additional functionality like dashboards are needed
