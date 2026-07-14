import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from prefect import task, flow
from prefect.logging import get_run_logger


# Part 2: Mini-Project: World Happiness Pipeline

# Task 1: load multiple years of data
@task(retries=3, retry_delay_seconds=2)  # 1 initial + 3 retries
def merge_data(): 
    logger = get_run_logger()
    df_merged = pd.DataFrame()

    # read all files from CSV
    for year in range(2015, 2025): 
        df = pd.read_csv(f'csv/world_happiness_{str(year)}.csv', delimiter=';', header=0)    
        df['Year'] = year
        if year == 2024: 
            df.rename(columns={'Ladder score':'Happiness score'}, inplace=True)     # renaming 2024 happiness score column
        df_merged = pd.concat([df_merged, df])
        
    # write the combined file to CSV (assignments_01/outputs/merged_happiness.csv)
    df_merged.to_csv("./outputs/merged_happiness.csv", sep=';', index=True, header=True, encoding=None)

    logger.info(df_merged.info())
    logger.info(f'Files merged, export succeeded!')
    return df_merged


# data cleaning and type conversion
@task(retries=3, retry_delay_seconds=3)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()

    df = df.copy()
    df['Happiness score'] = df['Happiness score'].str.replace(",", ".", regex=False)
    df['Happiness score'] = pd.to_numeric(df['Happiness score'], errors="coerce")
    
    df['GDP per capita'] = df['GDP per capita'].str.replace(",", ".", regex=False)
    df['GDP per capita'] = pd.to_numeric(df['GDP per capita'], errors="coerce")
    
    df = df.dropna(subset=['Happiness score', "Regional indicator"])
    
    logger.info("Data cleaned")
    return df


# Task 2: descriptive statistics
# Overall descriptive statistics for happiness_score: mean, median, and standard deviation
# Summary of the mean happiness score grouped by year and by region
def happiness_score(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    stats = df['Happiness score'].describe()
    summary = df.groupby(['Regional indicator', 'Year'])['Happiness score'].describe()
    countries = df.groupby('Country').agg({'Happiness score': 'mean'})
    countries.sort_values(by='Happiness score',ascending=False,inplace=True)
    logger.info(f'Descriptive statistics: {stats}')
    logger.info(f'Grouping: {summary}')
    logger.info(f'Country: {countries}')

# Task 3: Visual Exploration
@task
def plot(df: pd.DataFrame) -> None:
    logger = get_run_logger()
    
    # A histogram of happiness scores across all years, exported as happiness_histogram.png.
    plt.hist(df['Happiness score'], bins=20, color='green', alpha=0.7)
    plt.title("Happiness levels 2015-2024")
    plt.xlabel("Happiness score")
    plt.ylabel("Frequency")
    plt.savefig("./outputs/happiness_histogram.png")
    logger.info('Plot saved to "happiness_histogram.png"')
    plt.close()

    # A boxplot comparing happiness score distributions across years (one box per year), exported as happiness_by_year.png.
    sns.boxplot(x="Year", y="Happiness score", data=df)
    plt.title("Happiness Score Distribution by Year")
    plt.ylabel("Happiness Score")
    plt.xlabel("Year")
    plt.savefig("./outputs/happiness_by_year.png")
    logger.info('Plot saved to "happiness_by_year.png"')
    plt.close()

    # A scatter plot showing the relationship between GDP per capita and happiness score. Exported as gdp_vs_happiness.png.
    plt.scatter(df['GDP per capita'], df['Happiness score'])
    plt.title("GDP per capita vs Happiness scores")
    plt.xlabel("GDP")
    plt.ylabel("Score")
    plt.savefig("./outputs/gdp_vs_happiness.png")
    logger.info('Plot saved to "gdp_vs_happiness.png"')
    plt.close()

    # A correlation heatmap (using sns.heatmap() with annot=True) showing the Pearson correlations between 
    # all numeric columns, exported as correlation_heatmap.png.
    correlation_matrix = df.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("./outputs/correlation_heatmap.png")
    logger.info('Plot saved to "correlation_heatmap.png"')
    plt.close()


# Task 4: Hypothesis Testing

# The pandemic began in early 2020. Did it affect global happiness scores? Test this directly: 
# run an independent samples t-test comparing happiness scores from 2019 to 2020.
# Log the t-statistic, p-value, the mean happiness for each group, and a plain-language interpretation of the 
# result at alpha = 0.05
@task
def run_ttest(df: pd.DataFrame) -> tuple[float, float]:
    """
    Independent samples t-test:
      - Compares average Score between Class A and Class B.
      - Returns (t_statistic, p_value).
    """
    logger = get_run_logger()

    a = df[df["Year"] == 2019]["Happiness score"]
    b = df[df["Year"] == 2020]["Happiness score"]

    t_stat, p_val = stats.ttest_ind(a, b)

    logger.info("Happiness levels t-test result: t=%.2f, p=%.4f", t_stat, p_val)    # T-test result: t=-0.52, p=0.6059
    logger.info("Since p is higher than significance level, it is not enough statistical significance to " \
    "reject the null hypothesis. The data does not show that the happiness levels change significantly from 2019 to 2020")
    
    return t_stat, p_val


# Task 5: Correlation and Multiple Comparisons
@task
def run_corr(df: pd.DataFrame) -> None:
    logger = get_run_logger()
    r, p = pearsonr(df['Happiness score'], df['GDP per capita'])
    logger.info(f"Correlation:{round(r, 2)}")
    logger.info(f"p-value:, {round(p, 4)}")


# Task 6: Summary Report
@task
def report_results(ttest_result: tuple[float, float], df: pd.DataFrame) -> None:
    """
    Report the results of the t-test with group means.
    """
    logger = get_run_logger()

    t_stat, p_val = ttest_result
    
    mean_a = df[df["Year"] == 2019]["Happiness score"].mean()
    mean_b = df[df["Year"] == 2020]["Happiness score"].mean()

    logger.info("2019 mean: %.1f, 2020 mean: %.1f", mean_a, mean_b)

    if p_val < 0.05:
        logger.info("Conclusion: The difference is statistically significant (p < 0.05).")
    else:
        logger.info("Conclusion: No statistically significant difference (p ≥ 0.05).")


    logger.info("Summary report: this report is based on data from 10 datasets 2015-2024 across 175 countries. "
    "The top 3 happiest countries are Finland, Denmark and Iceland, with the least happiest countries South Sudan, Central"
    "African Republic and Afganistan. "
    "The hypothesis testing shows that there is no statistically significant difference in happiness levels before and during"
    "the pandemic (2019 vs 2020)")


@flow
def data_pipeline_flow():
    df = merge_data()
    clean_df = clean_data(df)
    happiness_score(clean_df)
    plot(clean_df)
    result = run_ttest(clean_df)
    run_corr(clean_df)
    report_results(result, clean_df)


if __name__ == "__main__":
    data_pipeline_flow()




