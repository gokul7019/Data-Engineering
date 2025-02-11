from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, to_date, lit, mean, stddev, count, regexp_replace, initcap, round, datediff
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create a Spark session
spark = SparkSession.builder \
    .appName("Healthcare Data Analysis") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# Read the data
df = spark.read.csv("C://Pyspark Project//healthcare_data_streamlit.csv", header=True, inferSchema=True)

# Drop duplicates
df = df.dropDuplicates()

# Rename columns to remove spaces
new_column_names = [col(c).alias(c.replace(" ", "_")) for c in df.columns]
df = df.select(*new_column_names)

# Filter data for the last two years
df = df.filter(col("Date_of_Admission") >= to_date(lit("2022-01-01"), "yyyy-MM-dd"))

# Convert column names to camel case
def to_camel_case(column_name):
    return initcap(regexp_replace(column_name, "_", " "))

df = df.withColumn("Name", to_camel_case(col("Name")))

# Convert date columns to datetime
df = df.withColumn("Date_of_Admission", to_date(col("Date_of_Admission"), "MM/dd/yyyy"))
df = df.withColumn("Discharge_Date", to_date(col("Discharge_Date"), "MM/dd/yyyy"))

# Round billing amount to 2 decimal places
df = df.withColumn("Billing_Amount", round(col("Billing_Amount"), 2))

# Calculate length of stay
df = df.withColumn("Length_of_Stay", datediff(col("Discharge_Date"), col("Date_of_Admission")))


# Calculate basic summary statistics
summary_stats = df.select(
    count("*").alias("count"),
    round(mean("Age"), 2).alias("mean_age"),
    round(mean("Length_of_Stay"), 2).alias("mean_length_of_stay"),
    round(mean("Billing_Amount"), 2).alias("mean_billing_amount"),
    round(stddev("Age"), 2).alias("stddev_age"),
    round(stddev("Length_of_Stay"), 2).alias("stddev_length_of_stay"),
    round(stddev("Billing_Amount"), 2).alias("stddev_billing_amount")
)
summary_stats.show(truncate=False)

# Calculate median for numerical columns without rounding
median_age = df.approxQuantile("Age", [0.5], 0.01)[0]
median_billing = df.approxQuantile("Billing_Amount", [0.5], 0.01)[0]
median_length_of_stay = df.approxQuantile("Length_of_Stay", [0.5], 0.01)[0]

print(f"Median Age: {median_age}")
print(f"Median Billing Amount: {median_billing}")
print(f"Median Length of Stay: {median_length_of_stay}")

# Data Aggregation and Grouping
# Group by patient demographics and calculate average length of stay
df.groupBy("Age", "Gender").agg(round(mean("Length_of_Stay"), 2).alias("Average_Length_of_Stay")).show()

# Group by hospital and calculate readmission rate
df.groupBy("Hospital").agg(round(mean(when(col("Readmission") == "Yes", 1).otherwise(0)), 2).alias("Readmission_Rate")).show()

# Group by diagnosis type and calculate average length of stay
df.groupBy("Medical_Condition").agg(round(mean("Length_of_Stay"), 2).alias("Average_Length_of_Stay")).show()

# Additional Analysis Possibilities

# Distribution of patients by blood type
df.groupBy("Blood_Type").count().show()

# Average billing amount by insurance provider
df.groupBy("Insurance_Provider").agg(round(mean("Billing_Amount"), 2).alias("Average_Billing_Amount")).show()

# Survival rate by medical condition
df.groupBy("Medical_Condition").agg(round(mean(when(col("Survived") == "Yes", 1).otherwise(0)), 2).alias("Survival_Rate")).show()

# Data Visualization
# Convert to Pandas DataFrame for visualization
pandas_df = df.toPandas()

# EDA: Summary Statistics
print("Summary Statistics:")
print(pandas_df.describe())

# EDA: Distribution Plots
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
sns.histplot(pandas_df['Age'], bins=30, kde=True, color='blue')
plt.title('Age Distribution')
plt.subplot(1, 3, 2)
sns.histplot(pandas_df['Billing_Amount'], bins=30, kde=True, color='green')
plt.title('Billing Amount Distribution')
plt.subplot(1, 3, 3)
sns.histplot(pandas_df['Length_of_Stay'], bins=30, kde=True, color='red')
plt.title('Length of Stay Distribution')
plt.show()

# EDA: Missing Values
print("Missing Values:")
print(pandas_df.isnull().sum())

# Line Chart: Average Length of Stay by Medical Condition
plt.figure(figsize=(10, 6))
avg_length_of_stay = pandas_df.groupby('Medical_Condition')['Length_of_Stay'].mean().reset_index()
sns.lineplot(x='Medical_Condition', y='Length_of_Stay', data=avg_length_of_stay, marker='o', color='purple')
plt.xticks(rotation=45)
plt.title('Average Length of Stay by Medical Condition')
plt.show()

# Line Chart: Average Length of Stay by Gender
plt.figure(figsize=(10, 6))
avg_length_of_stay_gender = pandas_df.groupby('Gender')['Length_of_Stay'].mean().reset_index()
sns.lineplot(x='Gender', y='Length_of_Stay', data=avg_length_of_stay_gender, marker='o', color='orange')
plt.title('Average Length of Stay by Gender')
plt.show()

# Pie Chart: Distribution of Patients by Medical Condition
plt.figure(figsize=(8, 8))
medical_condition_distribution = pandas_df['Medical_Condition'].value_counts()
plt.pie(medical_condition_distribution, labels=medical_condition_distribution.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set3"))
plt.title('Distribution of Patients by Medical Condition')
plt.show()

# Step Plot: Trend of Length of Stay over Time by Medical Condition
plt.figure(figsize=(10, 6))
pandas_df['Date_of_Admission'] = pd.to_datetime(pandas_df['Date_of_Admission'])
length_of_stay_trend = pandas_df.groupby(['Date_of_Admission', 'Medical_Condition'])['Length_of_Stay'].mean().reset_index()
sns.lineplot(x='Date_of_Admission', y='Length_of_Stay', hue='Medical_Condition', data=length_of_stay_trend, palette='tab10', drawstyle='steps-post')
plt.xticks(rotation=45)
plt.title('Trend of Length of Stay over Time by Medical Condition')
plt.show()

# Box Plot: Distribution of Length of Stay by Medical Condition
plt.figure(figsize=(10, 6))
sns.boxplot(x='Medical_Condition', y='Length_of_Stay', data=pandas_df, palette='Set2')
plt.xticks(rotation=45)
plt.title('Distribution of Length of Stay by Medical Condition')
plt.show()

# Line Chart: Treatment Success Rate by Medical Condition
treatment_success_rate = pandas_df.groupby('Medical_Condition')['Test_Results'].apply(lambda x: (x == 'Normal').mean()).reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='Medical_Condition', y='Test_Results', data=treatment_success_rate, marker='o', color='brown')
plt.xticks(rotation=45)
plt.title('Treatment Success Rate by Medical Condition')
plt.show()

# Line Chart: Survival Count by Medical Condition
survival_count = pandas_df.groupby('Medical_Condition')['Survived'].apply(lambda x: (x == 'Yes').sum()).reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='Medical_Condition', y='Survived', data=survival_count, marker='o', color='cyan')
plt.xticks(rotation=45)
plt.title('Survival Count by Medical Condition')
plt.show()
