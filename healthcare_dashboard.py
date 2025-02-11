import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_date, round, mean, datediff

@st.cache_data
def load_data():
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Healthcare Data Analysis") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()

    # Read the data
    df = spark.read.csv("C://Pyspark Project//healthcare_dataset.csv", header=True, inferSchema=True)

    # Drop duplicates
    df = df.dropDuplicates()

    # Rename columns to remove spaces
    new_column_names = [col(c).alias(c.replace(" ", "_")) for c in df.columns]
    df = df.select(*new_column_names)

    # Drop duplicates based on specific columns
    df = df.dropDuplicates(subset=["Name", "Gender", "Blood_Type", "Medical_Condition", "Date_of_Admission", "Doctor", "Hospital", "Insurance_Provider", "Billing_Amount", "Room_Number", "Admission_Type", "Discharge_Date", "Test_Results"])

    # Convert column names to camel case
    def to_camel_case(column_name):
        return col(column_name).alias(column_name.replace("_", " ").title())

    df = df.select([to_camel_case(c) for c in df.columns])

    # Convert date columns to datetime
    df = df.withColumn("Date Of Admission", to_date(col("Date Of Admission"), "MM/dd/yyyy"))
    df = df.withColumn("Discharge Date", to_date(col("Discharge Date"), "MM/dd/yyyy"))

    # Round billing amount to 2 decimal places
    df = df.withColumn("Billing Amount", round(col("Billing Amount"), 2))

    # Calculate length of stay
    df = df.withColumn("Length Of Stay", datediff(col("Discharge Date"), col("Date Of Admission")))

    # Ensure billing amount is non-negative
    df = df.withColumn("Billing Amount", when(col("Billing Amount") < 0, 0).otherwise(col("Billing Amount")))

    # Convert to Pandas DataFrame for visualization
    pandas_df = df.toPandas()
    return pandas_df

# Load data
pandas_df = load_data()

# Streamlit App
st.title("Healthcare Data Analysis Dashboard")

# Sidebar filters
st.sidebar.header("Filter Data")
hospital_filter = st.sidebar.multiselect("Select Hospital", options=pandas_df["Hospital"].unique(), default=pandas_df["Hospital"].unique())
treatment_filter = st.sidebar.multiselect("Select Treatment Type", options=pandas_df["Treatment"].unique(), default=pandas_df["Treatment"].unique())

# Filter data based on selections
filtered_df = pandas_df[(pandas_df["Hospital"].isin(hospital_filter)) & (pandas_df["Treatment"].isin(treatment_filter))]

# Display filtered data
st.dataframe(filtered_df)

# Visualizations
st.header("Visualizations")

# Bar Chart: Average Length of Stay by Medical Condition
st.subheader("Average Length of Stay by Medical Condition")
avg_length_of_stay = filtered_df.groupby('Medical Condition')['Length Of Stay'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(x='Medical Condition', y='Length Of Stay', data=avg_length_of_stay, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Bar Chart: Average Billing Amount by Insurance Provider
st.subheader("Average Billing Amount by Insurance Provider")
avg_billing_amount = filtered_df.groupby('Insurance Provider')['Billing Amount'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(x='Insurance Provider', y='Billing Amount', data=avg_billing_amount, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Box Plot: Distribution of Length of Stay by Medical Condition
st.subheader("Distribution of Length of Stay by Medical Condition")
fig, ax = plt.subplots()
sns.boxplot(x='Medical Condition', y='Length Of Stay', data=filtered_df, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Bar Chart: Treatment Success Rate by Medical Condition
st.subheader("Treatment Success Rate by Medical Condition")
treatment_success_rate = filtered_df.groupby('Medical Condition')['Test Results'].apply(lambda x: (x == 'Normal').mean()).reset_index()
fig, ax = plt.subplots()
sns.barplot(x='Medical Condition', y='Test Results', data=treatment_success_rate, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Bar Chart: Survival Count by Medical Condition
st.subheader("Survival Count by Medical Condition")
survival_count = filtered_df.groupby('Medical Condition')['Survived'].apply(lambda x: (x == 'Yes').sum()).reset_index()
fig, ax = plt.subplots()
sns.barplot(x='Medical Condition', y='Survived', data=survival_count, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
