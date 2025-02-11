import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_date, round, mean, datediff, initcap, regexp_replace

@st.cache_data
def load_data():
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

    # Drop duplicates based on specific columns
    df = df.dropDuplicates(subset=["Name", "Gender", "Blood_Type", "Medical_Condition", "Date_of_Admission", "Doctor", "Hospital", "Insurance_Provider", "Billing_Amount", "Room_Number", "Admission_Type", "Discharge_Date", "Test_Results"])

    # Convert column names to camel case
    def to_camel_case(column_name):
        return initcap(regexp_replace(column_name, "_+", " "))

    df = df.withColumn("Name", to_camel_case(col("Name")))

    # Convert date columns to datetime
    df = df.withColumn("Date_Of_Admission", to_date(col("Date_Of_Admission"), "MM/dd/yyyy"))
    df = df.withColumn("Discharge_Date", to_date(col("Discharge_Date"), "MM/dd/yyyy"))

    # Round billing amount to 2 decimal places
    df = df.withColumn("Billing_Amount", round(col("Billing_Amount"), 2))

    # Calculate length of stay
    df = df.withColumn("Length_Of_Stay", datediff(col("Discharge_Date"), col("Date_Of_Admission")))

    # Ensure billing amount is non-negative
    df = df.withColumn("Billing_Amount", when(col("Billing_Amount") < 0, 0).otherwise(col("Billing_Amount")))

    # Convert to Pandas DataFrame for visualization
    pandas_df = df.toPandas()
    return pandas_df

# Load data
pandas_df = load_data()

# Streamlit App
st.title("Healthcare Data Analysis Dashboard")

# Sidebar filters
st.sidebar.header("Filter Data")

# Medical Condition Filter
medical_conditions = pandas_df["Medical_Condition"].unique().tolist()
medical_condition_filter = st.sidebar.multiselect("Select Medical Condition", options=medical_conditions, default=medical_conditions)

# Readmission Filter
readmission_status = pandas_df["Readmission"].unique().tolist()
readmission_filter = st.sidebar.multiselect("Select Readmission Status", options=readmission_status, default=readmission_status)

# Survival Filter
survival_status = pandas_df["Survived"].unique().tolist()
survived_filter = st.sidebar.multiselect("Select Survival Status", options=survival_status, default=survival_status)

# Filter data based on selections
filtered_df = pandas_df[
    (pandas_df["Medical_Condition"].isin(medical_condition_filter)) &
    (pandas_df["Readmission"].isin(readmission_filter)) &
    (pandas_df["Survived"].isin(survived_filter))
]

# Convert date columns to datetime
filtered_df.loc[:, 'Date_Of_Admission'] = pd.to_datetime(filtered_df['Date_Of_Admission'])

# Display filtered data
st.dataframe(filtered_df)

# EDA: Summary Statistics
st.header("Exploratory Data Analysis (EDA)")
st.subheader("Summary Statistics")
st.write(filtered_df.describe())

# EDA: Distribution Plots
st.subheader("Distribution Plots")
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
sns.histplot(filtered_df['Age'], bins=30, kde=True, ax=ax[0])
ax[0].set_title('Age Distribution')
sns.histplot(filtered_df['Billing_Amount'], bins=30, kde=True, ax=ax[1])
ax[1].set_title('Billing Amount Distribution')
sns.histplot(filtered_df['Length_Of_Stay'], bins=30, kde=True, ax=ax[2])
ax[2].set_title('Length of Stay Distribution')
st.pyplot(fig)

# EDA: Missing Values
st.subheader("Missing Values")
missing_values = filtered_df.isnull().sum()
st.write(missing_values)

# Bar Chart: Average Length of Stay by Medical Condition
st.subheader("Average Length of Stay by Medical Condition")
avg_length_of_stay = filtered_df.groupby('Medical_Condition')['Length_Of_Stay'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(x='Medical_Condition', y='Length_Of_Stay', data=avg_length_of_stay, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Bar Chart: Average Length of Stay by Gender
st.subheader("Average Length of Stay by Gender")
avg_length_of_stay_gender = filtered_df.groupby('Gender')['Length_Of_Stay'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(x='Gender', y='Length_Of_Stay', data=avg_length_of_stay_gender, ax=ax)
plt.title('Average Length of Stay by Gender')
st.pyplot(fig)

# Pie Chart: Distribution of Patients by Medical Condition
st.subheader("Distribution of Patients by Medical Condition")
fig, ax = plt.subplots(figsize=(8, 8))
medical_condition_distribution = filtered_df['Medical_Condition'].value_counts()
ax.pie(medical_condition_distribution, labels=medical_condition_distribution.index, autopct='%1.1f%%', startangle=140)
ax.set_title('Distribution of Patients by Medical Condition')
st.pyplot(fig)

# Step Plot: Trend of Length of Stay over Time by Medical Condition
st.subheader("Trend of Length of Stay over Time by Medical Condition")
length_of_stay_trend = filtered_df.groupby(['Date_Of_Admission', 'Medical_Condition'])['Length_Of_Stay'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='Date_Of_Admission', y='Length_Of_Stay', hue='Medical_Condition', data=length_of_stay_trend, ax=ax, drawstyle='steps-post')
plt.xticks(rotation=45)
ax.set_title('Trend of Length of Stay over Time by Medical Condition')
st.pyplot(fig)

# Box Plot: Distribution of Length of Stay by Medical Condition
st.subheader("Distribution of Length of Stay by Medical Condition")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='Medical_Condition', y='Length_Of_Stay', data=filtered_df, ax=ax)
plt.xticks(rotation=45)
ax.set_title('Distribution of Length of Stay by Medical Condition')
st.pyplot(fig)

# Bar Chart: Treatment Success Rate by Medical Condition
st.subheader("Treatment Success Rate by Medical Condition")
treatment_success_rate = filtered_df.groupby('Medical_Condition')['Test_Results'].apply(lambda x: (x == 'Normal').mean()).reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Medical_Condition', y='Test_Results', data=treatment_success_rate, ax=ax)
plt.xticks(rotation=45)
ax.set_title('Treatment Success Rate by Medical Condition')
st.pyplot(fig)

# Bar Chart: Survival Count by Medical Condition
st.subheader("Survival Count by Medical Condition")
survival_count = filtered_df.groupby('Medical_Condition')['Survived'].apply(lambda x: (x == 'Yes').sum()).reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Medical_Condition', y='Survived', data=survival_count, ax=ax)
plt.xticks(rotation=45)
ax.set_title('Survival Count by Medical Condition')
st.pyplot(fig)

# Bar Chart: Readmission Rate by Medical Condition
st.subheader("Readmission Rate by Medical Condition")
readmission_rate = filtered_df.groupby('Medical_Condition')['Readmission'].apply(lambda x: (x == 'Yes').mean()).reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Medical_Condition', y='Readmission', data=readmission_rate, ax=ax)
plt.xticks(rotation=45)
ax.set_title('Readmission Rate by Medical Condition')
st.pyplot(fig)

# Bar Chart: Mortality Rate by Medical Condition
st.subheader("Mortality Rate by Medical Condition")
mortality_rate = filtered_df.groupby('Medical_Condition')['Survived'].apply(lambda x: (x == 'No').mean()).reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Medical_Condition', y='Survived', data=mortality_rate, ax=ax)
plt.xticks(rotation=45)
ax.set_title('Mortality Rate by Medical Condition')
st.pyplot(fig)
