
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
sns.set_theme(style="whitegrid")

# Create output folder
os.makedirs("outputs", exist_ok=True)

# 1. Load dataset
print("Loading Titanic Dataset...")
df = pd.read_csv("train.csv")

print("\nDataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())

# 2. Understand dataset
print("\nDataset Shape:", df.shape)

print("\nColumn Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# 3. Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# 4. Data Cleaning

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Create a feature indicating whether Cabin information is available
df["Has_Cabin"] = df["Cabin"].notna().astype(int)

# Fill missing Cabin values
df["Cabin"] = df["Cabin"].fillna("Unknown")

# Create FamilySize feature
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Create IsAlone feature
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("outputs/cleaned_titanic.csv", index=False)

# 5. Exploratory Data Analysis (EDA)

# Survival count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Survived")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.savefig("outputs/survival_count.png", dpi=300)
plt.show()

# Survival by gender
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.savefig("outputs/survival_by_gender.png", dpi=300)
plt.show()

# Survival by passenger class
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.savefig("outputs/survival_by_class.png", dpi=300)
plt.show()

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.savefig("outputs/age_distribution.png", dpi=300)
plt.show()

# Fare distribution by passenger class
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Pclass", y="Fare")
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Ticket Fare")
plt.savefig("outputs/fare_by_class.png", dpi=300)
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=300)
plt.show()

# 6. Calculate survival rates
print("\n========== EDA INSIGHTS ==========")

print("\nOverall Survival Rate:")
print(round(df["Survived"].mean() * 100, 2), "%")

print("\nSurvival Rate by Gender:")
print((df.groupby("Sex")["Survived"].mean() * 100).round(2))

print("\nSurvival Rate by Passenger Class:")
print((df.groupby("Pclass")["Survived"].mean() * 100).round(2))

print("\nSurvival Rate by Embarkation Port:")
print((df.groupby("Embarked")["Survived"].mean() * 100).round(2))

print("\nSurvival Rate by Family Size:")
print((df.groupby("FamilySize")["Survived"].mean() * 100).round(2))

print("\nCleaned dataset saved successfully!")
print("All charts saved in the outputs folder.")
print("\nPRODIGY TASK 02 ANALYSIS COMPLETED!")
