# 🚢 Titanic Dataset - Exploratory Data Analysis (EDA)

This repository contains an in-depth Exploratory Data Analysis (EDA) on the **Titanic dataset** available through Seaborn's built-in datasets. This work is submitted as **Week 4 Assignment** for the **Celebal Technologies Data Science Internship**.

---

## 📌 Assignment Objective

> **Perform comprehensive EDA on the Titanic dataset and derive meaningful insights through statistical and visual analysis.**

---

## 📊 Dataset Details

- **Source**: Seaborn's `titanic` dataset (`sns.load_dataset('titanic')`)
- **Rows**: 891  
- **Columns**: 15  
- **Target Variable**: `survived`

---

## 🔍 EDA Coverage

The notebook performs the following steps:

### ✅ 1. Dataset Overview
- Structure, data types, column names, sample entries

### ✅ 2. Missing Values
- Count & percentage
- Bar plots showing missing value distribution

### ✅ 3. Descriptive Statistics
- Summary of numerical and categorical variables

### ✅ 4. Univariate Analysis
- Distribution plots for features like `age`, `fare`, `class`, etc.

### ✅ 5. Bivariate Analysis
- Survival relationships with `sex`, `pclass`, `embarked`, etc.
- Box plots and bar charts

### ✅ 6. Correlation Analysis
- Heatmap of numerical features
- Insight into linear relationships

### ✅ 7. Outlier Detection
- IQR method applied to key numerical columns

### ✅ 8. Feature Engineering
- Created features: `FamilySize`, `IsAlone`, `AgeGroup`, `Title`
- Analyzed survival rates using engineered features

### ✅ 9. Advanced Statistical Insights
- Chi-Square test for independence (`sex` vs `survived`)
- Multi-variable group survival analysis (`sex` & `pclass`)

### ✅ 10. Final Summary
- Data summary and key insights

---

## 🧠 Key Insights

- **Women had a 74.2% survival rate** vs **18.9% for men**
- **1st Class passengers** had the highest survival probability
- Being alone (IsAlone = 1) negatively impacted survival
- Family size and title have strong correlations with survival outcomes

---