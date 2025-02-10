# -*- coding: utf-8 -*-
"""
**WQD7006 Group Assignment**

<center>

**Machine Learning Approach for Marketing Campaigns in
Food Company**

| Name                                   |
|----------------------------------- -----|
| Chong Jia En                            |
| Melanie Ng Huei Yee                     |
| Yan Yen Wei                             |
| Syaidatul Salmah Nurbalqis Binti Saiful |

# **1.0 Project Background**

iFood, a popular online food delivery platform in Brazil carried out a pilot campaign on introducing its new developed gadget to the customers. Unfortunately, iFood gained a negative profit from the campaign as the revenue generated fell short to cover the cost (Nailson, 2020). This data science project aims to develop a predictive model, to forecast customer’s behaviour in accepting the offer to achieve a better performance for the future campaign. Based on the dataset, the information includes the customer’s profiles, product preferences, campaign outcomes, and channel performance metrics.

This project targets the iFood marketing departments, includes stakeholders which responsible for planning, conducting, and executing marketing campaigns to promote iFood’s new developed products. This involves a range of tasks including strategizing, implementing, and evaluating the effectiveness of marketing campaigns.

This project aims to study customer characteristics for targeted campaign offers. By interpreting the meaningful information, the company can focus on the potential customers, leaving the non-respondents and hence conduct a campaign with higher profit (Zulaikha et al., 2020). With the study of correlations between customer’s background and actions of purchasing, iFood can enhance the customer segmentation and convey advanced service to the customers. Additionally, this project’s finding may also be valuable in generating a predictive model to estimate the success of the future campaigns, by forecasting the customer’s purchasing intention.

## **1.1 Problem Statement**

The negative profit generated from the pilot campaign of iFood shows the need for a data-driven approaches to improve its marketing strategies. It is challenging to accurately estimate customer’s behaviour due to insufficient predictive capabilities of iFood. According to Zulaikha et al. (2020), the company needs to optimize the resource allocation by identifying the potential customers. As iFood lack of interpreted meaningful information about customer’s background, resources were allocated inappropriately, consumed more time and energy for the non-respondents. This has led to a high cost, but low revenue generated from the campaign. Besides, this may also negatively impact iFood’s customer’s experience as they may found that the product promoted is not advantageous to them, hence increase in the number of complaints.

In order to overcome the problem, a dataset is collected from the past campaigns for data analysis, which contains the information about customer’s profile, product preferences, campaign outcomes, and channel performance metrics. By discovering actionable insights hidden in the data, a predictive model is developed to forecast customer’s purchasing behaviour. With this meaningful information, iFood can narrow down the targeted customers effectively, reduce the cost and optimize the future campaign performance, resulting in a highly profitable marketing strategy and sustainable growth in the competitive online food delivery industry.

## **1.2 Project Objectives**

The goal of this project is to develop a predictive model that allows the user to forecast customer’s behaviour in purchasing the products, based on their background, such as annual household income, level of education, purchasing preferences, etc. The objectives of the project can be summarized as follows:  

i) To identify the relationship between customer’s background and purchasing behaviour

ii) To implement customer segmentation and targeting by incoporating membership-based segmentation.

iii)	To develop a predictive model to estimate customer’s intention to purchase the product

# **2.0 Methodology**

## **2.1 Data Loading**
"""

import pandas as pd
import numpy as np
import regex as re
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy import stats

df = pd.read_excel('marketing_campaign.xlsx')
df.head(100)

# Check data type
df.info()

"""## **2.2 Scrub**"""

# Check missing values for each columns
print(df.isnull().values.any())
print(df.isnull().sum())

# Visualize missing values using a matrix plot
msno.matrix(df)
plt.show()

# Drop missing value rows from Income
df = df.dropna(subset=['Income'])
print(df.isnull().values.any())
print(df.isnull().sum())

# Check unique values for selected columns
for feature in ['Education', 'Marital_Status']:
    feature_unique = df[feature].unique()
    print(feature, len(feature_unique), 'unique values are:', feature_unique)

# Plotting histograms for each categorical feature
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # Adjust size as needed

for index, feature in enumerate(['Education', 'Marital_Status']):
    df[feature].value_counts().plot(kind='bar', ax=axes[index], title=feature)
    axes[index].set_ylabel('Count')
    axes[index].set_xlabel(feature)

plt.tight_layout()
plt.show()

# Update 'Education' column: Group '2n Cycle' and 'Basic' into 'Others'
df['Education'] = df['Education'].replace(['2n Cycle', 'Basic'], 'Others')

# Update 'Marital_Status' column: Change 'Alone' to 'Single', and 'Absurd', 'YOLO' to 'Others'
df['Marital_Status'] = df['Marital_Status'].replace({'Alone': 'Single', 'Absurd': 'Others', 'YOLO': 'Others'})

# Print the unique values of these columns
print("Unique values in 'Education':", df['Education'].unique())
print("Unique values in 'Marital_Status':", df['Marital_Status'].unique())

# Plotting histograms for each categorical feature
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # Adjust size as needed

for index, feature in enumerate(['Education', 'Marital_Status']):
    df[feature].value_counts().plot(kind='bar', ax=axes[index], title=feature)
    axes[index].set_ylabel('Count')
    axes[index].set_xlabel(feature)

plt.tight_layout()
plt.show()

# Check the number of duplicates
duplicate_count = df.duplicated().sum()
print("Number of duplicates:", duplicate_count)

numeric_features = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
                    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                    'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                    'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
                    'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response']

# Creating a 5x5 grid of box plots
plt.figure(figsize=(20, 20))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(5, 5, i)  # 5x5 grid, position i
    sns.boxplot(y=df[feature])
    plt.title(feature)
    plt.tight_layout()

plt.show()

# Set the threshold for defining an outlier
threshold = 3

# List of columns to assess for outliers
columns = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
           'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
           'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
           'NumWebVisitsMonth']


# Calculate the Z-scores for each column and collecting indices of outliers
outlier_indices = []

for column in columns:
    z_scores = np.abs(stats.zscore(df[column]))
    outliers = np.where(z_scores > threshold)
    outlier_indices.extend(outliers[0])

# Remove duplicate indices
outlier_indices = list(set(outlier_indices))

print("Indices of outliers:", outlier_indices)

# Remove the outliers from the DataFrame
df = df.drop(index=outlier_indices)
print("Data after removing outliers has", df.shape[0], "rows.")

numeric_features = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
                    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                    'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                    'NumWebVisitsMonth']

# Creating a 5x5 grid of box plots
plt.figure(figsize=(20, 20))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(5, 5, i)  # 5x5 grid, position i
    sns.boxplot(y=df[feature])
    plt.title(feature)
    plt.tight_layout()

plt.show()

# Create a new DataFrame by dropping the 'Dt_Customer','Education', 'Marital_Status', 'ID' column
new_df = df.drop(columns=['Dt_Customer','Education', 'Marital_Status','ID'])

# Calculate the correlation matrix for the new DataFrame
correlation_matrix_new = new_df.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(16, 12))  # Adjust the size for better readability
sns.heatmap(correlation_matrix_new, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of Features in new_df')
plt.show()

# Left trim all dataset
df.columns = df.columns.str.replace(' ', '')

# Ensure the dt_Customer column are in datetime format
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df.head(100)

# Fill empty income to 0
df['Income'].fillna(0, inplace=True)
df.head(100)

# Change the date customer to date format to ensure data format
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

# Calculate the difference in years between 2024 and the year of each date
df['Age']=2024-df.Year_Birth
df['Years_joining_as_customer'] = 2024 - df['Dt_Customer'].dt.year
df['Age_joining_as_customer'] = df['Age']-df['Years_joining_as_customer']
df.head(100)

# Rename the data column to more understandable name
df = df.rename(columns={
       'Kidhome' : 'KidHome',
       'Teenhome' : 'TeenHome',
       'Dt_customer' : 'Dt_Customer',
       'Recency' : 'Recency',
        'MntWines' :'WineSales' ,
       'MntFruits' :  'FruitSales',
       'MntMeatProducts' : 'MeatSales',
       'MntFishProducts' : 'FishSales',
       'MntSweetProducts' : 'SweetSales',
       'MntGoldProds' : 'GoldSales',
       'NumDealsPurchases' : 'DealPurchases',
       'NumWebPurchases' : 'WebPurchases',
       'NumCatalogPurchases' : 'CatalogPurchases',
       'NumStorePurchases' : 'StorePurchases',
       'NumWebVisitsMonth' : 'WebVisitsMonth',
       'AcceptedCmp3' : 'Cmp3',
       'AcceptedCmp4' : 'Cmp4',
       'AcceptedCmp5' : 'Cmp5',
       'AcceptedCmp1' : 'Cmp1',
       'AcceptedCmp2' : 'Cmp2',
       'Response' : 'Cmp6',
       'Complain' : 'Complaints',
})

# Rearrange the column in dataframe
df = df[['ID', 'Age','Years_joining_as_customer','Age_joining_as_customer','Recency', 'Income', 'KidHome', 'TeenHome', 'Complaints',
        'Cmp1', 'Cmp2', 'Cmp3', 'Cmp4', 'Cmp5', 'Cmp6', 'Education', 'Marital_Status',
        'Dt_Customer', 'WineSales', 'FruitSales', 'MeatSales', 'FishSales', 'SweetSales',
        'GoldSales', 'CatalogPurchases', 'DealPurchases', 'StorePurchases', 'WebPurchases', 'WebVisitsMonth']]

"""## **2.3 Explore - Data Analysis**

"""

# Sum up deal purchases, web purchases , catalog purchase and store purchase for each consumer to know their total purchases
df['Total_purchases'] = df[['DealPurchases',
       'WebPurchases',
       'CatalogPurchases',
       'StorePurchases', ]].sum(axis=1)

# Sum up wine sales, FruitSales , MeatSales , FishSales and SweetSales for each consumer to know their total purchases

df['Sales'] = df[['WineSales', 'FruitSales', 'MeatSales', 'FishSales', 'SweetSales']].sum(axis=1)
df.head(100)

# Assume the date collection is 2024.3.12
# Recency day assume from 2024.3.12
from datetime import datetime, timedelta
today_date = datetime.today()
df['Last_Visit_Date'] = today_date - pd.to_timedelta(df['Recency'], unit='D')
df.head(100)

# Count on mean for years_joining_as_Customer, age, income, recency
meandf = df[['Age', 'Income','Recency','Years_joining_as_customer']] \
            .mean().sort_values().to_frame(name = 'Mean')
display(meandf)

# Corelation between total purchase and income, the coefficient is 55% related to each others
correlation = df['Total_purchases'].corr(df['Income'])

print("Correlation coefficient between Total Purchase and Income:", correlation)

# Understand the relationship between income and total purchases using a scatter plot.
plt.figure(figsize=(10, 6))
plt.scatter(df['Income'], df['Total_purchases'], alpha=0.5, color='blue')
plt.title('Total Purchase vs Income')
plt.xlabel('Income')
plt.ylabel('Total Purchase')
plt.grid(True)
plt.show()

filtered_df = df[df['Cmp1'] == 1]

# Sum up the values of specified columns
total_sales = filtered_df[['WineSales', 'FruitSales', 'MeatSales', 'FishSales', 'SweetSales']].sum()

print(total_sales)

columns_to_sum = ['WineSales', 'FruitSales', 'MeatSales', 'FishSales', 'SweetSales']

# Group by 'Cmp1' through 'Cmp5' and sum the sales for each category
total_sales_by_cmp = filtered_df.groupby(['Cmp1', 'Cmp2', 'Cmp3', 'Cmp4', 'Cmp5'])[columns_to_sum].sum()

def highlight_top(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

# Apply the highlighting function to each column of the DataFrame
styled_total_sales_by_cmp = total_sales_by_cmp.style.apply(highlight_top, axis=0)

# Display the styled dataframe
styled_total_sales_by_cmp

# Group last visit date and count on the number of complaint in a day
summary_df = df.groupby('Last_Visit_Date')['Complaints'].sum().reset_index()
summary_df.columns = ['Last Visit Date', 'Total Number of Complaints']
pd.set_option('display.max_rows', None)
summary_df

# Title Total Number of Complaints

from matplotlib import pyplot as plt
summary_df['Total Number of Complaints'].plot(kind='hist', bins=20, title='Total Number of Complaints')
plt.gca().spines[['top', 'right',]].set_visible(False)

# List out all historical complaint customer
complaint_customer = df[df['Complaints'] > 0]
complaint_customer

# Statistics summary for complaint customer on total purchases and sales

num_rows_complaint_customer = complaint_customer.shape[0]
sum_total_purchases = complaint_customer['Total_purchases'].sum()
mean_total_purchases = complaint_customer['Total_purchases'].mean()
median_total_purchases = complaint_customer['Total_purchases'].median()
sum_sales = complaint_customer['Sales'].sum()
mean_sales = complaint_customer['Sales'].mean()
median_sales = complaint_customer['Sales'].median()
print("\nSummary Statistics for Total Purchases for Complaint Customer:")
print("Number of complaints:",len(complaint_customer))
print("Sum of Total purchase:", sum_total_purchases)
print("Mean of Total purchase:", mean_total_purchases)
print("Median of Total purchase:", median_total_purchases)

print("Sum of Total sales:", sum_sales)
print("Mean of Total sales:", mean_sales)
print("Median of Total sales:", median_sales)

# Split the sum of sales by complaint and non complaint customer and to see the trend of sales of complaint vs non compliant customer
complaint_df = df[df['Complaints'] > 0]
non_complaint_df = df[df['Complaints'] == 0]
complaint_sales_by_date = complaint_df.groupby('Last_Visit_Date')['Sales'].sum()
non_complaint_sales_by_date = non_complaint_df.groupby('Last_Visit_Date')['Sales'].sum()
plt.figure(figsize=(10, 6))
plt.plot(complaint_sales_by_date.index, complaint_sales_by_date.values, label='Complaint Customers', marker='o')
plt.plot(non_complaint_sales_by_date.index, non_complaint_sales_by_date.values, label='Non-Complaint Customers', marker='o')
plt.title('Last Visit Date vs. Sum of Sales')
plt.xlabel('Last Visit Date')
plt.ylabel('Sum of Sales')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting a histogram for the 'Income' column
df['Income'].hist(bins=100)
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

# Graphical representation of the distribution of data based on the five-number summary: minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum.
df.boxplot(column=['Income'])
plt.title('Income Box Plot')
plt.ylabel('Income')
plt.show()

df.boxplot(column=['Age'])
plt.title('Age Box Plot')
plt.ylabel('Age')
plt.show()

# Define age bins and labels for the age groups
bins = [0, 18, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['<18', '18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']

# Create 'Age Group' column
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Calculate the number of responses for each age group
response_counts = df.groupby(['Age Group','Cmp6']).size().unstack(fill_value=0)

# Plotting the bar chart
ax= response_counts.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'lightgreen'])

plt.title('Number of Responses by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Responses')
plt.xticks(rotation=45)

# Annotate frequency on top of each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()

"""The 50-59 age group has the highest number of responses, both non-responders and positive responders, indicating strong engagement with the campaign. The highest overall engagement is seen in the middle-aged groups (40-69)."""

# Group by 'Education' and 'Cmp6', then count instances
response_counts = df.groupby(['Education', 'Cmp6']).size().unstack(fill_value=0)

# Plotting the bar chart
ax = response_counts.plot(kind='bar', figsize=(10, 6), width=0.8, color=['skyblue', 'lightgreen'])

plt.title('Number of Cmp6 Responses (0 and 1) by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Count of Responses')
plt.xticks(rotation=45)  # Rotate labels for better readability

# Annotate frequency on top of each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()

"""The "Graduation" category has the highest number of responses in both categories, indicating that graduates make up the largest portion of the dataset."""

# Group by 'Marital_Status' and 'Cmp6', then count instances
response_counts = df.groupby(['Marital_Status', 'Cmp6']).size().unstack(fill_value=0)

# Plotting the bar chart
ax = response_counts.plot(kind='bar', figsize=(10, 6), width=0.8, color=['skyblue', 'lightgreen'])

plt.title('Number of Cmp6 Responses (0 and 1) by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count of Responses')
plt.xticks(rotation=45)

# Annotate frequency on top of each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()

"""Single individuals constitute the largest group of respondents, followed by those who are married. Married groups have a higher number of non-responders as well."""

# Define income bins and labels, including the >100000 group
income_bins = [0, 20000, 40000, 60000, 80000, 100000, float('inf')]
income_labels = ['0-20k', '20k-40k', '40k-60k', '60k-80k', '80k-100k', '>100k']

# Create 'Income Range' column
df['Income Range'] = pd.cut(df['Income'], bins=income_bins, labels=income_labels, right=False)

# Group by 'Income Range' and 'Cmp6', then count instances
response_counts = df.groupby(['Income Range', 'Cmp6']).size().unstack(fill_value=0)

# Plotting the bar chart
ax = response_counts.plot(kind='bar', figsize=(10, 6), width=0.8, color=['skyblue', 'lightgreen'])

plt.title('Number of Cmp6 Responses (0 and 1) by Income Range')
plt.xlabel('Income Range')
plt.ylabel('Count of Responses')
plt.xticks(rotation=45)

# Annotate frequency on top of each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()

"""Individuals in higher income ranges (60K and above) show a higher tendency to respond positively to the campaign, indicating that income may influence engagement levels."""

# Group by 'Kidhome' and get counts for each category of 'Cmp6'
response_counts = df.groupby(['KidHome', 'Cmp6']).size().unstack(fill_value=0)

# Plotting the bar chart
ax = response_counts.plot(kind='bar', figsize=(10, 6), width=0.8,color=['skyblue', 'lightgreen'])

plt.title('Number of Cmp6 Responses (0 and 1) by Number of Children at Home')
plt.xlabel('Number of Children at Home')
plt.ylabel('Count of Responses')
plt.xticks(rotation=0)

# Annotate frequency on top of each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()

"""Families with no children at home show the highest engagement levels, followed by those with one child. Families with two children exhibit the lowest engagement."""

# Group by 'TeenHome' and get counts for each category of 'Cmp6'
response_counts = df.groupby(['TeenHome', 'Cmp6']).size().unstack(fill_value=0)

# Plotting the bar chart
ax = response_counts.plot(kind='bar', figsize=(10, 6), width=0.8, color=['skyblue', 'lightgreen'])

plt.title('Number of Cmp6 Responses (0 and 1) by Number of Teenagers at Home')
plt.xlabel('Number of Teenagers at Home')
plt.ylabel('Count of Responses')
plt.xticks(rotation=0)

# Annotate frequency on top of each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()

"""Families with no teenagers at home show the highest engagement levels, followed by those with one teenager. Families with two teenagers exhibit the lowest engagement.

##**2.4 Clustering Model**

###RFM Segmentation

RFM segmentation is based on customer behavior such as spending, comsumption habit, product/service usage and previously purchased product.

* Recency (R): The last time customer bought a product.
* Frequency (F): Frequency of customer making a purchase.
* Monetary Value(M): The total value of expenditure from customer.
"""

# Create 2 additional variables for F and M
# Frequency= the total number of times a customer purchases a product.
# Monetary = the total of customer expenditure.
# because there is no price on the dataset, we will calculate from the summation of the product sales without multiplying the product price

df_clustering=df.copy()
df_clustering['Frequency']=df_clustering['DealPurchases']+df_clustering['WebPurchases']+df_clustering['CatalogPurchases']+df_clustering['StorePurchases']
df_clustering['Monetary']=df_clustering['WineSales']+df_clustering['FruitSales']+df_clustering['MeatSales']+df_clustering['FishSales']+df_clustering['SweetSales']+df_clustering['GoldSales']
df_clustering.head(10)

# Recency
recency = df_clustering[['ID','Recency']]
recency.head(10)

sns.histplot(df_clustering['Recency'], kde=True)
plt.title('Recency')

"""The distribution of the recency of the customer is close to uniform distribution. This means that the distribution is fairly spread from the customers that recently did a transaction to the customers that did the transaction almost 100 days ago."""

# Frequency
frequency = df_clustering[['ID','Frequency']]
frequency.head(10)

sns.histplot(df_clustering['Frequency'], kde=True)
plt.title('Frequency')

"""The transaction frequency of each customer is between 1 transaction up to around 44 transactions. With most of them doing around 4 to 10 transactions followed by 16 to 24 transactions."""

# Monetary Value
monetary = df_clustering[['ID','Monetary']]
monetary.head(10)

sns.histplot(df_clustering['Monetary'], kde=True)
plt.title('Monetary')

"""Most customers spend around $200."""

temp = recency.merge(frequency,on='ID')
RFM_Segmentation  = temp.merge(monetary,on='ID')
RFM_Segmentation.columns = ['ID','Recency','Frequency','Monetary']
RFM_Segmentation.head(10)

"""###Data Processing
* Standardising data
* Principal Component Analysis (PCA)

"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist,pdist

# Standardising Data
scaler = StandardScaler()
cols_for_clustering = ['Recency', 'Frequency', 'Monetary']
data_scaled = df_clustering.copy()
data_scaled[cols_for_clustering] = scaler.fit_transform(df_clustering[cols_for_clustering])
data_scaled[cols_for_clustering].describe()

"""The mean value for all columns is almost zero and the standard deviation is almost 1. All the data points were replaced by their z-scores."""

# Principal Component Analysis (PCA)

pca = decomposition.PCA(n_components = 3)
pca_res = pca.fit_transform(data_scaled[cols_for_clustering])
data_scaled['pc1'] = pca_res[:,0]
data_scaled['pc2'] = pca_res[:,1]
data_scaled['pc3'] = pca_res[:,2]
data_scaled.head()

"""###K-means Clustering

K-means clustering is an unsupervised machine learning algorithm used to cluster data based on similarity. K-means clustering usually works well in practice and scales well to the large datasets.
"""

# Elbow Method
X = data_scaled[cols_for_clustering]
inertia_list = []
for K in range(2,10):
    inertia = KMeans(n_clusters=K,n_init=10, random_state=7).fit(X).inertia_
    inertia_list.append(inertia)
plt.figure(figsize=[7,5])
plt.plot(range(2,10), inertia_list, color=(54 / 255, 113 / 255, 130 / 255))
plt.title("Inertia vs. Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()

"""The "elbow" shows that the optimal cluster is around 4 clusters as it provides a good balance between reducing inertia and avoiding overfitting."""

# Silhouette score analysis
silhouette_list = []
for K in range(2,10):
    model = KMeans(n_clusters = K, n_init=10, random_state=7)
    clusters = model.fit_predict(X)
    s_avg = silhouette_score(X, clusters)
    silhouette_list.append(s_avg)

plt.figure(figsize=[7,5])
plt.plot(range(2,10), silhouette_list, color=(54 / 255, 113 / 255, 130 / 255))
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()

"""The Silhouette Method suggests that 2 clusters provide the best separation and cohesion for the dataset. A higher silhouette score indicates that the clusters are well-separated, while a lower score suggests that the clusters may overlap or are poorly defined.

We chose 4 clusters as the best option for our dataset. While the Silhouette Score peaks at k=2, indicating better-defined clusters, the Elbow Method suggests an optimal point at k=4. Opting for 4 clusters provides greater flexibility for future adjustments. If our data or business requirements evolve, having more clusters allows us to better capture the diversity in customer behavior, making our segmentation more adaptable to changing needs.
"""

model = KMeans(n_clusters=4, n_init=10, random_state = 7)
model.fit(data_scaled[cols_for_clustering])
# cluster start with 1
data_scaled['k Cluster'] = model.predict(data_scaled[cols_for_clustering]) + 1
# Adding variables to df_clustering
df_clustering['k Means Cluster'] = data_scaled['k Cluster']
df_clustering["pc1"]=data_scaled['pc1']
df_clustering["pc2"]=data_scaled['pc2']
df_clustering["pc3"]=data_scaled['pc3']
df_clustering.head()

"""###Gaussian Mixture Model (GMM)

Gaussian Mixture Model (GMM) clustering is a probabilistic approach that assumes data points are generated from a mixture of Gaussian distributions using the Expectation-Maximization algorithm to assign points to clusters based on likelihood.
"""

from sklearn.mixture import GaussianMixture

# Fit GMM with a range of clusters to find the best one
gmm_scores = []
for K in range(2, 10):
    gmm = GaussianMixture(n_components=K, random_state=7)
    gmm.fit(X)
    labels = gmm.predict(X)
    gmm_scores.append(silhouette_score(X, labels))

# Range of cluster numbers to test
k_range = range(2, 10)
log_likelihoods = []

# Fit GMM for different numbers of clusters and store the log-likelihood
for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)
    log_likelihoods.append(gmm.score(X))  # The log-likelihood of the model

# Plot log-likelihood vs number of clusters (elbow plot)
plt.figure(figsize=(8, 6))
plt.plot(k_range, log_likelihoods, marker='o')
plt.title('Elbow Method for GMM')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Log-Likelihood')
plt.grid(True)
plt.show()

"""Based on the Elbow Method plot, k=4 is a suitable choice as the log-likelihood shows a notable increase up to this point, after which the improvement slows. This suggests that four clusters capture the main structure in the data effectively."""

# Plot silhouette scores for GMM
plt.figure(figsize=[7,5])
plt.plot(range(2, 10), gmm_scores, color=(54 / 255, 113 / 255, 130 / 255))
plt.title("Silhouette Score vs. Number of Components (GMM)")
plt.xlabel("Number of Components")
plt.ylabel("Silhouette Score")
plt.show()

"""The highest Silhouette Score is at k=2, indicating that 2 components (or clusters) would be optimal.

We chose 4 clusters based on the Elbow Method, which suggests this number captures the main structure in the data. Although the Silhouette Method recommended 2 clusters, using 4 allows us to see finer differences between customer groups. This extra detail helps us tailor marketing strategies to meet the unique needs of each segment more effectively.
"""

# Set optimal number of components based on silhouette score or BIC
gmm = GaussianMixture(n_components=4, random_state=7)
gmm.fit(X)
df_clustering['GMM Cluster'] = gmm.predict(X) + 1  # Start cluster labels from 1
df_clustering.head()

"""###Hierarchical Clustering

Hierarchical clustering builds a tree-like structure of nested clusters by either merging smaller clusters (agglomerative) or splitting larger ones (divisive) based on similarity measures.
"""

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate linkage matrix for the dendrogram
linkage_matrix = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer Index")
plt.ylabel("Distance")
plt.show()

"""Based on the dendrogram, the data can be grouped into three main clusters. The height at which the clusters are joined indicates their similarity with the three main branches representing distinct groups. Cutting the dendrogram at a height of around 40 separates the data into three clear clusters. This suggests that the data naturally divides into three major groups."""

# Fit Hierarchical Clustering with a range of clusters and calculate silhouette scores
hierarchical_scores = []
for K in range(2, 10):
    hc = AgglomerativeClustering(n_clusters=K, linkage='ward')
    labels = hc.fit_predict(X)

    # Calculate the silhouette score
    score = silhouette_score(X, labels)
    hierarchical_scores.append(score)

# Plot silhouette scores for Hierarchical Clustering
plt.figure(figsize=[7, 5])
plt.plot(range(2, 10), hierarchical_scores, color=(133 / 255, 89 / 255, 103 / 255))
plt.title("Silhouette Score vs. Number of Clusters (Hierarchical Clustering)")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

"""The Silhouette Method suggests that 2 clusters is the optimal choice for the data as it gives the highest silhouette score of 0.35. This indicates a moderate level of separation between the clusters suggesting that two clusters is a reasonable and effective choice for grouping the data.

Although the Silhouette Method suggests 2 clusters with a moderate score of 0.35, we chose 3 clusters based on the dendrogram which clearly shows a natural division at a reasonable cutting height. While the silhouette score indicates some overlap in features, the dendrogram provides a stronger visual representation of how the data naturally groups into 3 distinct clusters. This choice enhances customer segmentation and targeting allowing for more refined and tailored marketing strategies.
"""

# Fit Agglomerative Clustering with optimal clusters
hierarchical_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
df_clustering['Hierarchical Cluster'] = hierarchical_model.fit_predict(X) + 1
df_clustering.head()

"""###Compare Model Performance

In order to choose the best clustering model, we compare the three models in terms of:
* Silhouette scores
* Average RFM values
* 3D plot cluster analysis
* Cluster distribution
* Cross-tabulation
"""

# Silhouette Scores for each method
kmeans_silhouette = silhouette_score(X, df_clustering['k Means Cluster'] - 1)  # Adjust for zero-based indexing
gmm_silhouette = silhouette_score(X, df_clustering['GMM Cluster'] - 1)
hierarchical_silhouette = silhouette_score(X, df_clustering['Hierarchical Cluster'] - 1)

print("Silhouette Scores:")
print(f"K-means: {kmeans_silhouette}")
print(f"GMM: {gmm_silhouette}")
print(f"Hierarchical: {hierarchical_silhouette}")

"""The K-means clustering model achieved the highest silhouette score among the models evaluated indicating well-separated and cohesive clusters. It shows that K-means effectively captures the structure of the data and outperforms other clustering methods in terms of cluster quality."""

# Average R, F, M values for each model

# Calculate average R, F, M values for each K-means cluster
kmeans_avg_rfm = df_clustering.groupby('k Means Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

# Calculate average R, F, M values for GMM clusters
gmm_avg_rfm = df_clustering.groupby('GMM Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

# Calculate average R, F, M values for Hierarchical clusters
hierarchical_avg_rfm = df_clustering.groupby('Hierarchical Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

print("\nAverage RFM values per cluster for each model:")
print("\nK-means Clusters:\n", kmeans_avg_rfm)
print("\nGMM Clusters:\n", gmm_avg_rfm)
print("\nHierarchical Clusters:\n", hierarchical_avg_rfm)

"""**K-means Clusters**  
- Cluster 1 have **recent with high frequency and monetary values** to indicate loyal and high-value customers.  
-Cluster 4 have **high frequency and monetary values** but lower recency to indicate high-value customers.  
- Cluster 2 shows **recent but low-value customers** with low frequency and monetary values but high recency.
- Cluster 3 represents **low-value customers** with low frequency and monetary.  
-k-means clustering offers clear differentiation particularly between high and low-value customers.

---

**GMM Clusters**  
- Cluster 1 captures **very frequent, high-value customers** with low recency.
- Cluster 2 consists of **very low-value customers** with extremely low frequency and monetary values.  
- Cluster 3 indicates **moderate-value customers** with low monetary and frequency values.
- Cluster 4 represents **high-value but slightly less frequent customers** with moderate recency.  
- GMM shows well-separated clusters but has overlapping segments compared to K-means.

---

**Hierarchical Clusters**  
- Cluster 1 includes **moderate-value customers**.
- Cluster 2 captures **frequent, very high-value customers**.   
- Cluster 3 represents **low-value customers** with lower frequency and monetary values but higher recency.  
- Hierarchical clustering provides clear segmentation but has broader clusters compared to K-means and GMM.

---

  
**K-means** is the best cluster choice as it provides distinct and interpretable clusters with clear differentiation between high-value, low-value, and recent customers. It also aligns well with the objective of identifying actionable customer segments for targeting.
"""

#3D cluster plot analysis
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

Cluster = df_clustering[['k Means Cluster', 'pc1', 'pc2','pc3']]
# Scatter plot
fig = px.scatter_3d(Cluster, x='pc1', y='pc2', z='pc3', color=Cluster['k Means Cluster'],
                   labels={'0':'pc1','1':'pc2','2':'pc3'})
fig.update_layout(title='k-Means Clustering Analysis')
fig.show()

# Set up cluster data with PCA components for visualization consistency
Cluster_gmm = df_clustering[['GMM Cluster', 'pc1', 'pc2', 'pc3']]
Cluster_hierarchical = df_clustering[['Hierarchical Cluster', 'pc1', 'pc2', 'pc3']]

# 3D Scatter Plot for GMM Clustering
fig_gmm = px.scatter_3d(Cluster_gmm, x='pc1', y='pc2', z='pc3', color=Cluster_gmm['GMM Cluster'],
                        labels={'0':'pc1','1':'pc2','2':'pc3'})
fig_gmm.update_layout(title='GMM Clustering Analysis')
fig_gmm.show()

# 3D Scatter Plot for Hierarchical Clustering
fig_hierarchical = px.scatter_3d(Cluster_hierarchical, x='pc1', y='pc2', z='pc3', color=Cluster_hierarchical['Hierarchical Cluster'],
                                 labels={'0':'pc1','1':'pc2','2':'pc3'})
fig_hierarchical.update_layout(title='Hierarchical Clustering Analysis')
fig_hierarchical.show()

""" **K-means Clustering:**  
- The clusters appear compact and well-separated.  
- Each cluster is clearly distinguishable which aligns with the high silhouette score for K-means.  
- The points within clusters show good cohesion meaning that K-means performs well for this dataset.  

---

**GMM Clustering:**  
- GMM clustering produces distinguishable clusters but has some overlapping between cluster boundaries.  
- The Gaussian-based approach allows flexibility but some clusters like the yellow and blue seem less distinct compared to K-means.  
- This suggests that GMM captures variability in the data but may struggle with tighter boundaries.  

---

**Hierarchical Clustering:**  
- Hierarchical clustering shows broader and overlapping clusters.  
- The pink cluster spread over a larger area indicating that it has weaker cohesion.  
- The separation between clusters is less clear which might suggest this method might not be ideal for high-dimensional RFM data.  

---
The K-means model provides the clearest and most compact clusters. GMM is a reasonable alternative but hierarchical clustering lacks the clarity and separation needed for actionable segmentation.
"""

# Cluster distribution
# Distribution of k-means cluster
df_clustering['k Means Cluster'].value_counts().plot.bar()
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('k-Means Cluster Distribution')
plt.show()

# Distribution of clusters for GMM
df_clustering['GMM Cluster'].value_counts().plot.bar()
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('GMM Cluster Distribution')
plt.show()

# Distribution of clusters for Hierarchical Clustering
df_clustering['Hierarchical Cluster'].value_counts().plot.bar()
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Hierarchical Cluster Distribution')
plt.show()

"""**k-Means Cluster Distribution**:
  
   - The distribution is relatively balanced but Cluster 4 has a slightly smaller count than the others. This shows the equal-partitioning tendency of k-Means when distributing data into clusters.

---

**GMM Cluster Distribution**:
   - GMM shows more variation in cluster sizes compared to k-Means. This is because GMM accommodates cluster shapes and sizes better which leads to an uneven distribution based on probabilities.

---

**Hierarchical Cluster Distribution**:
   - The hierarchical clustering distribution is more balanced compared to GMM but less balanced than k-Means. Clusters 3 and 1 are dominant, while Cluster 2 is smaller.

---

k-Means shows the most balanced distribution, followed by Hierarchical while GMM showing the most variability.

"""

# Cross-tabulation
# Cross-tabulation between K-means and GMM clusters
kmeans_gmm_crosstab = pd.crosstab(df_clustering['k Means Cluster'], df_clustering['GMM Cluster'])
plt.figure(figsize=(8, 6))
# Heatmap for K-means vs GMM
sns.heatmap(kmeans_gmm_crosstab, annot=True, cmap="YlGnBu", fmt="d")
plt.title("K-means vs GMM Cluster Overlap")
plt.xlabel("GMM Cluster")
plt.ylabel("K-means Cluster")
plt.show()


# Cross-tabulation between K-means and Hierarchical clusters
kmeans_hierarchical_crosstab = pd.crosstab(df_clustering['k Means Cluster'], df_clustering['Hierarchical Cluster'])
plt.figure(figsize=(8, 6))
# Heatmap for K-means vs Hierarchical
sns.heatmap(kmeans_hierarchical_crosstab, annot=True, cmap="YlGnBu", fmt="d")
plt.title("K-means vs Hierarchical Cluster Overlap")
plt.xlabel("Hierarchical Cluster")
plt.ylabel("K-means Cluster")
plt.show()


# Cross-tabulation between GMM and Hierarchical clusters
gmm_hierarchical_crosstab = pd.crosstab(df_clustering['GMM Cluster'], df_clustering['Hierarchical Cluster'])
plt.figure(figsize=(8, 6))
# Heatmap for GMM vs Hierarchical
sns.heatmap(gmm_hierarchical_crosstab, annot=True, cmap="YlGnBu", fmt="d")
plt.title("GMM vs Hierarchical Cluster Overlap")
plt.xlabel("Hierarchical Cluster")
plt.ylabel("GMM Cluster")
plt.show()

"""Cross-tabulation between clustering models helps evaluate the consistency of cluster assignments by comparing overlaps by providing insights into the stability and interpretability of clusters across the models. This helps in selecting the best model by identifying the one that aligns well with our objectives to produce meaningful segmentation.

**K-means vs. GMM**
- K-means cluster 1 overlaps largely with GMM cluster 4 (435 points).
- K-means cluster 2 corresponds strongly with GMM cluster 2 (304 points).
-There are off-diagonal overlaps. K-means cluster 2 also overlaps significantly with GMM cluster 3 (210 points) indicating  mixed alignment.
- While there is some consistency the mapping between K-means and GMM is not perfectly aligned.

---
**K-means vs. Hierarchical Clustering**
- K-means cluster 2 aligns almost entirely with Hierarchical cluster 2 (518 points).
- K-means cluster 3 overlaps strongly with Hierarchical cluster 1 (435 points).
- However, K-means cluster 4 overlaps with both Hierarchical cluster 1 (231 points) and cluster 2 (199 points). This indicates some off-diagonal discrepancies.
- K-means and Hierarchical clustering are relatively more consistent with each other as it has a stronger alignment compared to k-means vs GMM .

---

**GMM vs. Hierarchical Clustering**
- GMM cluster 4 overlaps strongly with Hierarchical cluster 2 (419 points).
- GMM cluster 3 aligns largely with Hierarchical cluster 3 (286 points).
- However, GMM cluster 1 is split between Hierarchical clusters 1 (189 points) and 2 (169 points) indicating mixed alignment.
- GMM and Hierarchical clustering show a reasonable level of consistency but the alignment is not as strong as between K-means and Hierarchical clustering.

---

- K-means vs. Hierarchical Clustering shows the strongest alignment as the clusters mapping are  more cleanly to one another compared to the other two pairings.
-The alignment between these two models shows minimal off-diagonal discrepancies meaning their clusters represent similar groupings of customers.
- **K-means or hierarchical clustering appears to provide the most effective customer segmentation.**

#### **The Best Clustering Model**

**K-Means clustering** is the best model as it has  the highest silhouette score (0.3774) with clearly distinguishable, compact, and well-separated clusters along with a balanced cluster distribution. Additionally, the average RFM values highlight clear differentiation between clusters and cross-tabulation confirms that K-Means delivers effective and meaningful segmentation.

## **2.5 Predictive Model**
"""

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve

df_predictive = df[['Age', 'Years_joining_as_customer','Recency', 'Income', 'KidHome', 'TeenHome', 'Complaints',
                    'Cmp1', 'Cmp2', 'Cmp3', 'Cmp4', 'Cmp5', 'Cmp6', 'Education', 'Marital_Status',
                    'WineSales', 'FruitSales', 'MeatSales', 'FishSales', 'SweetSales', 'GoldSales',
                    'CatalogPurchases', 'DealPurchases', 'StorePurchases', 'WebPurchases', 'WebVisitsMonth',
                    'Total_purchases', 'Sales']].copy()

print(df_predictive.dtypes)

# Create binary column for categorical data (Education and Marital_Status)
encoded_education = pd.get_dummies(df_predictive['Education'], prefix='', prefix_sep='')
encoded_marital_status = pd.get_dummies(df_predictive['Marital_Status'], prefix='', prefix_sep='')

# Rename the columns
encoded_education.columns = ['Education_Graduation', 'Education_PhD', 'Education_Master', 'Education_Others']
encoded_marital_status.columns = ['Marital_Status_Single', 'Marital_Status_Together', 'Marital_Status_Married', 'Marital_Status_Divorced', 'Marital_Status_Widow', 'Marital_Status_Others']

# Concatenate the one-hot encoded columns with the original DataFrame
df_predictive = pd.concat([df_predictive, encoded_education, encoded_marital_status], axis=1)

# Drop the original 'Education' and 'Marital_Status' columns
df_predictive.drop(['Education', 'Marital_Status'], axis=1, inplace=True)

print(df_predictive.dtypes)

X = df_predictive.drop(columns='Cmp6')
y = df_predictive['Cmp6']

print(y.value_counts())

# Apply SelectKBest for feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected features
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {selected_features}")

print("Original number of features:", X.shape[1])
print("Reduced number of features:", X_selected.shape[1])

scores = selector.scores_
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores,})
feature_scores = feature_scores.sort_values(by='Score', ascending=False)
print(feature_scores)

# Split the dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)

# Create SMOTE object
smote = SMOTE(random_state=42)

# Use SMOTE to perform oversampling to training dataset
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(y_train_res.value_counts())

# Convert X_train_res back to a DataFrame
X_train_res = pd.DataFrame(X_train_res, columns=selected_features)
X_test = pd.DataFrame(X_test, columns=selected_features)

# Scaling
scaler = StandardScaler()
X_train_res[['Years_joining_as_customer', 'Recency',
             'WineSales', 'MeatSales', 'CatalogPurchases', 'Sales']] = scaler.fit_transform(
           X_train_res[['Years_joining_as_customer', 'Recency',
                        'WineSales', 'MeatSales', 'CatalogPurchases', 'Sales']])

X_test[['Years_joining_as_customer', 'Recency',
        'WineSales', 'MeatSales', 'CatalogPurchases', 'Sales']] = scaler.transform(
    X_test[['Years_joining_as_customer', 'Recency',
            'WineSales', 'MeatSales', 'CatalogPurchases', 'Sales']])

# Output the size of the datasets after splitting
print("Training set feature size:", X_train_res.shape)
print("Test set feature size:", X_test.shape)
print("Training set target variable size:", y_train_res.shape)
print("Test set target variable size:", y_test.shape)

kf = StratifiedKFold(n_splits=5 , shuffle=True, random_state=42)

def evaluate_train_model(model):

    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
    scores = cross_validate(model, X_train_res, y_train_res, cv=kf, scoring=scoring_metrics)

    mean_accuracy = np.mean(scores['test_accuracy']) * 100
    mean_precision = np.mean(scores['test_precision']) * 100
    mean_recall = np.mean(scores['test_recall']) * 100
    mean_f1 = np.mean(scores['test_f1']) * 100

    print("Average accuracy: {:.4f}%".format(mean_accuracy))
    print("Average precision: {:.4f}%".format(mean_precision))
    print("Average recall: {:.4f}%".format(mean_recall))
    print("Average F1-score: {:.4f}%".format(mean_f1))

"""#### Default Parameters

Use default parameters for the following classifiers:

*   Logistic Regression
*   Decision Tree
*   Random Forest
"""

# Logistic Regression
lr = LogisticRegression(random_state=42)

evaluate_train_model(lr)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)

evaluate_train_model(dt)

# Random Forest
rf = RandomForestClassifier(random_state=42)

evaluate_train_model(rf)

"""Logistic Regression provides a solid baseline performance with decent metrics across the board. Random Forest outperforms both Logistic Regression and Decision Tree, showing the highest accuracy and balanced precision, recall, and F1-score. This suggests that Random Forest, with its ensemble approach, captures more complex patterns in the data, making it the best-performing model among the three.

### Tuning Hyperparameter

We conduct a GridSearch for each chosen hyperparameter and select the one with the highest F1-score.
"""

# Function to find the best parameters for a model
def find_best_parameters(model, param_grid):
  # Create GridSearchCV
  grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=kf, verbose=1, n_jobs=-1, refit='F1')
  grid_search.fit(X_train_res, y_train_res)
  best_f1_score = grid_search.best_score_

  # Print the best parameters and best F1-score
  print("Best parameters:", grid_search.best_params_)
  print("Best cross-validation F1-score: {:.4f}%".format(grid_search.best_score_*100))
  return grid_search.best_estimator_, best_f1_score

# Logistic Regression
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [50, 100, 200, 300, 400]
}

lr_best_model, lr_best_f1 = find_best_parameters(lr, lr_param_grid)

# Decision Tree
dt_param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}

dt_best_model, dt_best_f1 = find_best_parameters(dt, dt_param_grid)

# Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

rf_best_model, rf_best_f1 = find_best_parameters(rf, rf_param_grid)

"""### Model Evaluation Metric

### Confusion Matrix, Accuracy, Precision, Recall, and F1-score
"""

# Function to evaluate the model on the test data
def evaluate_test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Evaluate the model performance
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    print(f"\nModel: {model.__class__.__name__}")
    print('\nConfusion Matrix:')
    print(cm)
    print('\nTest Set Accuracy: {:.4f}%'.format(accuracy))
    print('Test Set Precision: {:.4f}%'.format(precision))
    print('Test Set Recall: {:.4f}%'.format(recall))
    print('Test Set F1-Score: {:.4f}%'.format(f1))

    return accuracy, precision, recall, f1

# Plot a bar chart to compare performance of models

# Initialize lists to store evaluation metrics
models = [lr_best_model, dt_best_model, rf_best_model]
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracy_values = []
precision_values = []
recall_values = []
f1_values = []

# Evaluate models and store metrics
for model, name in zip(models, model_names):
    metrics = evaluate_test_model(model, X_test, y_test)
    accuracy_values.append(metrics[0])
    precision_values.append(metrics[1])
    recall_values.append(metrics[2])
    f1_values.append(metrics[3])

# Sort the models based on accuracy in descending order
sorted_indices = np.argsort(f1_values)[::-1]
model_names = [model_names[i] for i in sorted_indices]
accuracy_values = [accuracy_values[i] for i in sorted_indices]
precision_values = [precision_values[i] for i in sorted_indices]
recall_values = [recall_values[i] for i in sorted_indices]
f1_values = [f1_values[i] for i in sorted_indices]

# Plotting
labels = model_names
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - 3/2 * width, accuracy_values, width, label='Accuracy')
rects2 = ax.bar(x - width/2, precision_values, width, label='Precision')
rects3 = ax.bar(x + width/2, recall_values, width, label='Recall')
rects4 = ax.bar(x + 3/2 * width, f1_values, width, label='F1-Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Model Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Add exact values above each bar
for rects in [rects1, rects2, rects3, rects4]:
    for rect in rects:
        yval = round(rect.get_height())
        plt.text(rect.get_x() + rect.get_width()/2.0, yval, "{}%".format(yval), ha='center', va='bottom', fontsize=8)

# Move the legend to the bottom
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4)

plt.show()

"""Random Forest remains the best-performing model in terms of accuracy and overall metrics, both before and after tuning. Hyperparameter tuning provided marginal improvements in accuracy for Logistic Regression and Decision Tree, but also revealed trade-offs in precision and recall.

### ROC Curve and Precision Recall Curve
"""

# Function to plot the ROC curve
def plot_roc_curve(models, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for model in models:
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_value = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {auc_value:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Plot the ROC curves for each model
plot_roc_curve(models, X_test, y_test)

"""Both Logistic Regression and Random Forest have high AUC values of 0.90, indicating excellent model performance. These models are effective at classifying positive and negative instances."""

# Function to plot the Precision-Recall curve
def plot_precision_recall_curve(models, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for model in models:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{model.__class__.__name__} (AUC = {pr_auc:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

# Plot the Precision-Recall curves for each model
plot_precision_recall_curve(models, X_test, y_test)

"""The Precision-Recall Curve analysis demonstrates that the Random Forest Classifier is the best-performing model in terms of maintaining a good balance between precision and recall, followed by the Decision Tree and Logistic Regression.

## **2.6 Customer Profiling**

* Membership tier
*   Age distribution by cluster
*   Total spending vs income by cluster
* Education status by cluster
* Marital status by cluster
* Number of customer's children at home by cluster
* Count of Customers for Target Campaign (Cmp6)
"""

# Calculate average R, F, M values for each K-means cluster
kmeans_avg_rfm = df_clustering.groupby('k Means Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

print("\nAverage RFM values for k-means clustering:")
kmeans_avg_rfm

"""Based on average RFM values we can catagorize the customer clusters into membership-tiers:

|Cluster  | Tier  | Characteristics  |
|-------|----------|----------------------------------|
| 1   | Platinum | Recent, high frequency & monetary (loyal) |
| 4   | Gold | High frequency & monetary, low recency |
| 2   | Silver | Recent, low frequency & monetary |
| 3 | Bronze | Low recency, frequency & monetary |

"""

sns.kdeplot(data=df_clustering, x='Age', hue='k Means Cluster')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Age Distribution by k-Means Cluster')
plt.show()

"""Platinum members (cluster 1) shows high density in the senior age group, particularly after retirement age. Bronze members (cluster 3) dominates the younger age range which is those before 40 years old. Silver (clusters 2) and gold members (cluster 4) cover quite a broad range of middle-aged individuals."""

sns.scatterplot(data=df_clustering, x='Monetary', y='Income', hue='k Means Cluster')
plt.xlabel('Total Spending')
plt.ylabel('Income')
plt.xlim(0, 2500)
plt.ylim(0, 200000)
plt.title('Scatter Plot of Total Spending vs. Income by k-Means Cluster')
plt.show()

"""The overall pattern in the scatter plot suggests a roughly linear relationship, where higher income levels are associated with higher total spending. Silver (Cluster 2) and bronze members (cluster 3) represents individuals with lower total spending and income, indicating a budget-conscious group. Platinum (Cluster 1) and gold members (cluster 4) shows a wider range in both spending and income."""

sns.countplot(data=df_clustering, x='k Means Cluster', hue='Education')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Count of Customers by k-Means Cluster and Education Status')
plt.show()

"""Graduation-level education is the most common across all clusters, indicating a generally well-educated customer base. PhDs and Masters make up significant proportions, especially in platinum members (clusters 1) and bronze members (cluster 3), suggesting higher educational qualifications."""

sns.countplot(data=df_clustering, x='k Means Cluster', hue='Marital_Status')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Count of Customers by k-Means Cluster and Marital Status')
plt.show()

"""A significant portion of the customer base is married. Together and Single status, these two categories follow behind, suggesting a mix of cohabitating partners and single individuals."""

df_clustering['CustomerChildren'] = df_clustering["KidHome"] + df_clustering['TeenHome'] #total number of kids (baby and teens) at home
sns.countplot(df_clustering, x='k Means Cluster', hue='CustomerChildren')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title("Count of Customers by k-Means Cluster and Customer's Children")
plt.show()

"""Platinum and gold members (clusters 1 and 4) have a significant proportion of customers with no children, which could indicate a focus on a different lifestyle or life stage. Silver and bronze members (clusters 2 and 3) show a diverse range of family sizes, suggesting varied needs and preferences within these groups."""

sns.countplot(data=df_clustering, x='k Means Cluster', hue='Cmp6')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Count of Customers by k-Means Cluster and Target Campaign (Cmp6)')
plt.show()

"""Gold members (cluster 4) has the highest target of campaign engagement from customers. Gold members is the most responsive to the campaign, whereas silver members (cluster 2) is the least.

# **3.0 Conclusion**

Our predictive model has effectively segmented customers and identified key features influencing purchasing behavior. Notably, recency, catalog purchases, and income, along with sales-related features like MeatSales and StorePurchases, stand out as crucial predictors. The customer segmentation reveals that platinum and gold members share a similar profile, typically aged 40-60, with higher income and spending over $500, and most having no children.

Using these insights, we can create targeted campaigns for platinum and gold member customers. For example, we can create exclusive promotions targeting the 40-60 age group with higher disposable income and no children by bundling premium product lines into the campaign. Since catalog purchases are a key feature in our predictive model, we can incorporate catalogs into the campaign to reach the target market, offering incentives for customers to try new products or categories they haven't purchased yet.

Additionally, the company can leverage our predictive model to develop future targeted marketing campaigns that are more effective and tailored to customer preferences and behaviors. This data-driven approach will enhance customer engagement, satisfaction, and ultimately, drive higher sales.

# **4.0 References**

Nailson. (2020, February 19). Nailson/ifood-data-business-analyst-TEST: Ifood Brain Team Data Challenge for data analysts role. GitHub. https://github.com/nailson/ifood-data-business-analyst-test

Zulaikha, S., Mohamed, H., Kurniawati, M., Rusgianto, S., & Rusmita, S. A. (2020). Customer predictive analytics using artificial intelligence. The Singapore Economic Review, 1–12. https://doi.org/10.1142/s0217590820480021
"""