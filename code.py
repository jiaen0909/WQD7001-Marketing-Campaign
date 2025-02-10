import pandas as pd
import numpy as np
import regex as re
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy import stats

df = pd.read_csv('ifood_df.csv')
# Check missing values for each columns
print(df.isnull().values.any())
print(df.isnull().sum())

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

# Total Number of Complaints
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
