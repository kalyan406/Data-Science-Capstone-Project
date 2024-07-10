import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/CAR_DETAILS.csv')
# Display the first few rows and check column names
print(data.head())
print(data.columns)

# Summary statistics
print(data.describe())

#DATA ANALASYS

# Distribution plots
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
sns.histplot(data['year'], bins=20, kde=True)
plt.title('Distribution of Year')

plt.subplot(2, 2, 2)
sns.histplot(data['selling_price'], bins=20, kde=True)
plt.title('Distribution of Selling Price')

plt.subplot(2, 2, 3)
sns.histplot(data['km_driven'], bins=20, kde=True)
plt.title('Distribution of Kilometers Driven')

plt.tight_layout()
plt.show()


# Count plots for categorical variables
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='fuel', data=data)
plt.title('Count of Cars by Fuel Type')

plt.subplot(2, 2, 2)
sns.countplot(x='seller_type', data=data)
plt.title('Count of Cars by Seller Type')

plt.subplot(2, 2, 3)
sns.countplot(x='transmission', data=data)
plt.title('Count of Cars by Transmission Type')

plt.subplot(2, 2, 4)
sns.countplot(x='owner', data=data)
plt.title('Count of Cars by Owner Type')

plt.tight_layout()
plt.show()


# Scatter plot of selling price vs. year
plt.figure(figsize=(10, 6))
sns.scatterplot(x='year', y='selling_price', data=data)
plt.title('Selling Price vs. Year')
plt.show()
