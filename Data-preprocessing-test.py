import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori

# Read the data
df = pd.read_csv("Groceries data.csv")

# Basic information about the data
print("Dataset shape:", df.shape)
print("\nDataset info:")
df.info()

# Data preprocessing
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

print("\nAfter date conversion:")
df.info()

# Check for null values
print("\nNull values:", df.isnull().sum().sum())

# Sample data for faster processing (take first 1000 records)
df_sample = df.head(1000).copy()

# Display basic statistics
print(f"\nSample dataset shape: {df_sample.shape}")
print(f"\nNumber of unique customers in sample: {df_sample['Member_number'].nunique()}")
print(f"Number of unique items in sample: {df_sample['itemDescription'].nunique()}")

# Show most popular items in sample
print("\nTop 10 most popular items in sample:")
item_counts = df_sample['itemDescription'].value_counts().head(10)
print(item_counts)

# Create transaction data for Apriori algorithm
transactions = df_sample.groupby('Member_number')['itemDescription'].apply(list).tolist()

print(f"\nNumber of transactions in sample: {len(transactions)}")
print(f"Sample transaction: {transactions[0]}")

# Apply Apriori algorithm with relaxed parameters for sample data
print("\nRunning Apriori algorithm on sample data...")
try:
    rules = apriori(transactions=transactions,
                   min_support=0.01,  # Reduced from 0.002 for sample
                   min_confidence=0.05,
                   min_lift=1.5,      # Reduced from 3 for sample
                   min_length=2)
    
    results = list(rules)
    print(f"Number of association rules found: {len(results)}")
    
    # Display first few rules
    if len(results) > 0:
        print("\nFirst 5 Association Rules:")
        for i, rule in enumerate(results[:5]):
            print(f"\nRule {i+1}:")
            print(f"Items: {list(rule.items)}")
            print(f"Support: {rule.support:.4f}")
            
            for statistic in rule.ordered_statistics:
                print(f"Antecedent: {list(statistic.items_base)}")
                print(f"Consequent: {list(statistic.items_add)}")
                print(f"Confidence: {statistic.confidence:.4f}")
                print(f"Lift: {statistic.lift:.4f}")
    else:
        print("No association rules found with the given parameters.")
        
except Exception as e:
    print(f"Error running Apriori: {e}")

# Create a simple visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
item_counts.plot(kind='bar')
plt.title('Top 10 Items in Sample')
plt.xlabel('Items')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
# Monthly sales in sample
df_sample.set_index('Date').resample("ME")["itemDescription"].count().plot()
plt.title('Items Sold by Month (Sample)')
plt.xlabel('Month')
plt.ylabel('Number of Items')

plt.tight_layout()
plt.savefig('sample_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAnalysis complete! Visualization saved as 'sample_analysis.png'")