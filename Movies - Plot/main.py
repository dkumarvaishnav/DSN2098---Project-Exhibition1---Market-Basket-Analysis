import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('final(20).csv')

# Filter out rows with non-numeric values in the "Age" column
data = data[data['Age'].str.isnumeric()]

# Check if there are any rows left in the DataFrame
if not data.empty:
    # Convert the "Age" column to integers
    data['Age'] = data['Age'].astype(int)

    # Group the data by age rating and count the number of movies in each rating
    age_counts = data['Age'].value_counts().sort_index()

    # Plot the distribution of movies by age rating
    plt.figure(figsize=(10, 6))
    age_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Age Rating')
    plt.ylabel('Number of Movies')
    plt.title('Distribution of Movies by Age Rating')
    plt.xticks(rotation=0)
    plt.show()
else:
    print("No valid data found in the DataFrame.")
