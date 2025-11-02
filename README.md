# Market Basket Analysis using Apriori Algorithm

This project performs comprehensive Market Basket Analysis on grocery store transaction data to identify frequently purchased items and association rules between them. The analysis leverages the Apriori algorithm to uncover meaningful patterns in customer purchasing behavior and generate actionable business insights.

## Project Overview

Market Basket Analysis is a data mining technique used by retailers to understand customer purchasing patterns and increase sales through strategic product placement, cross-selling recommendations, and targeted marketing campaigns. This project demonstrates the application of association rule mining to real-world grocery transaction data.

## Dataset

### Dataset Specifications
- **File**: `Groceries data.csv`
- **Total Records**: 38,765 transaction records (individual items purchased)
- **Unique Customers**: 5,000 (Member_number range: 1000-5000)
- **Data Period**: 2014-2015 (24 months)
- **Data Quality**: No missing values detected

### Dataset Structure
The dataset contains transactional data from a grocery store with the following columns:
- `Member_number`: Unique customer identifier
- `Date`: Transaction date (YYYY-MM-DD format)
- `itemDescription`: Product name/description
- `year`: Year of transaction
- `month`: Month of transaction
- `day`: Day of transaction
- `day_of_week`: Day of week (0-6, where 0=Sunday)

## Key Findings

### Most Frequent Items
1. Whole milk: 2,513 occurrences (6.48%)
2. Other vegetables: 1,903 occurrences (4.91%)
3. Rolls/buns: 1,809 occurrences (4.67%)
4. Soda: 1,715 occurrences (4.43%)
5. Yogurt: 1,372 occurrences (3.54%)
6. Bottled water: 1,087 occurrences (2.80%)
7. Root vegetables: 1,072 occurrences (2.77%)
8. Tropical fruit: 1,032 occurrences (2.66%)
9. Shopping bags: 969 occurrences (2.50%)
10. Sausage: 924 occurrences (2.38%)

### High-Impact Association Rules (Lift > 10)
- **Berries + soda + brown bread → shopping bags + bottled water**
  - Lift: 12.58, Confidence: 52.94%
  - Interpretation: Health-conscious shoppers bundle

- **Domestic eggs + root vegetables + brown bread → pastry + bottled beer**
  - Lift: 12.53, Confidence: 42.11%
  - Interpretation: Weekend breakfast combo

- **Citrus fruit + brown bread + soda → specialty chocolate + yogurt**
  - Lift: 12.27, Confidence: 19.51%
  - Interpretation: Premium healthy snacking

### Customer Segments Identified
1. **Health-Conscious Shoppers**: Strong associations between fruits, vegetables, yogurt, berries, bottled water, and eco-friendly bags
2. **Home Cooking Enthusiasts**: Patterns involving eggs, root vegetables, and baking products
3. **Premium Product Buyers**: Combinations of specialty chocolate and yogurt products

### Algorithm Parameters
- **Minimum Support**: 0.002 (0.2%) - Items must appear in at least 0.2% of transactions
- **Minimum Confidence**: 0.05 (5%) - Rules must have at least 5% confidence
- **Minimum Lift**: 3.0 - Rules must show 3x improvement over random association
- **Minimum Length**: 2 - Focus on relationships between 2+ items

## Business Recommendations

### Actionable Strategies
1. **Cross-Selling Opportunities**: Place complementary products in proximity (e.g., berries near bottled water and reusable bags)
2. **Product Bundling**: Create promotional bundles based on high-lift associations (e.g., "Healthy Breakfast Bundle" with eggs, root vegetables, and brown bread)
3. **Store Layout Optimization**: Position frequently co-purchased items strategically to increase basket size
4. **Targeted Marketing**: Develop campaigns for specific customer segments (health-conscious, home cooking, premium shoppers)
5. **Inventory Management**: Optimize stock levels based on association patterns to ensure complementary products are available together

### Potential Impact
- Increase average transaction value through strategic cross-selling of 100+ identified product combinations
- Improve customer satisfaction with intelligent product recommendations
- Enhance inventory planning based on demand prediction patterns
- Optimize store layout for better customer experience and increased sales

## Installation

To run this project, you need to have Python 3.7+ installed. Install the required libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib apyori mlxtend
```

### Required Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **seaborn**: Statistical data visualization
- **matplotlib**: Plotting and visualization
- **apyori**: Apriori algorithm implementation
- **mlxtend**: Machine learning extensions (alternative implementation)

## Usage

### Running the Main Script
The main script for this project is `Data-preprocessing.py`. Run it from the command line:

```bash
python Data-preprocessing.py
```

This script will:
1. Load and preprocess the data
2. Perform exploratory data analysis
3. Run the Apriori algorithm
4. Generate association rules
5. Output the discovered patterns

### Interactive Analysis
The project includes a Jupyter Notebook `Untitled1.ipynb` with detailed steps and visualizations:

```bash
jupyter notebook Untitled1.ipynb
```

### Generating Documentation
Run the detailed explanation generator to create a comprehensive PDF report:

```bash
python Detailed_explaination.py
```

## Project Structure

```
DSN2098---Project-Exhibition1---Market-Basket-Analysis/
├── Groceries data.csv              # Main dataset
├── Data-preprocessing.py           # Preprocessing script
├── Data-preprocessing-test.py      # Testing script
├── Untitled1.ipynb                 # Jupyter notebook analysis
├── Detailed_explaination.py        # PDF report generator
├── README.md                       # Project documentation
└── Images/                         # Visualization outputs
```

## Technical Details

### Data Preprocessing
1. Load CSV file using pandas
2. Convert 'Date' column to datetime64[ns] format
3. Validate data integrity (check for null values)
4. Group data by Member_number to create transaction baskets
5. Transform individual item records into transaction lists

### Association Rule Mining
The project implements two approaches:

**Approach 1: Using Apyori**
```python
from apyori import apriori
rules = apriori(transactions=transactions, 
               min_support=0.002, 
               min_confidence=0.05, 
               min_lift=3, 
               min_length=2)
```

**Approach 2: Using MLxtend**
```python
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(df_encoded, min_support=0.002, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=3)
```

### Evaluation Metrics
- **Support**: Measures how frequently itemsets appear together
- **Confidence**: Measures reliability of the inference made by the rule
- **Lift**: Measures how much more likely item B is purchased when item A is purchased
- **Leverage**: Difference between observed and expected support
- **Conviction**: Measure of implication strength

## Results

The analysis successfully identified:
- 100+ significant association rules with high lift values
- Multiple customer segments based on purchasing patterns
- Clear opportunities for cross-selling and product bundling
- Temporal trends in sales data (2014-2015)

Association rules indicate which items are frequently purchased together. For example, the rule `{berries, brown bread} -> {shopping bags, bottled water}` with a lift of 11.18 suggests that customers buying berries and brown bread are 11 times more likely to also purchase shopping bags and bottled water compared to random chance.

## Future Enhancements

1. **Temporal Analysis**: Implement seasonal pattern detection
2. **Customer Segmentation**: Add clustering analysis for deeper profiling
3. **Real-Time Recommendations**: Develop API for live product suggestions
4. **A/B Testing Framework**: Measure impact of recommendations on sales
5. **Deep Learning Models**: Implement sequence prediction algorithms
6. **Dashboard Development**: Create interactive visualizations for business users

## License

This project is developed for educational and research purposes.

## Contact

For questions or collaboration opportunities, please refer to the project repository.
