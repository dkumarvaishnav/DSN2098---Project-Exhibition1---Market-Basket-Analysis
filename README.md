# Market Basket Analysis using Apriori Algorithm

This project performs Market Basket Analysis on a grocery dataset to identify frequently purchased items and association rules between them. The analysis is done using the Apriori algorithm, which is a classic algorithm for learning association rules.

## Dataset

The dataset used in this project is `Groceries data.csv`. It contains transactional data of a grocery store. The columns include `Member_number`, `Date`, and `itemDescription`.

## Installation

To run this project, you need to have Python installed. You also need to install the following libraries:

- pandas
- numpy
- seaborn
- matplotlib
- apyori

You can install these libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib apyori
```

## Usage

The main script for this project is `Data-preprocessing.py`. You can run it from the command line:

```bash
python Data-preprocessing.py
```

This script will perform the data preprocessing, run the Apriori algorithm, and print the association rules found in the dataset.

The project also includes a Jupyter Notebook `Untitled1.ipynb` which contains the same analysis with detailed steps and visualizations.

## Results

The script will output the association rules found by the Apriori algorithm. These rules indicate which items are frequently purchased together. For example, a rule might be `{whole milk} -> {other vegetables}`, which means that customers who buy whole milk are also likely to buy other vegetables.
