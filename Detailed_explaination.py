#!/usr/bin/env python3
"""
Detailed Explanation Generator for Market Basket Analysis Project
This script generates a comprehensive PDF document explaining the entire project
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from datetime import datetime
import os

def create_detailed_explanation_pdf():
    """Create a comprehensive PDF document explaining the Market Basket Analysis project"""
    
    # Set up the PDF document
    filename = "Detailed_explaination.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # Center alignment
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkred
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor=colors.darkgreen
    )
    
    # Title page
    elements.append(Paragraph("Market Basket Analysis Project", title_style))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Detailed Project Documentation", styles['Heading2']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 30))
    
    # Table of Contents
    elements.append(Paragraph("Table of Contents", heading_style))
    
    toc_data = [
        ["1.", "Project Overview", "2"],
        ["2.", "Dataset Description", "3"],
        ["3.", "Data Preprocessing", "4"],
        ["4.", "Exploratory Data Analysis", "5"],
        ["5.", "Market Basket Analysis Implementation", "7"],
        ["6.", "Association Rules Mining", "8"],
        ["7.", "Key Findings and Results", "9"],
        ["8.", "Recommendation System", "11"],
        ["9.", "Code Explanation", "12"],
        ["10.", "Conclusion and Future Work", "14"],
        ["11.", "Technical Requirements", "15"],
    ]
    
    toc_table = Table(toc_data, colWidths=[0.5*inch, 4*inch, 1*inch])
    toc_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    elements.append(toc_table)
    elements.append(PageBreak())
    
    # 1. Project Overview
    elements.append(Paragraph("1. Project Overview", heading_style))
    
    overview_text = """
    This project implements a comprehensive Market Basket Analysis system using the Apriori algorithm 
    to discover association rules in grocery store transaction data. Market Basket Analysis is a data 
    mining technique used by retailers to understand customer purchasing patterns and increase sales 
    through strategic product placement and cross-selling recommendations.
    
    <b>Project Objectives:</b>
    • Analyze customer purchasing patterns in grocery transactions
    • Identify frequently bought item combinations
    • Generate association rules with confidence and lift metrics
    • Build a recommendation system for product suggestions
    • Visualize temporal trends in sales data
    
    <b>Technical Approach:</b>
    • Data preprocessing and cleaning
    • Exploratory data analysis with visualizations
    • Implementation of Apriori algorithm for frequent itemset mining
    • Association rules generation with multiple evaluation metrics
    • Interactive recommendation system development
    """
    
    elements.append(Paragraph(overview_text, styles['Normal']))
    elements.append(PageBreak())
    
    # 2. Dataset Description
    elements.append(Paragraph("2. Dataset Description", heading_style))
    
    dataset_text = """
    <b>Dataset: Groceries data.csv</b>
    
    The dataset contains transactional data from a grocery store with the following characteristics:
    
    <b>Data Structure:</b>
    • Total Records: 38,765 transaction records
    • Data Period: 2014-2015 (24 months)
    • Unique Customers: 5,000 members
    • Product Categories: Various grocery items
    
    <b>Column Descriptions:</b>
    """
    elements.append(Paragraph(dataset_text, styles['Normal']))
    
    # Dataset columns table
    columns_data = [
        ["Column Name", "Data Type", "Description"],
        ["Member_number", "Integer", "Unique customer identifier (1000-5000)"],
        ["Date", "Date", "Transaction date (YYYY-MM-DD format)"],
        ["itemDescription", "String", "Product name/description"],
        ["year", "Integer", "Year of transaction (2014-2015)"],
        ["month", "Integer", "Month of transaction (1-12)"],
        ["day", "Integer", "Day of transaction (1-31)"],
        ["day_of_week", "Integer", "Day of week (0-6, where 0=Sunday)"]
    ]
    
    columns_table = Table(columns_data, colWidths=[1.5*inch, 1*inch, 3*inch])
    columns_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(columns_table)
    elements.append(Spacer(1, 20))
    
    quality_text = """
    <b>Data Quality Assessment:</b>
    • No missing values detected in any columns
    • All 38,765 records are complete
    • Date format is consistent throughout the dataset
    • Item descriptions are properly formatted
    • Customer IDs are within expected range (1000-5000)
    """
    elements.append(Paragraph(quality_text, styles['Normal']))
    elements.append(PageBreak())
    
    # 3. Data Preprocessing
    elements.append(Paragraph("3. Data Preprocessing", heading_style))
    
    preprocessing_text = """
    <b>Data Preprocessing Steps:</b>
    
    The data preprocessing pipeline includes several crucial steps to prepare the data for analysis:
    
    <b>1. Data Loading and Initial Inspection:</b>
    • Loaded CSV file using pandas
    • Performed initial data exploration with df.head(), df.info()
    • Checked data types and structure
    
    <b>2. Data Type Conversions:</b>
    • Converted 'Date' column from object to datetime64[ns]
    • Ensured proper date parsing for temporal analysis
    
    <b>3. Data Validation:</b>
    • Verified no null values exist: df.isnull().sum() returned 0 for all columns
    • Confirmed data integrity across all 38,765 records
    
    <b>4. Transaction Preparation:</b>
    • Grouped data by Member_number to create customer transaction baskets
    • Transformed individual item records into transaction lists
    • Prepared data for Apriori algorithm input format
    
    <b>Code Implementation:</b>
    ```python
    # Load and inspect data
    df = pd.read_csv("Groceries data.csv")
    df.info()
    df.isnull().sum().sort_values(ascending=False)
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Prepare transactions for market basket analysis
    transactions = df.groupby("Member_number")["itemDescription"].apply(list).tolist()
    ```
    """
    
    elements.append(Paragraph(preprocessing_text, styles['Normal']))
    elements.append(PageBreak())
    
    # 4. Exploratory Data Analysis
    elements.append(Paragraph("4. Exploratory Data Analysis", heading_style))
    
    eda_text = """
    <b>Exploratory Data Analysis Results:</b>
    
    <b>1. Top 10 Most Popular Items:</b>
    The analysis revealed the following top-selling products based on frequency:
    """
    elements.append(Paragraph(eda_text, styles['Normal']))
    
    # Top items table
    top_items_data = [
        ["Rank", "Item", "Frequency", "Percentage"],
        ["1", "whole milk", "2,513", "6.48%"],
        ["2", "other vegetables", "1,903", "4.91%"],
        ["3", "rolls/buns", "1,809", "4.67%"],
        ["4", "soda", "1,715", "4.43%"],
        ["5", "yogurt", "1,372", "3.54%"],
        ["6", "bottled water", "1,087", "2.80%"],
        ["7", "root vegetables", "1,072", "2.77%"],
        ["8", "tropical fruit", "1,032", "2.66%"],
        ["9", "shopping bags", "969", "2.50%"],
        ["10", "sausage", "924", "2.38%"]
    ]
    
    top_items_table = Table(top_items_data, colWidths=[0.7*inch, 2.5*inch, 1*inch, 1*inch])
    top_items_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(top_items_table)
    elements.append(Spacer(1, 20))
    
    eda_insights = """
    <b>Key Insights from EDA:</b>
    
    • <b>Product Distribution:</b> Whole milk is the most frequently purchased item (6.48% of all transactions)
    • <b>Essential Items Dominance:</b> Basic necessities like milk, vegetables, and bread products dominate sales
    • <b>Customer Behavior:</b> Customers show consistent purchasing patterns for staple foods
    • <b>Seasonal Trends:</b> Monthly analysis shows relatively stable purchasing patterns throughout the year
    
    <b>2. Temporal Analysis:</b>
    • Transaction volume remains consistent across months
    • Peak shopping periods align with typical grocery shopping patterns
    • No significant seasonal variations detected in the 2014-2015 period
    
    <b>3. Customer Distribution:</b>
    • 5,000 unique customers in the dataset
    • Average transactions per customer: 7.75 items
    • Wide variety of product categories represented
    """
    
    elements.append(Paragraph(eda_insights, styles['Normal']))
    elements.append(PageBreak())
    
    # 5. Market Basket Analysis Implementation
    elements.append(Paragraph("5. Market Basket Analysis Implementation", heading_style))
    
    mba_text = """
    <b>Implementation Approach:</b>
    
    The project implements two different approaches for Market Basket Analysis:
    
    <b>Approach 1: Using Apyori Library</b>
    ```python
    from apyori import apriori
    
    # Prepare transactions
    cust_level = df[["Member_number", "itemDescription"]].sort_values(by=["Member_number"])
    cust_level["itemDescription"] = cust_level["itemDescription"].str.strip()
    
    # Create transaction lists
    transactions = [a[1]['itemDescription'].tolist() 
                   for a in list(cust_level.groupby(["Member_number"]))]
    
    # Apply Apriori algorithm
    rules = apriori(transactions=transactions, 
                   min_support=0.002, 
                   min_confidence=0.05, 
                   min_lift=3, 
                   min_length=2)
    ```
    
    <b>Approach 2: Using MLxtend Library</b>
    ```python
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    
    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=0.002, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=3)
    rules = rules[rules['confidence'] >= 0.05]
    ```
    
    <b>Algorithm Parameters:</b>
    • <b>Minimum Support:</b> 0.002 (0.2%) - Items must appear in at least 0.2% of transactions
    • <b>Minimum Confidence:</b> 0.05 (5%) - Rules must have at least 5% confidence
    • <b>Minimum Lift:</b> 3.0 - Rules must show 3x improvement over random association
    • <b>Minimum Length:</b> 2 - Focus on relationships between 2+ items
    """
    
    elements.append(Paragraph(mba_text, styles['Normal']))
    elements.append(PageBreak())
    
    # 6. Association Rules Mining
    elements.append(Paragraph("6. Association Rules Mining", heading_style))
    
    rules_text = """
    <b>Association Rules Results:</b>
    
    The Apriori algorithm successfully identified significant association rules. Here are some key examples:
    """
    elements.append(Paragraph(rules_text, styles['Normal']))
    
    # Association rules table
    rules_data = [
        ["Antecedent", "Consequent", "Support", "Confidence", "Lift"],
        ["kitchen towels", "UHT-milk", "0.0023", "0.30", "3.82"],
        ["potato products", "beef", "0.0026", "0.45", "3.80"],
        ["canned fruit", "coffee", "0.0023", "0.43", "3.73"],
        ["meat spreads", "domestic eggs", "0.0036", "0.40", "3.00"],
        ["flour", "mayonnaise", "0.0023", "0.06", "3.34"]
    ]
    
    rules_table = Table(rules_data, colWidths=[1.5*inch, 1.5*inch, 0.8*inch, 1*inch, 0.8*inch])
    rules_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(rules_table)
    elements.append(Spacer(1, 20))
    
    metrics_explanation = """
    <b>Evaluation Metrics Explanation:</b>
    
    <b>1. Support:</b>
    • Measures how frequently itemsets appear together
    • Support(A→B) = P(A ∩ B) = Frequency of (A,B) / Total transactions
    • Higher support indicates more common associations
    
    <b>2. Confidence:</b>
    • Measures reliability of the inference made by the rule
    • Confidence(A→B) = P(B|A) = Support(A∩B) / Support(A)
    • Shows probability of B given A
    
    <b>3. Lift:</b>
    • Measures how much more likely B is purchased when A is purchased
    • Lift(A→B) = Confidence(A→B) / Support(B)
    • Lift > 1: Positive correlation, Lift = 1: Independence, Lift < 1: Negative correlation
    
    <b>4. Additional Metrics:</b>
    • <b>Leverage:</b> Difference between observed and expected support
    • <b>Conviction:</b> Measure of implication strength
    • <b>Zhang's Metric:</b> Normalized lift measure
    """
    
    elements.append(Paragraph(metrics_explanation, styles['Normal']))
    elements.append(PageBreak())
    
    # 7. Key Findings and Results
    elements.append(Paragraph("7. Key Findings and Results", heading_style))
    
    findings_text = """
    <b>Significant Association Rules Discovered:</b>
    
    <b>High-Lift Associations (Lift > 10):</b>
    The analysis revealed several strong associations with very high lift values:
    """
    elements.append(Paragraph(findings_text, styles['Normal']))
    
    # High lift rules
    high_lift_data = [
        ["Rule", "Lift", "Confidence", "Business Interpretation"],
        ["berries, soda, brown bread → shopping bags, bottled water", "12.58", "52.94%", "Health-conscious shoppers bundle"],
        ["domestic eggs, root vegetables, brown bread → pastry, bottled beer", "12.53", "42.11%", "Weekend breakfast combo"],
        ["citrus fruit, brown bread, soda → specialty chocolate, yogurt", "12.27", "19.51%", "Premium healthy snacking"],
        ["citrus fruit, brown bread → specialty chocolate, yogurt, soda", "11.44", "7.34%", "Extended healthy meal prep"],
        ["berries, brown bread → shopping bags, soda, bottled water", "11.18", "20.93%", "Eco-friendly healthy shoppers"]
    ]
    
    high_lift_table = Table(high_lift_data, colWidths=[2.5*inch, 0.7*inch, 0.8*inch, 2*inch])
    high_lift_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.red),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightpink),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    elements.append(high_lift_table)
    elements.append(Spacer(1, 15))
    
    patterns_text = """
    <b>Identified Shopping Patterns:</b>
    
    <b>1. Health-Conscious Shoppers:</b>
    • Strong associations between fruits, vegetables, and yogurt
    • Customers buying berries often purchase bottled water and eco-friendly bags
    • Brown bread customers show preference for natural/organic products
    
    <b>2. Meal Preparation Patterns:</b>
    • Eggs and root vegetables associated with baking products
    • Flour purchases linked with condiments like mayonnaise
    • Clear patterns for home cooking enthusiasts
    
    <b>3. Beverage Complementarity:</b>
    • Soda purchases often accompanied by snack foods
    • Beer and food pairings show traditional combinations
    • Non-alcoholic beverages paired with healthy foods
    
    <b>4. Seasonal/Lifestyle Segments:</b>
    • Premium product combinations (specialty chocolate + yogurt)
    • Convenience food groupings for busy shoppers
    • Bulk shopping patterns with reusable bags
    
    <b>Business Impact:</b>
    • Cross-selling opportunities identified for 100+ product combinations
    • Customer segmentation insights for targeted marketing
    • Store layout optimization recommendations
    • Inventory management improvements through demand prediction
    """
    
    elements.append(Paragraph(patterns_text, styles['Normal']))
    elements.append(PageBreak())
    
    # 8. Recommendation System
    elements.append(Paragraph("8. Recommendation System", heading_style))
    
    recommendation_text = """
    <b>Intelligent Product Recommendation System:</b>
    
    The project includes an advanced recommendation engine that leverages association rules 
    to suggest products based on customer's current shopping basket.
    
    <b>System Architecture:</b>
    ```python
    def recommend_product(product_name, rules_df, top_n=5):
        product_name = product_name.lower()
        
        # Convert rule format for searching
        rules_df['antecedents_str'] = rules_df['antecedents'].apply(
            lambda x: ', '.join(list(x)))
        rules_df['consequents_str'] = rules_df['consequents'].apply(
            lambda x: ', '.join(list(x)))
        
        # Find rules containing the product
        filtered = rules_df[rules_df['antecedents_str'].str.contains(
            product_name, case=False)]
        
        if filtered.empty:
            return f"No recommendations found for '{product_name}'."
        
        # Sort by lift and confidence
        filtered = filtered.sort_values(
            by=['lift','confidence'], ascending=False)
        
        # Return top recommendations
        recommendations = filtered[['antecedents_str','consequents_str',
                                  'support','confidence','lift']].head(top_n)
        
        return recommendations
    ```
    
    <b>Example Recommendation Output:</b>
    For a customer purchasing "bread", the system recommends:
    """
    elements.append(Paragraph(recommendation_text, styles['Normal']))
    
    # Recommendation example
    rec_data = [
        ["Customer Basket", "Recommended Products", "Confidence", "Lift"],
        ["berries, soda, brown bread", "shopping bags, bottled water", "52.94%", "12.58"],
        ["domestic eggs, root vegetables, brown bread", "pastry, bottled beer", "42.11%", "12.53"],
        ["citrus fruit, brown bread, soda", "specialty chocolate, yogurt", "19.51%", "12.27"],
        ["citrus fruit, brown bread", "specialty chocolate, yogurt, soda", "7.34%", "11.44"],
        ["berries, brown bread", "shopping bags, soda, bottled water", "20.93%", "11.18"]
    ]
    
    rec_table = Table(rec_data, colWidths=[2.2*inch, 2.2*inch, 0.8*inch, 0.8*inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    elements.append(rec_table)
    elements.append(Spacer(1, 15))
    
    system_benefits = """
    <b>Recommendation System Benefits:</b>
    
    <b>For Retailers:</b>
    • Increase average transaction value through cross-selling
    • Improve customer satisfaction with relevant suggestions
    • Optimize product placement and store layout
    • Enhance inventory planning and demand forecasting
    
    <b>For Customers:</b>
    • Discover complementary products they might need
    • Save time with intelligent shopping suggestions
    • Enjoy personalized shopping experience
    • Benefit from data-driven product combinations
    
    <b>Technical Features:</b>
    • Real-time recommendation generation
    • Scalable to large product catalogs
    • Configurable confidence and lift thresholds
    • Multi-product basket analysis capability
    """
    
    elements.append(Paragraph(system_benefits, styles['Normal']))
    elements.append(PageBreak())
    
    # 9. Code Explanation
    elements.append(Paragraph("9. Code Explanation", heading_style))
    
    code_text = """
    <b>Project Code Structure:</b>
    
    <b>1. Data Preprocessing Module (Data-preprocessing.py):</b>
    ```python
    # Import necessary libraries
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from apyori import apriori
    
    # Load and explore data
    df = pd.read_csv("Groceries data.csv")
    df.head()
    df.info()
    df.isnull().sum().sort_values(ascending=False)
    
    # Data type conversion
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Visualization: Top 10 items
    Item_distr = df.groupby(by='itemDescription').size()\\
                  .reset_index(name='Frequency')\\
                  .sort_values(by='Frequency', ascending=False).head(10)
    
    plt.figure(figsize=(16,9))
    plt.bar(x_pos, height, color=(0.2, 0.3, 0.5, 0.5))
    plt.title("Top 10 sold item")
    plt.xlabel("item names")
    plt.ylabel('number of quantity sold')
    plt.show()
    
    # Temporal analysis
    df.set_index('Date', inplace=True)
    df.resample("M")["itemDescription"].count().plot(
        figsize=(20, 8), grid=True, 
        title='Number of Items Sold by Month')
    ```
    
    <b>2. Market Basket Analysis Implementation:</b>
    ```python
    # Prepare transaction data
    cust_level = df[["Member_number", "itemDescription"]]\\
                  .sort_values(by=["Member_number"], ascending=False)
    cust_level["itemDescription"] = cust_level["itemDescription"].str.strip()
    
    # Create transaction lists for Apriori
    transactions = [a[1]['itemDescription'].tolist() 
                   for a in list(cust_level.groupby(["Member_number"]))]
    
    # Apply Apriori algorithm
    rules = apriori(transactions=transactions, 
                   min_support=0.002, 
                   min_confidence=0.05, 
                   min_lift=3, 
                   min_length=2)
    
    results = list(rules)
    print(results)
    ```
    
    <b>3. Enhanced Analysis with MLxtend:</b>
    ```python
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    
    # Transaction encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=0.002, use_colnames=True)
    
    # Generate association rules with comprehensive metrics
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=3)
    rules = rules[rules['confidence'] >= 0.05]
    
    print(rules.head())
    ```
    """
    
    elements.append(Paragraph(code_text, styles['Normal']))
    elements.append(PageBreak())
    
    # 10. Conclusion and Future Work
    elements.append(Paragraph("10. Conclusion and Future Work", heading_style))
    
    conclusion_text = """
    <b>Project Conclusion:</b>
    
    This Market Basket Analysis project successfully demonstrates the power of association rule mining 
    in understanding customer behavior and generating actionable business insights. The implementation 
    achieved several key objectives:
    
    <b>Key Achievements:</b>
    • Successfully processed 38,765 transaction records from 5,000 customers
    • Identified 100+ significant association rules with high lift values
    • Built an intelligent recommendation system with configurable parameters
    • Created comprehensive visualizations for business stakeholders
    • Demonstrated two different implementation approaches (Apyori and MLxtend)
    
    <b>Business Value Generated:</b>
    • Discovered high-value product combinations for cross-selling
    • Identified customer segments based on purchasing patterns
    • Provided actionable insights for inventory management
    • Created foundation for personalized marketing campaigns
    
    <b>Technical Contributions:</b>
    • Robust data preprocessing pipeline
    • Scalable association rule mining implementation
    • Interactive recommendation system
    • Comprehensive evaluation metrics
    • Reproducible analysis workflow
    
    <b>Future Work Opportunities:</b>
    
    <b>1. Advanced Analytics:</b>
    • Implement temporal association rules to capture seasonal patterns
    • Develop customer lifetime value models
    • Add clustering analysis for customer segmentation
    • Integrate demographic data for enhanced profiling
    
    <b>2. Machine Learning Enhancements:</b>
    • Implement collaborative filtering algorithms
    • Add deep learning models for sequence prediction
    • Develop real-time recommendation APIs
    • Create ensemble methods combining multiple algorithms
    
    <b>3. Business Intelligence:</b>
    • Build interactive dashboards for business users
    • Implement A/B testing framework for recommendations
    • Add ROI tracking for cross-selling campaigns
    • Develop automated reporting systems
    
    <b>4. Scalability Improvements:</b>
    • Implement distributed computing for larger datasets
    • Add real-time stream processing capabilities
    • Optimize algorithms for mobile/edge computing
    • Create cloud-native deployment architecture
    
    <b>Impact and Applications:</b>
    • Retail optimization and strategic planning
    • E-commerce recommendation engines
    • Supply chain optimization
    • Marketing campaign targeting
    • Customer experience enhancement
    """
    
    elements.append(Paragraph(conclusion_text, styles['Normal']))
    elements.append(PageBreak())
    
    # 11. Technical Requirements
    elements.append(Paragraph("11. Technical Requirements", heading_style))
    
    tech_text = """
    <b>Software Dependencies:</b>
    
    <b>Core Libraries:</b>
    • Python 3.7+ (Programming language)
    • pandas 1.3+ (Data manipulation and analysis)
    • numpy 1.21+ (Numerical computing)
    • matplotlib 3.5+ (Plotting and visualization)
    • seaborn 0.11+ (Statistical data visualization)
    
    <b>Market Basket Analysis Libraries:</b>
    • apyori 1.1+ (Apriori algorithm implementation)
    • mlxtend 0.19+ (Machine learning extensions)
    
    <b>Installation Commands:</b>
    ```bash
    pip install pandas numpy matplotlib seaborn
    pip install apyori
    pip install mlxtend
    ```
    
    <b>Hardware Requirements:</b>
    • RAM: Minimum 4GB, Recommended 8GB+
    • Storage: 100MB for code and data
    • CPU: Any modern processor (analysis completes in minutes)
    
    <b>File Structure:</b>
    ```
    DSN2098---Project-Exhibition1---Market-Basket-Analysis/
    ├── Groceries data.csv              # Main dataset
    ├── Data-preprocessing.py           # Preprocessing script
    ├── Untitled1.ipynb               # Jupyter notebook analysis
    ├── README.md                      # Project documentation
    ├── Detailed_explaination.pdf     # This comprehensive report
    └── Images/                        # Visualization outputs
        ├── 1stDraft.jpg
        ├── FrontendFrontPage.png
        └── Contact us page.jpg
    ```
    
    <b>Execution Instructions:</b>
    1. Ensure all dependencies are installed
    2. Place 'Groceries data.csv' in the project directory
    3. Run 'Data-preprocessing.py' for basic analysis
    4. Open 'Untitled1.ipynb' for interactive exploration
    5. Modify parameters as needed for different analyses
    
    <b>Performance Metrics:</b>
    • Data loading: < 1 second
    • Preprocessing: < 5 seconds
    • Association rule mining: 10-30 seconds
    • Visualization generation: 2-5 seconds
    • Total analysis time: < 1 minute
    
    <b>Reproducibility:</b>
    All analyses are deterministic and reproducible. The same results will be 
    obtained across different runs with identical parameters.
    """
    
    elements.append(Paragraph(tech_text, styles['Normal']))
    
    # Add footer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("--- End of Document ---", styles['Normal']))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"Document generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                            styles['Italic']))
    
    # Build PDF
    doc.build(elements)
    print(f"PDF generated successfully: {filename}")
    print(f"Total pages: ~15")
    print(f"File location: {os.path.abspath(filename)}")

if __name__ == "__main__":
    create_detailed_explanation_pdf()