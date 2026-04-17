# Retail Sales Analysis with Customer Segmentation

## Project Overview

This project performs comprehensive retail sales analysis with advanced customer segmentation using K-means clustering. The analysis helps identify distinct customer groups based on purchasing behavior, demographics, and engagement patterns, enabling targeted business strategies for each segment.

---

## Features

### 1. **Data Preprocessing**
- Automatic data cleaning (duplicate and missing value removal)
- Date conversion and validation
- Feature aggregation at customer level
- RFM (Recency, Frequency, Monetary) analysis
- Demographic feature extraction (Age, Gender)

### 2. **Customer Segmentation**
- K-means clustering with optimal K determination
- Elbow method analysis
- Silhouette score evaluation
- Feature standardization using StandardScaler
- Customer lifetime calculations

### 3. **Segment Characterization**
- Detailed profiling of each customer segment
- Revenue contribution analysis
- Purchase behavior metrics
- Demographic insights
- Engagement patterns (recency analysis)

### 4. **Data Visualization**
- 8 comprehensive visualizations including:
  - Optimal cluster determination charts
  - Segment distribution pie chart
  - Customer segment scatter plots
  - Boxplots for feature comparison
  - Heatmaps of normalized segment profiles
  - Gender distribution analysis
  - Sales trend and top products analysis

### 5. **Sales Analysis**
- Total sales metrics
- Average transaction value
- Daily sales trends
- Top-performing product categories

---

## File Structure

```
Retail Sales Analysis/
├── Retail_Sales_Analysis.py          # Main analysis script
├── retail_sales_dataset.csv          # Source data
├── README.md                         # This file
└── Output Visualizations/
    ├── 01_Optimal_Clusters.png       # Elbow method + Silhouette analysis
    ├── 02_Top_Products.png           # Top 10 products by sales
    ├── 03_Sales_Trend.png            # Sales over time
    ├── 04_Segment_Distribution.png   # Pie chart of segments
    ├── 05_Segment_Characteristics.png # Boxplots by segment
    ├── 06_Segment_Scatter.png        # Frequency vs Spend scatter
    ├── 07_Segment_Heatmap.png        # Normalized metrics heatmap
    └── 08_Gender_by_Segment.png      # Gender distribution
```

---

## Installation & Requirements

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Versions
- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0

---

## Usage

### Running the Analysis

```bash
python Retail_Sales_Analysis.py
```

### Output
- Console output with detailed segment characterizations
- 8 PNG visualization files saved to the working directory
- Segment profiles with size, revenue, and metrics

---

## Data Requirements

The dataset must contain the following columns:
- `Transaction ID`: Unique transaction identifier
- `Date`: Transaction date (YYYY-MM-DD format)
- `Customer ID`: Unique customer identifier
- `Gender`: Male/Female
- `Age`: Customer age (numeric)
- `Product Category`: Category of product purchased
- `Quantity`: Number of items purchased
- `Price per Unit`: Unit price
- `Total Amount`: Total transaction amount

---

## Segment Characteristics

### Adaptive Segment Naming
Segments are automatically labeled based on the following characteristics:

#### **Segment Levels:**
1. **VIP Loyalists**
   - High total spending + High purchase frequency
   - Most valuable customer group
   - Recommendation: Premium rewards, exclusive offers, VIP support

2. **Engaged Premium Customers**
   - High spending + Recent activity
   - Recently active high-value customers
   - Recommendation: Maintain engagement, upsell premium products

3. **Budget-Conscious Regulars**
   - High frequency + Lower spend
   - Price-sensitive but loyal
   - Recommendation: Bulk discounts, loyalty rewards

4. **At-Risk/Dormant Customers**
   - Long time since last purchase (>100 days)
   - Need re-engagement
   - Recommendation: Win-back campaigns, special offers

5. **Standard Customers**
   - Average spending and frequency
   - Regular baseline customers
   - Recommendation: Standard marketing campaigns

### Key Metrics Explained

| Metric | Description |
|--------|-------------|
| **Total Spend** | Cumulative purchase amount by customer |
| **Avg Purchase Value** | Average transaction amount |
| **Purchase Frequency** | Number of purchases made |
| **Recency** | Days since last purchase |
| **Customer Lifetime Days** | Time span from first to last purchase |
| **Age** | Customer demographics |

---

## Technical Details

### Clustering Algorithm
- **Method**: K-means clustering
- **Optimization**: Elbow method + Silhouette analysis
- **Feature Scaling**: StandardScaler (zero mean, unit variance)
- **Initialization**: k-means++ (10 iterations)

### Features Used for Clustering
1. Total Spend (Monetary value)
2. Average Purchase Value
3. Purchase Count (Frequency)
4. Recency (Days since last purchase)
5. Age (Demographics)

### Validation Metrics
- **Silhouette Score**: Measures how similar objects are to their own cluster (range: -1 to 1)
  - Higher is better
  - Value > 0.5 indicates good cluster separation
- **Inertia**: Sum of squared distances from each point to its assigned centroid
  - Lower is better
  - Used to identify elbow point

---

## Key Findings & Insights

### Analysis Output Includes:
1. **Cluster Validation**
   - Optimal number of clusters determined automatically
   - Silhouette scores for quality assessment

2. **Segment Profiles**
   - Customer count per segment
   - Revenue contribution
   - Average metrics (spend, frequency, age)
   - Gender distribution

3. **Business Insights**
   - Revenue concentration by segment
   - Customer retention indicators
   - Growth potential segments
   - At-risk customer identification

4. **Sales Analytics**
   - Total revenue
   - Average transaction value
   - Daily sales trends
   - Top-performing product categories

---

## Visualizations Explained

### 01_Optimal_Clusters.png
- **Left**: Elbow chart showing inertia vs K value
- **Right**: Silhouette scores across K values
- **Use**: Determine optimal cluster count

### 02_Top_Products.png
- Bar chart of top 10 product categories by revenue
- Identifies bestselling product lines

### 03_Sales_Trend.png
- Line chart of cumulative sales over time
- Shows seasonality and trends

### 04_Segment_Distribution.png
- Pie chart showing customer distribution across segments
- Highlights relative segment sizes

### 05_Segment_Characteristics.png
- 6 boxplots comparing features across segments
- Shows distribution, median, and outliers per feature

### 06_Segment_Scatter.png
- 2D scatter plot: Purchase Frequency vs Total Spend
- Points colored by segment
- Shows cluster separation

### 07_Segment_Heatmap.png
- Normalized metrics by segment
- Heatmap colors indicate relative strength per metric

### 08_Gender_by_Segment.png
- Stacked bar chart of gender distribution
- Gender breakdown per segment

---

## Recommendations by Segment

### For VIP Loyalists
- ✓ Personalized communication
- ✓ Exclusive early access to new products
- ✓ Premium loyalty rewards
- ✓ Dedicated customer support

### For Engaged Premium Customers
- ✓ Upselling opportunities
- ✓ Cross-sell complementary products
- ✓ Maintain engagement through targeted campaigns
- ✓ Premium product recommendations

### For Budget-Conscious Regulars
- ✓ Volume-based discounts
- ✓ Bundle offers
- ✓ Frequency-based rewards
- ✓ Long-term loyalty incentives

### For At-Risk/Dormant Customers
- ✓ Win-back campaigns
- ✓ Special reactivation offers
- ✓ Feedback surveys
- ✓ Recent bestseller recommendations

### For Standard Customers
- ✓ Regular promotional campaigns
- ✓ Newsletter communications
- ✓ Seasonal promotions
- ✓ Product recommendations

---

## Customization

### Adjusting Number of Segments
Modify the range in Section 4:
```python
K_range = range(2, 11)  # Change upper limit
```

### Changing Clustering Features
Edit the feature list in Section 3:
```python
clustering_features = ['Total_Spend', 'Avg_Purchase', 'Purchase_Count', 'Recency', 'Age']
```

### Modifying Segment Labels
Update the naming logic in Section 7 based on your domain knowledge.

---

## Performance Considerations

- **Data Size**: Optimized for datasets up to 100,000 rows
- **Processing Time**: Typically < 1 minute for 1,000 customers
- **Memory Usage**: ~100 MB for standard dataset
- **Visualization Quality**: High DPI (300 dpi) for print-ready output

---

## Troubleshooting

### Issue: Low Silhouette Scores
- **Cause**: Features may not be well-separated
- **Solution**: Review/modify clustering features or data quality

### Issue: Flat Elbow Curve
- **Cause**: Homogeneous customer base
- **Solution**: Increase K range or add more features

### Issue: Unbalanced Segments
- **Cause**: Data distribution
- **Solution**: Review segment labels and consider re-aggregation

### Issue: Memory Error
- **Cause**: Large dataset
- **Solution**: Sample data or increase available RAM

---

## Future Enhancements

1. **Hierarchical Clustering**: Compare with agglomerative clustering
2. **RFM Segmentation**: Dedicated RFM scoring system
3. **Customer Lifetime Value (CLV)**: Predict future value
4. **Anomaly Detection**: Identify unusual purchasing patterns
5. **Time-Series Forecasting**: Predict segment trends
6. **Interactive Dashboard**: Web-based visualization
7. **Automated Recommendations**: ML-based action recommendations
8. **A/B Testing**: Validate segment-specific strategies

---

## Contact & Support

For questions or improvements, refer to the well-documented code sections with detailed comments.

---

## License

This project is provided as-is for retail analysis purposes.

---

## Version History

- **v1.0** (Current)
  - K-means clustering implementation
  - 8 comprehensive visualizations
  - Automatic segment naming
  - Detailed segment profiling
  - RFM analysis integration

---

## References

### Clustering
- Scikit-learn K-means: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- Silhouette Analysis: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

### Customer Segmentation
- RFM Analysis: https://en.wikipedia.org/wiki/Customer_lifetime_value
- Clustering Evaluation: https://en.wikipedia.org/wiki/Silhouette_(clustering)

---

**Generated**: April 2026  
**Dataset Format**: CSV (Transaction-level retail data)  
**Analysis Type**: Customer Segmentation & Sales Analytics