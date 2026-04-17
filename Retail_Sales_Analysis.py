import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

df = pd.read_csv('retail_sales_dataset.csv')
print("="*80)
print("DATA PREPROCESSING")
print("="*80)

# Data Cleaning
print(f"\nInitial dataset shape: {df.shape}")
df.dropna(inplace=True)

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)
print(f"After cleaning: {df.shape}")
print(f"Data types:\n{df.dtypes}")

sales_col = 'Total Amount'
product_col = 'Product Category'

# ============================================================================
# SECTION 2: CUSTOMER SEGMENTATION - DATA AGGREGATION
# ============================================================================

print("\n" + "="*80)
print("CUSTOMER SEGMENTATION - AGGREGATING CUSTOMER METRICS")
print("="*80)

# Create customer-level aggregated features
customer_segments = df.groupby('Customer ID').agg({
    'Total Amount': ['sum', 'mean', 'count'],
    'Age': 'first',
    'Gender': 'first',
    'Date': ['min', 'max'],
    'Quantity': 'sum'
}).reset_index()

# Flatten column names
customer_segments.columns = ['Customer_ID', 'Total_Spend', 'Avg_Purchase', 'Purchase_Count',
                               'Age', 'Gender', 'First_Purchase', 'Last_Purchase', 'Total_Quantity']

# Calculate Days Since Last Purchase (Recency)
max_date = df['Date'].max()
customer_segments['Recency'] = (max_date - customer_segments['Last_Purchase']).dt.days
customer_segments['Customer_Lifetime_Days'] = (customer_segments['Last_Purchase'] - 
                                               customer_segments['First_Purchase']).dt.days

print(f"\nCustomer segments created: {len(customer_segments)} unique customers")
print(f"\nCustomer Metrics Summary:")
print(customer_segments[['Total_Spend', 'Avg_Purchase', 'Purchase_Count', 'Age', 'Recency']].describe())

# ============================================================================
# SECTION 3: FEATURE ENGINEERING FOR CLUSTERING
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING FOR CLUSTERING")
print("="*80)

# Select features for clustering (RFM + AOV + Demographics)
clustering_features = ['Total_Spend', 'Avg_Purchase', 'Purchase_Count', 'Recency', 'Age']

# Create a copy for clustering
X = customer_segments[clustering_features].copy()

print(f"\nFeatures selected for clustering: {clustering_features}")
print(f"\nBefore standardization:\n{X.describe()}")

# Handle any remaining NaN values
X = X.fillna(X.mean())

# Standardize features (essential for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nFeatures standardized using StandardScaler")

# ============================================================================
# SECTION 4: OPTIMAL K DETERMINATION (ELBOW METHOD + SILHOUETTE)
# ============================================================================

print("\n" + "="*80)
print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
print("="*80)

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_scores[-1]:.3f}")

# Optimal K is typically where silhouette score is highest
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k} (based on Silhouette Score)")

# ============================================================================
# SECTION 5: K-MEANS CLUSTERING WITH OPTIMAL K
# ============================================================================

print("\n" + "="*80)
print(f"PERFORMING K-MEANS CLUSTERING WITH K={optimal_k}")
print("="*80)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_segments['Segment'] = kmeans.fit_predict(X_scaled)

print(f"\nClustering completed successfully")
print(f"Silhouette Score: {silhouette_score(X_scaled, customer_segments['Segment']):.3f}")
print(f"\nSegment distribution:")
print(customer_segments['Segment'].value_counts().sort_index())

# ============================================================================
# SECTION 6: SEGMENT CHARACTERIZATION AND ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SEGMENT CHARACTERISTICS AND DETAILED ANALYSIS")
print("="*80)

segment_profiles = []

for segment_id in sorted(customer_segments['Segment'].unique()):
    segment_data = customer_segments[customer_segments['Segment'] == segment_id]
    
    profile = {
        'Segment': segment_id,
        'Size': len(segment_data),
        'Pct_of_Customers': f"{(len(segment_data)/len(customer_segments)*100):.1f}%",
        'Avg_Total_Spend': f"${segment_data['Total_Spend'].mean():.2f}",
        'Avg_Purchase_Value': f"${segment_data['Avg_Purchase'].mean():.2f}",
        'Avg_Purchase_Frequency': f"{segment_data['Purchase_Count'].mean():.1f}",
        'Avg_Age': f"{segment_data['Age'].mean():.1f}",
        'Avg_Recency_Days': f"{segment_data['Recency'].mean():.1f}",
        'Male_Pct': f"{(segment_data['Gender']=='Male').sum()/len(segment_data)*100:.1f}%",
        'Female_Pct': f"{(segment_data['Gender']=='Female').sum()/len(segment_data)*100:.1f}%",
        'Total_Revenue': f"${segment_data['Total_Spend'].sum():.2f}"
    }
    segment_profiles.append(profile)
    
    print(f"\n{'─'*80}")
    print(f"SEGMENT {segment_id} - Customer Profile")
    print(f"{'─'*80}")
    for key, value in profile.items():
        print(f"{key:.<35} {value:>20}")

# Create summary dataframe
segment_summary = pd.DataFrame(segment_profiles)

# ============================================================================
# SECTION 7: ASSIGN DESCRIPTIVE LABELS TO SEGMENTS
# ============================================================================

print("\n" + "="*80)
print("SEGMENT NAMING AND INTERPRETATIONS")
print("="*80)

# Calculate metrics for naming
segment_metrics = customer_segments.groupby('Segment').agg({
    'Total_Spend': 'mean',
    'Purchase_Count': 'mean',
    'Recency': 'mean',
    'Age': 'mean'
}).reset_index()

segment_names = {}
interpretations = {}

# Define segment labels based on characteristics
avg_spend = segment_metrics['Total_Spend'].mean()
avg_frequency = segment_metrics['Purchase_Count'].mean()

for idx, row in segment_metrics.iterrows():
    segment_id = row['Segment']
    spend = row['Total_Spend']
    frequency = row['Purchase_Count']
    recency = row['Recency']
    
    # Naming logic
    if spend > avg_spend and frequency > avg_frequency:
        name = "VIP Loyalists"
        interpretation = "High-value, frequent customers with strong loyalty"
    elif spend > avg_spend and recency < 30:
        name = "Engaged Premium Customers"
        interpretation = "Recently active, high-spend customers showing strong engagement"
    elif frequency > avg_frequency and spend <= avg_spend:
        name = "Budget-Conscious Regulars"
        interpretation = "Frequent but lower-spend customers, price-sensitive"
    elif recency > 100:
        name = "At-Risk/Dormant Customers"
        interpretation = "Inactive for extended period, need re-engagement campaigns"
    else:
        name = "Standard Customers"
        interpretation = "Average spending and purchasing patterns"
    
    segment_names[segment_id] = name
    interpretations[segment_id] = interpretation

print("\nSegment Descriptions:\n")
for segment_id in sorted(segment_names.keys()):
    print(f"Segment {segment_id}: {segment_names[segment_id]}")
    print(f"  └─ {interpretations[segment_id]}\n")

# ============================================================================
# SECTION 8: BASIC ANALYSIS (ORIGINAL FUNCTIONALITY)
# ============================================================================

print("\n" + "="*80)
print("OVERALL SALES ANALYSIS")
print("="*80)

total_sales = df[sales_col].sum()
average_sales = df[sales_col].mean()
average_daily_sales = df.groupby(df['Date'].dt.date)[sales_col].sum().mean()
identifying_top_products = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(10)

print(f'Total Sales: ${total_sales:.2f}')
print(f'Average Sales per Transaction: ${average_sales:.2f}')
print(f'Average Daily Sales: ${average_daily_sales:.2f}')
print('\nTop 10 Products by Sales:')
print(identifying_top_products)

# ============================================================================
# SECTION 9: DATA VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# 1. Elbow Method and Silhouette Score
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Inertia', fontsize=11, fontweight='bold')
axes[0].set_title('Elbow Method for Optimal K', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
axes[1].set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_Optimal_Clusters.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_Optimal_Clusters.png")
plt.close()

# 2. Top Products by Sales
fig, ax = plt.subplots(figsize=(12, 6))
identifying_top_products.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_title('Top 10 Product Categories by Sales', fontsize=13, fontweight='bold')
ax.set_xlabel('Product Category', fontsize=11, fontweight='bold')
ax.set_ylabel('Total Sales ($)', fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('02_Top_Products.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_Top_Products.png")
plt.close()

# 3. Sales Trend Over Time
fig, ax = plt.subplots(figsize=(14, 6))
sales_trend = df.groupby('Date')[sales_col].sum()
sales_trend.plot(ax=ax, linewidth=2, color='darkgreen')
ax.fill_between(sales_trend.index, sales_trend.values, alpha=0.3, color='lightgreen')
ax.set_title('Sales Trend Over Time', fontsize=13, fontweight='bold')
ax.set_xlabel('Date', fontsize=11, fontweight='bold')
ax.set_ylabel('Total Sales ($)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('03_Sales_Trend.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_Sales_Trend.png")
plt.close()

# 4. Segment Distribution (Pie Chart)
fig, ax = plt.subplots(figsize=(10, 8))
segment_counts = customer_segments['Segment'].value_counts().sort_index()
segment_labels = [f"Segment {i}\n({segment_names[i]})\n({segment_counts[i]} customers)" 
                  for i in sorted(segment_names.keys())]
colors = plt.cm.Set3(np.linspace(0, 1, len(segment_names)))
wedges, texts, autotexts = ax.pie(segment_counts, labels=segment_labels, autopct='%1.1f%%',
                                    colors=colors, startangle=90, textprops={'fontsize': 10})
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
ax.set_title('Customer Distribution Across Segments', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('04_Segment_Distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_Segment_Distribution.png")
plt.close()

# 5. Segment Characteristics - Boxplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Segment Characteristics by Feature', fontsize=14, fontweight='bold', y=1.00)

features_to_plot = ['Total_Spend', 'Avg_Purchase', 'Purchase_Count', 'Age', 'Recency', 'Customer_Lifetime_Days']
axes = axes.flatten()

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx]
    segment_data = [customer_segments[customer_segments['Segment']==i][feature].values 
                   for i in sorted(customer_segments['Segment'].unique())]
    bp = ax.boxplot(segment_data, labels=[f"S{i}" for i in sorted(customer_segments['Segment'].unique())],
                    patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel(feature.replace('_', ' '), fontsize=10, fontweight='bold')
    ax.set_xlabel('Segment', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('05_Segment_Characteristics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_Segment_Characteristics.png")
plt.close()

# 6. Scatter plot: Total Spend vs Purchase Frequency (colored by Segment)
fig, ax = plt.subplots(figsize=(12, 8))
for segment_id in sorted(customer_segments['Segment'].unique()):
    segment_data = customer_segments[customer_segments['Segment'] == segment_id]
    ax.scatter(segment_data['Purchase_Count'], segment_data['Total_Spend'],
              label=f"S{segment_id}: {segment_names[segment_id]}", s=150, alpha=0.7)

ax.set_xlabel('Purchase Frequency (Number of Purchases)', fontsize=11, fontweight='bold')
ax.set_ylabel('Total Spend ($)', fontsize=11, fontweight='bold')
ax.set_title('Customer Segments: Purchase Frequency vs Total Spend', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('06_Segment_Scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_Segment_Scatter.png")
plt.close()

# 7. Heatmap of segment metrics
fig, ax = plt.subplots(figsize=(12, 6))
segment_heatmap = customer_segments.groupby('Segment')[['Total_Spend', 'Avg_Purchase', 
                                                          'Purchase_Count', 'Age', 'Recency']].mean()
segment_heatmap_scaled = (segment_heatmap - segment_heatmap.min()) / (segment_heatmap.max() - segment_heatmap.min())
segment_heatmap_scaled.index = [f"S{i}\n{segment_names[i]}" for i in segment_heatmap_scaled.index]

sns.heatmap(segment_heatmap_scaled.T, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Normalized Value'},
           ax=ax, linewidths=0.5)
ax.set_title('Normalized Segment Profiles (Heatmap)', fontsize=13, fontweight='bold')
ax.set_xlabel('Segment', fontsize=11, fontweight='bold')
ax.set_ylabel('Metrics', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('07_Segment_Heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_Segment_Heatmap.png")
plt.close()

# 8. Gender distribution by segment
fig, ax = plt.subplots(figsize=(12, 6))
gender_segment = pd.crosstab(customer_segments['Segment'], customer_segments['Gender'], normalize='index') * 100
gender_segment.index = [f"S{i}\n{segment_names[i]}" for i in gender_segment.index]
gender_segment.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', width=0.7)
ax.set_title('Gender Distribution by Segment (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Segment', fontsize=11, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right')
ax.legend(title='Gender', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('08_Gender_by_Segment.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_Gender_by_Segment.png")
plt.close()

print("\n" + "="*80)
print("All visualizations saved successfully!")
print("="*80)

# ============================================================================
# SECTION 10: CLASS-BASED ANALYSIS (ORIGINAL)
# ============================================================================

class RetailSalesAnalysis:
    def __init__(self, data):
        self.data = data

    def total_sales(self):
        return self.data[sales_col].sum()

    def average_sales(self):
        return self.data[sales_col].mean()

    def average_daily_sales(self):
        return self.data.groupby(self.data['Date'].dt.date)[sales_col].sum().mean()

    def top_products(self, n=10):
        return self.data.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(n)

analysis = RetailSalesAnalysis(df)
print(f'Total Sales: ${analysis.total_sales():.2f}')
print(f'Average Sales: ${analysis.average_sales():.2f}')
print(f'Average Daily Sales: ${analysis.average_daily_sales():.2f}')
print('Top Products by Sales:')
print(analysis.top_products())





