# ======================================================================================
# VISUALIZATION CODE - Thêm vào cuối notebook
# ======================================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\n>>> [VISUALIZATION] Creating Charts...")

# ======================================================================================
# 1. FEATURE IMPORTANCE
# ======================================================================================
print("\n1. Feature Importance Chart")

# Define feature names
feature_names = [
    'als_score', 'itemcf_score', 'combined_score',
    'popularity', 'log_popularity',
    'user_purchase_count', 'log_user_purchase_count',
    'has_als_score', 'has_cf_score',
    'user_item_frequency', 'log_user_item_frequency',
    'is_repeat_buyer', 'is_frequent_buyer',
    'recency_score', 'days_since_last', 'log_days_since_last',
    'days_since_first', 'repurchase_rate'
]

# Get feature importance
importance = model_lgb.feature_importance(importance_type='gain')
total_importance = importance.sum()
importance_pct = (importance / total_importance) * 100

# Create DataFrame
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance,
    'importance_pct': importance_pct
}).sort_values('importance', ascending=False)

print(feature_importance_df)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance_df)))
bars = ax.barh(range(len(feature_importance_df)), 
               feature_importance_df['importance_pct'],
               color=colors)

# Add value labels
for i, (idx, row) in enumerate(feature_importance_df.iterrows()):
    ax.text(row['importance_pct'] + 0.5, i, 
            f"{row['importance_pct']:.1f}%", 
            va='center', fontsize=9)

ax.set_yticks(range(len(feature_importance_df)))
ax.set_yticklabels(feature_importance_df['feature'], fontsize=10)
ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
ax.set_title('LightGBM Feature Importance (Gain)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Saved: feature_importance.png")

# ======================================================================================
# 2. TOP 5 FEATURES PIE CHART
# ======================================================================================
print("\n2. Top 5 Features Pie Chart")

top5 = feature_importance_df.head(5).copy()
others_pct = feature_importance_df.iloc[5:]['importance_pct'].sum()
top5 = pd.concat([top5, pd.DataFrame({
    'feature': ['Others'],
    'importance_pct': [others_pct]
})])

fig, ax = plt.subplots(figsize=(10, 8))
colors_pie = plt.cm.Set3(range(len(top5)))
wedges, texts, autotexts = ax.pie(
    top5['importance_pct'], 
    labels=top5['feature'],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors_pie,
    textprops={'fontsize': 11}
)

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

ax.set_title('Top 5 Most Important Features', 
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('top5_features_pie.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Saved: top5_features_pie.png")

# ======================================================================================
# 3. FEATURE GROUPS COMPARISON
# ======================================================================================
print("\n3. Feature Groups Comparison")

# Group features
feature_groups = {
    'Model Scores': ['als_score', 'itemcf_score', 'combined_score'],
    'Popularity': ['popularity', 'log_popularity'],
    'User Behavior': ['user_purchase_count', 'log_user_purchase_count'],
    'Binary Flags': ['has_als_score', 'has_cf_score', 'is_repeat_buyer', 'is_frequent_buyer'],
    'Frequency': ['user_item_frequency', 'log_user_item_frequency'],
    'Recency': ['recency_score', 'days_since_last', 'log_days_since_last', 'days_since_first'],
    'Item Stats': ['repurchase_rate']
}

group_importance = {}
for group_name, features in feature_groups.items():
    group_imp = feature_importance_df[
        feature_importance_df['feature'].isin(features)
    ]['importance_pct'].sum()
    group_importance[group_name] = group_imp

group_df = pd.DataFrame(list(group_importance.items()), 
                        columns=['Group', 'Importance'])
group_df = group_df.sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors_group = plt.cm.Spectral(np.linspace(0.2, 0.8, len(group_df)))
bars = ax.bar(range(len(group_df)), group_df['Importance'], color=colors_group)

# Add value labels
for i, (idx, row) in enumerate(group_df.iterrows()):
    ax.text(i, row['Importance'] + 1, 
            f"{row['Importance']:.1f}%", 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(range(len(group_df)))
ax.set_xticklabels(group_df['Group'], rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Total Importance (%)', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance by Groups', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_groups.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Saved: feature_groups.png")

# ======================================================================================
# 4. TRAINING DATA STATISTICS
# ======================================================================================
print("\n4. Training Data Statistics")

stats_data = {
    'Metric': [
        'Total Samples',
        'Total Groups',
        'Positive Samples',
        'Negative Samples',
        'Positive Rate (%)',
        'Avg Samples/Group'
    ],
    'Value': [
        f"{len(X_train):,}",
        f"{len(ltr_groups):,}",
        f"{y_train.sum():,.0f}",
        f"{(len(y_train) - y_train.sum()):,.0f}",
        f"{y_train.mean()*100:.2f}%",
        f"{len(X_train)/len(ltr_groups):.1f}"
    ]
}

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=[[m, v] for m, v in zip(stats_data['Metric'], stats_data['Value'])],
                colLabels=['Metric', 'Value'],
                cellLoc='left',
                loc='center',
                colWidths=[0.6, 0.4])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header
for i in range(2):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(stats_data['Metric']) + 1):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')
        else:
            table[(i, j)].set_facecolor('#F2F2F2')

ax.set_title('Training Data Statistics', 
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('training_stats.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Saved: training_stats.png")

# ======================================================================================
# 5. LABEL DISTRIBUTION
# ======================================================================================
print("\n5. Label Distribution")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
labels = ['Negative (0)', 'Positive (1)']
counts = [(y_train == 0).sum(), (y_train == 1).sum()]
colors_label = ['#FF6B6B', '#4ECDC4']

bars = ax1.bar(labels, counts, color=colors_label, alpha=0.8, edgecolor='black', linewidth=1.5)
for i, (label, count) in enumerate(zip(labels, counts)):
    percentage = (count / len(y_train)) * 100
    ax1.text(i, count + len(y_train)*0.02, 
             f"{count:,}\n({percentage:.2f}%)", 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Label Distribution (Bar Chart)', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Pie chart
ax2.pie(counts, labels=labels, autopct='%1.2f%%', 
        colors=colors_label, startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        explode=(0.05, 0.05))
ax2.set_title('Label Distribution (Pie Chart)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Saved: label_distribution.png")

# ======================================================================================
# 6. FEATURE CORRELATION HEATMAP (Top 10 Features)
# ======================================================================================
print("\n6. Feature Correlation Heatmap")

# Get top 10 features
top10_features = feature_importance_df.head(10)['feature'].tolist()
top10_indices = [feature_names.index(f) for f in top10_features]

# Create correlation matrix
X_top10 = X_train[:, top10_indices]
corr_matrix = np.corrcoef(X_top10.T)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation', fontsize=12, fontweight='bold')

# Set ticks
ax.set_xticks(range(len(top10_features)))
ax.set_yticks(range(len(top10_features)))
ax.set_xticklabels(top10_features, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(top10_features, fontsize=10)

# Add correlation values
for i in range(len(top10_features)):
    for j in range(len(top10_features)):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9)

ax.set_title('Feature Correlation Heatmap (Top 10 Features)', 
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Saved: feature_correlation.png")

# ======================================================================================
# 7. SUMMARY REPORT
# ======================================================================================
print("\n" + "="*60)
print("📊 VISUALIZATION SUMMARY")
print("="*60)
print(f"\n✅ Created 6 visualizations:")
print(f"   1. feature_importance.png - Feature importance bar chart")
print(f"   2. top5_features_pie.png - Top 5 features pie chart")
print(f"   3. feature_groups.png - Feature groups comparison")
print(f"   4. training_stats.png - Training data statistics table")
print(f"   5. label_distribution.png - Label distribution charts")
print(f"   6. feature_correlation.png - Feature correlation heatmap")

print(f"\n📈 TOP 5 MOST IMPORTANT FEATURES:")
for i, (idx, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):
    print(f"   {i}. {row['feature']:25s} - {row['importance_pct']:5.2f}%")

print(f"\n📊 FEATURE GROUPS RANKING:")
for i, (idx, row) in enumerate(group_df.iterrows(), 1):
    print(f"   {i}. {row['Group']:20s} - {row['Importance']:5.2f}%")

print("\n" + "="*60)
