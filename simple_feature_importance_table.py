# ======================================================================================
# SIMPLE FEATURE IMPORTANCE TABLE - Chỉ cần 1 cell này sau khi có feature_importance_df
# ======================================================================================

import matplotlib.pyplot as plt
import numpy as np

# Tính phần trăm
total_importance = feature_importance_df['importance'].sum()
feature_importance_df['importance_pct'] = (feature_importance_df['importance'] / total_importance) * 100

# In ra console
print("\n" + "="*70)
print("📊 FEATURE IMPORTANCE RANKING")
print("="*70)
print(f"\n{'Rank':<6} {'Feature':<30} {'Importance':<15} {'Percentage':<10}")
print("-"*70)
for i, (idx, row) in enumerate(feature_importance_df.iterrows(), 1):
    print(f"{i:<6} {row['feature']:<30} {row['importance']:<15,.0f} {row['importance_pct']:>6.2f}%")
print("="*70)

# Vẽ bảng đẹp
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('tight')
ax.axis('off')

# Chuẩn bị data cho bảng
table_data = []
for i, (idx, row) in enumerate(feature_importance_df.iterrows(), 1):
    table_data.append([
        str(i),
        row['feature'],
        f"{row['importance']:,.0f}",
        f"{row['importance_pct']:.2f}%"
    ])

# Tạo bảng
table = ax.table(
    cellText=table_data,
    colLabels=['Rank', 'Feature Name', 'Importance (Gain)', 'Percentage (%)'],
    cellLoc='left',
    loc='center',
    colWidths=[0.08, 0.35, 0.25, 0.15]
)

# Style bảng
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Style header (hàng đầu tiên)
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#2E75B6')
    cell.set_text_props(weight='bold', color='white', fontsize=11)
    cell.set_height(0.08)

# Style các hàng dữ liệu
for i in range(1, len(table_data) + 1):
    # Màu nền xen kẽ
    if i % 2 == 0:
        bg_color = '#F2F2F2'
    else:
        bg_color = '#FFFFFF'
    
    # Highlight top 5
    if i <= 5:
        bg_color = '#FFF2CC'  # Màu vàng nhạt cho top 5
    
    for j in range(4):
        cell = table[(i, j)]
        cell.set_facecolor(bg_color)
        
        # Bold cho top 5
        if i <= 5:
            cell.set_text_props(weight='bold')
        
        # Căn phải cho cột số
        if j in [2, 3]:
            cell.set_text_props(ha='right')

# Thêm chú thích
fig.text(0.5, 0.95, 'LightGBM Feature Importance Ranking', 
         ha='center', fontsize=16, fontweight='bold')
fig.text(0.5, 0.92, '(Top 5 features highlighted in yellow)', 
         ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig('feature_importance_table.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\n✅ Saved: feature_importance_table.png")

# Tóm tắt
print("\n" + "="*70)
print("📈 SUMMARY")
print("="*70)
print(f"\n🥇 TOP 5 MOST IMPORTANT FEATURES:")
for i, (idx, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):
    print(f"   {i}. {row['feature']:<30} {row['importance_pct']:>6.2f}%")

print(f"\n📊 Top 5 combined importance: {feature_importance_df.head(5)['importance_pct'].sum():.2f}%")
print(f"📊 Top 10 combined importance: {feature_importance_df.head(10)['importance_pct'].sum():.2f}%")
print("="*70)
