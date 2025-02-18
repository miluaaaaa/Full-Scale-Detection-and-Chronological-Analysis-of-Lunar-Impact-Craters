import pandas as pd
import numpy as np
from math import log10, exp

def calculate_age_neukum1983(craters_df, region_bounds, area_km2):
    """使用Neukum 1983模型计算年龄"""
    x1, y1, x2, y2 = region_bounds
    region_craters = craters_df[
        (craters_df['X1'] >= x1) & (craters_df['X1'] < x2) &
        (craters_df['Y1'] >= y1) & (craters_df['Y1'] < y2)
    ]
    
    D_ref = 1.0  # 1km参考直径
    craters_gt_1km = len(region_craters[region_craters['Diameter_KM'] >= D_ref])
    N1 = craters_gt_1km / area_km2
    
    if N1 == 0:
        return None
    
    # 使用Neukum 1983模型: N(1) = 5.44·10⁻¹⁴[exp(6.93T) - 1] + 8.38·10⁻⁴T
    try:
        def equation(t):
            return 5.44e-14 * (np.exp(6.93*t) - 1) + 8.38e-4 * t - N1
        
        # 使用Newton方法求解，初始猜测值为1 Ga
        t = 1.0
        for _ in range(100):  # 最多迭代100次
            f = equation(t)
            if abs(f) < 1e-10:  # 收敛条件
                break
            df = 5.44e-14 * 6.93 * np.exp(6.93*t) + 8.38e-4  # 导数
            t = t - f/df
            if t < 0:  # 确保年龄为正
                t = 0.1
        
        return t
    except:
        return None

# 读取数据
craters_df = pd.read_csv('t/644t/craters_report.csv')

# 定义4个区域
regions = [
    (0, 0, 2000, 2000),      # 左上
    (2000, 0, 4000, 2000),   # 右上
    (0, 2000, 2000, 4000),   # 左下
    (2000, 2000, 4000, 4000) # 右下
]

# 每个区域面积
area_per_region = 100 * 100  # 100km × 100km

# 创建结果列表
results = []
for i, region in enumerate(regions):
    x1, y1, x2, y2 = region
    age = calculate_age_neukum1983(craters_df, region, area_per_region)
    
    # 区域描述
    if i == 0:
        position = "Left Upper"
    elif i == 1:
        position = "Right Upper"
    elif i == 2:
        position = "Left Lower"
    else:
        position = "Right Lower"
    
    results.append({
        'Region': i + 1,
        'Position': position,
        'X1': x1,
        'Y1': y1,
        'X2': x2,
        'Y2': y2,
        'Age_Years': f"{age:.2f} Ga" if age else "Insufficient data"
    })

# 计算整个表面的年龄
whole_surface_age = calculate_age_neukum1983(craters_df, (0, 0, 4000, 4000), 200 * 200)
results.append({
    'Region': 5,
    'Position': "Whole Surface",
    'X1': 0,
    'Y1': 0,
    'X2': 4000,
    'Y2': 4000,
    'Age_Years': f"{whole_surface_age:.2f} Ga" if whole_surface_age else "Insufficient data"
})

# 创建DataFrame并保存为CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Age\\4\\4_N1983.csv', index=False)
print("Results saved to 2_N1983.csv")

# 打印平均年龄
valid_ages = [float(r['Age_Years'].split()[0]) for r in results if r['Age_Years'] != "Insufficient data"]
if valid_ages:
    mean_age = np.mean(valid_ages)
    print(f"\nMean age of the surface: {mean_age:.2f} Ga") 