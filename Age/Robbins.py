import pandas as pd
import numpy as np
from scipy.optimize import fsolve

def robbins_model(T, version='quadratic'):
    """
    Robbins的年代学模型
    T: 年龄（Ga）
    version: 'linear' 或 'quadratic'
    返回: N(1)值
    """
    if version == 'linear':
        # 指数-线性版本
        alpha = 9.83e-31
        beta = 16.7
        gamma = 1.19e-3
        return alpha * (np.exp(beta * T) - 1) + gamma * T
    else:
        # 指数-二次版本（推荐）
        alpha = 7.26e-41
        beta = 22.6
        gamma = 9.49e-4
        delta = 1.88e-4
        return alpha * (np.exp(beta * T) - 1) + gamma * T + delta * T**2

def calculate_age_robbins(craters_df, region_bounds, area_km2, version='quadratic'):
    """使用Robbins模型计算年龄"""
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
    
    try:
        # 定义要求解的方程
        def equation(T):
            return robbins_model(T, version) - N1
        
        # 使用数值方法求解年龄
        # 初始猜测值为2 Ga
        age = fsolve(equation, 2.0)[0]
        
        # 确保年龄在合理范围内
        age = np.clip(age, 0, 4.5)
        
        return age
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
    age = calculate_age_robbins(craters_df, region, area_per_region)
    
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
whole_surface_age = calculate_age_robbins(craters_df, (0, 0, 4000, 4000), 200 * 200)
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
results_df.to_csv('Age\\4\\4_R.csv', index=False)
print("Results saved to 2_R.csv")

# 打印平均年龄
valid_ages = [float(r['Age_Years'].split()[0]) for r in results if r['Age_Years'] != "Insufficient data"]
if valid_ages:
    mean_age = np.mean(valid_ages)
    print(f"\nMean age of the surface: {mean_age:.2f} Ga") 