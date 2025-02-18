import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def create_marchi_interpolator():
    """根据正确的Marchi数据创建插值函数"""
    # 表格中的实际数据点
    ages = np.array([
        4.35, 3.92, 3.92, 3.85, 3.85, 3.80, 3.70, 3.58, 3.30, 3.30,
        3.22, 3.15, 0.80, 0.80, 0.375, 0.120, 0.109, 0.109, 0.025
    ])
    
    # MBA模型的N(1)值
    densities = np.array([
        2.018e-1, 6.648e-2, 2.509e-2, 1.931e-2, 2.672e-2, 1.832e-2,
        1.585e-2, 9.357e-3, 5.520e-3, 5.520e-3, 2.377e-3, 3.695e-3,
        1.337e-3, 1.343e-3, 7.655e-4, 2.195e-4, 3.401e-4, 1.712e-4,
        7.131e-5
    ])
    
    # 创建插值函数
    log_densities = np.log10(densities)
    
    # 创建两个插值器：一个从密度到年龄，一个从年龄到密度
    age_from_density = interp1d(log_densities, ages, 
                               bounds_error=False, fill_value='extrapolate')
    density_from_age = interp1d(ages, log_densities, 
                               bounds_error=False, fill_value='extrapolate')
    
    return age_from_density, density_from_age

def create_neukum_interpolator():
    """根据正确的Neukum数据创建插值函数"""
    # 表格中的实际数据点
    ages = np.array([
        4.35, 3.92, 3.92, 3.85, 3.85, 3.80, 3.70, 3.58, 3.30, 3.30,
        3.22, 3.15, 0.80, 0.80, 0.375, 0.120, 0.109, 0.109, 0.025
    ])
    
    # NEO模型的N(1)值
    densities = np.array([
        7.851e-1, 1.327e-1, 2.490e-2, 1.968e-2, 2.553e-2, 1.836e-2,
        1.579e-2, 9.300e-3, 5.468e-3, 5.468e-3, 2.335e-3, 3.683e-3,
        1.321e-3, 1.348e-3, 1.267e-3, 3.835e-4, 3.391e-4, 1.644e-4,
        6.970e-5
    ])
    
    # 创建插值函数
    log_densities = np.log10(densities)
    
    age_from_density = interp1d(log_densities, ages, 
                               bounds_error=False, fill_value='extrapolate')
    density_from_age = interp1d(ages, log_densities, 
                               bounds_error=False, fill_value='extrapolate')
    
    return age_from_density, density_from_age

def calculate_age_marchi(craters_df, region_bounds, area_km2):
    """使用Marchi新拟合公式计算年龄"""
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
        # 使用新的Marchi拟合公式
        # N(1) = 1.208e-08 * (exp(3.825*t) - 1) + -2.469e-04*t
        a = 1.208e-08
        b = 3.825
        c = -2.469e-04
        
        # 使用数值方法求解方程
        def equation(t):
            return a * (np.exp(b * t) - 1) + c * t - N1
        
        # 在0-4.5 Ga范围内寻找解
        t = np.linspace(0, 4.5, 1000)
        N1_values = a * (np.exp(b * t) - 1) + c * t
        
        # 找到最接近的解
        idx = np.argmin(np.abs(N1_values - N1))
        age = t[idx]
        
        # 确保年龄在合理范围内
        age = np.clip(age, 0, 4.5)
        
        return age
    except:
        return None

def calculate_age_both_models(craters_df, region_bounds, area_km2):
    """使用两种模型计算年龄"""
    x1, y1, x2, y2 = region_bounds
    region_craters = craters_df[
        (craters_df['X1'] >= x1) & (craters_df['X1'] < x2) &
        (craters_df['Y1'] >= y1) & (craters_df['Y1'] < y2)
    ]
    
    D_ref = 1.0  # 1km参考直径
    craters_gt_1km = len(region_craters[region_craters['Diameter_KM'] >= D_ref])
    N1 = craters_gt_1km / area_km2
    
    if N1 == 0:
        return None, None
    
    try:
        # 使用两种模型计算年龄
        marchi_age_interpolator, _ = create_marchi_interpolator()
        neukum_age_interpolator, _ = create_neukum_interpolator()
        
        marchi_age = marchi_age_interpolator(np.log10(N1))
        neukum_age = neukum_age_interpolator(np.log10(N1))
        
        # 确保年龄在合理范围内
        marchi_age = np.clip(marchi_age, 0, 4.35)
        neukum_age = np.clip(neukum_age, 0, 4.35)
        
        return marchi_age, neukum_age
    except:
        return None, None

# 读取数据
craters_df = pd.read_csv('t/641t/craters_report.csv')

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
    marchi_age, neukum_age = calculate_age_both_models(craters_df, region, area_per_region)
    
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
        'Marchi_Age_Years': f"{marchi_age:.2f} Ga" if marchi_age else "Insufficient data",
        
    })

# 计算整个表面的年龄
whole_surface_age = calculate_age_both_models(craters_df, (0, 0, 4000, 4000), 200 * 200)
results.append({
    'Region': 5,
    'Position': "Whole Surface",
    'X1': 0,
    'Y1': 0,
    'X2': 4000,
    'Y2': 4000,
    'Marchi_Age_Years': f"{whole_surface_age[0]:.2f} Ga" if whole_surface_age[0] else "Insufficient data",
    
})

# 创建DataFrame并保存为CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Age\\1\\1_M.csv', index=False)
print("Results saved to 2_M.csv")

# 打印平均年龄
valid_ages = [float(r['Marchi_Age_Years'].split()[0]) for r in results if r['Marchi_Age_Years'] != "Insufficient data"]
if valid_ages:
    mean_age = np.mean(valid_ages)
    print(f"\nMean age of the surface (Marchi model): {mean_age:.2f} Ga")

# valid_ages = [float(r['Neukum_Age_Years'].split()[0]) for r in results if r['Neukum_Age_Years'] != "Insufficient data"]
# if valid_ages:
#     mean_age = np.mean(valid_ages)
#     print(f"\nMean age of the surface (Neukum model): {mean_age:.2f} Ga")
