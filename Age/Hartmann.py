import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def create_hartmann_interpolator():
    """根据表格数据创建Hartmann模型的插值函数"""
    # 表格数据
    ages = np.array([4.2, 4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2, 
                     2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8])
    
    # 新模型的N(1)值（注意：表中的数字如2.23(-1)表示2.23×10⁻¹）
    densities = np.array([
        2.23e-1, 4.62e-2, 1.88e-2, 6.97e-3, 4.00e-3, 2.86e-3, 2.19e-3,
        1.79e-3, 1.53e-3, 1.34e-3, 1.18e-3, 1.02e-3, 8.80e-4, 7.45e-4,
        6.19e-4, 5.03e-4, 3.96e-4, 2.98e-4
    ])
    
    # 创建插值函数
    # 使用对数插值，因为数据跨越多个数量级
    log_densities = np.log10(densities)
    
    # 创建两个插值器：一个从密度到年龄，一个从年龄到密度
    age_from_density = interp1d(log_densities, ages, 
                               bounds_error=False, fill_value='extrapolate')
    density_from_age = interp1d(ages, log_densities, 
                               bounds_error=False, fill_value='extrapolate')
    
    return age_from_density, density_from_age

def calculate_age_hartmann(craters_df, region_bounds, area_km2):
    """使用Hartmann新拟合公式计算年龄"""
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
        # 使用新的Hartmann拟合公式
        # N(1) = 2.490e-08 * (exp(3.554*t) - 1) + 2.267e-04*t
        a = 2.490e-08
        b = 3.554
        c = 2.267e-04
        
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
    age = calculate_age_hartmann(craters_df, region, area_per_region)
    
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
whole_surface_age = calculate_age_hartmann(craters_df, (0, 0, 4000, 4000), 200 * 200)
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
results_df.to_csv('Age\\1\\1_H.csv', index=False)
print("Results saved to 2_H_table.csv")

# 打印平均年龄
valid_ages = [float(r['Age_Years'].split()[0]) for r in results if r['Age_Years'] != "Insufficient data"]
if valid_ages:
    mean_age = np.mean(valid_ages)
    print(f"\nMean age of the surface: {mean_age:.2f} Ga")
