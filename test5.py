#LR


import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("最终完整数据.xlsx")

df.rename(columns={
    "创新技术水平（ R&D 经费 亿元）": "tech_rd",
    "电力碳排放因子（kgCO₂/kWh）": "Grid"               
}, inplace=True)

# 变量
df["总碳排放"] = df["燃油碳排放"] + df["新能源碳排放"]
df["净减排率"] = df["Contribution"] / df["燃油碳排放"]
df["ln_总碳排放"] = np.log(df["总碳排放"])
df["ln_PGDP"] = np.log(df["PGDP"])
df["ln_Tech"] = np.log(df["tech_rd"])

# 5% 缩尾
def winsorize_series(s, limits=(0.05, 0.05)):
    lower = s.quantile(limits[0])
    upper = s.quantile(1 - limits[1])
    return s.clip(lower, upper)

for var in ["净减排率", "总碳排放", "PGDP", "tech_rd", "IS", "Grid"]:  # 加上 UR
    df[var] = winsorize_series(df[var])
    df["ln_总碳排放"] = np.log(df["总碳排放"])
    df["ln_PGDP"] = np.log(df["PGDP"])
    df["ln_Tech"] = np.log(df["tech_rd"])

# 门槛变量
threshold_var = df["Grid"].values
y = df["ln_总碳排放"].values
x = df["净减排率"].values

# ========== 门槛效应 LR 计算 ==========
def calculate_rss(y, x, threshold_var, gamma):
    n = len(y)
    d = (threshold_var <= gamma).astype(int)
    X = np.column_stack([np.ones(n), x, x * d])
    try:
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        resid = y - X @ beta
        return np.sum(resid**2)
    except:
        return np.inf

threshold_candidates = np.percentile(threshold_var, np.arange(10, 91, 2))
rss_dict = {g: calculate_rss(y, x, threshold_var, g) for g in threshold_candidates}
best_gamma = min(rss_dict, key=rss_dict.get)
min_rss = rss_dict[best_gamma]

lr_stats = []
for gamma in threshold_candidates:
    rss = rss_dict[gamma]
    lr = (rss - min_rss) / (min_rss / len(y))
    lr_stats.append(lr)

lr_critical = 7.35

print("="*60)
print("门槛效应检验（年份固定效应）")
print("="*60)
print(f"有效样本量：{len(df)}")
print(f"最优门槛值 γ = {best_gamma:.4f}")
print(f"最小 RSS = {min_rss:.4f}")
print(f"5% 临界值 = {lr_critical}")
print(f"最小 LR 统计量 = {min(lr_stats):.4f}")

if min(lr_stats) <= lr_critical:
    print("结论：存在显著门槛效应")
else:
    print("结论：不存在显著门槛效应")

# ========== 估计 β₁ 和 β₂ ==========
print("\n" + "="*60)
print("门槛回归结果（β₁ 和 β₂）")
print("="*60)

df["D_low"] = (df["Grid"] <= best_gamma).astype(int)
df["D_high"] = (df["Grid"] > best_gamma).astype(int)
df["Rate_low"] = df["净减排率"] * df["D_low"]
df["Rate_high"] = df["净减排率"] * df["D_high"]


formula = "ln_总碳排放 ~ Rate_low + Rate_high + ln_PGDP + IS + ln_Tech  + C(年份)"
model = sm.OLS.from_formula(formula, data=df).fit(cov_type='HC3')

beta1 = model.params["Rate_low"]
beta2 = model.params["Rate_high"]
p1 = model.pvalues["Rate_low"]
p2 = model.pvalues["Rate_high"]

print(f"清洁电网区（Grid ≤ {best_gamma:.3f}）系数 β₁ = {beta1:.4f} (P值 = {p1:.4f})")
print(f"煤电电网区（Grid > {best_gamma:.3f}）系数 β₂ = {beta2:.4f} (P值 = {p2:.4f})")

# ========== 画 LR 趋势图 ==========
plt.figure(figsize=(8, 5), dpi=120)
plt.plot(threshold_candidates, lr_stats, color='crimson', linewidth=2)
plt.axhline(y=lr_critical, color='black', linestyle='--', label=f'5% 临界值 = {lr_critical}')
plt.axvline(x=best_gamma, color='gray', linestyle=':', label=f'最优门槛值 = {best_gamma:.3f}')

plt.xlabel("电网碳排放因子 (Grid)", fontsize=11)
plt.ylabel("LR 统计量", fontsize=11)
plt.title("门槛效应检验：LR 统计量趋势图（年份固定效应）", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("LR趋势图_年份固定.png", dpi=300)
plt.show()
