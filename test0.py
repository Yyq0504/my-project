#偏回归散点图


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("最终完整数据.xlsx")

df.rename(columns={                              # 改成 UR
    "创新技术水平（ R&D 经费 亿元）": "tech_rd",
}, inplace=True)

df["总碳排放"] = df["燃油碳排放"] + df["新能源碳排放"]
df["净减排率"] = df["Contribution"] / df["燃油碳排放"]

# 缩尾处理（5%）
def winsorize_series(s, limits=(0.05, 0.05)):
    lower = s.quantile(limits[0])
    upper = s.quantile(1 - limits[1])
    return s.clip(lower, upper)

for var in ["净减排率", "总碳排放", "PGDP", "tech_rd", "IS"]:   # 加上 UR
    df[var] = winsorize_series(df[var])

# 对数
df["ln_总碳排放"] = np.log(df["总碳排放"])
df["ln_PGDP"] = np.log(df["PGDP"])
df["ln_Tech"] = np.log(df["tech_rd"])

# ========== 模型二（年份固定效应） ==========
model = sm.OLS.from_formula(
    "ln_总碳排放 ~ 净减排率 + ln_PGDP + IS + ln_Tech  + C(年份)",   
    data=df
).fit()

print("模型二回归结果（年份固定效应）")
print(model.summary())

print("核心结论")
print(f"净减排率系数 = {model.params['净减排率']:.4f}")
print(f"P值 = {model.pvalues['净减排率']:.4f}")

# ========== 偏回归图 ==========
resid_y = sm.OLS.from_formula(
    "ln_总碳排放 ~ ln_PGDP + IS + ln_Tech + C(年份)",   # 加上 UR
    data=df
).fit().resid

resid_x = sm.OLS.from_formula(
    "净减排率 ~ ln_PGDP + IS + ln_Tech  + C(年份)",      # 加上 UR
    data=df
).fit().resid

plt.figure(figsize=(6, 4), dpi=100)
plt.scatter(resid_x, resid_y, alpha=0.7, color='steelblue', edgecolor='white', s=60)
fit = np.polyfit(resid_x, resid_y, 1)
plt.plot(resid_x, np.poly1d(fit)(resid_x), color='darkred', lw=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.axvline(0, color='black', linestyle='--', alpha=0.3)
plt.xlabel("净减排率 (残差)")
plt.ylabel("ln(总碳排放) (残差)")
plt.title("偏回归图（年份固定效应）")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("偏回归图_年份固定.png", dpi=300)
plt.show()


