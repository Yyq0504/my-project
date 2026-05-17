#稳健性测试

import pandas as pd
import numpy as np
import statsmodels.api as sm

# ========== 数据预处理（与主模型完全一致） ==========
df = pd.read_excel("最终完整数据.xlsx")
df.rename(columns={
    "创新技术水平（ R&D 经费 亿元）": "tech_rd",
    "电力碳排放因子（kgCO₂/kWh）": "Grid"
}, inplace=True)

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

for var in ["净减排率", "总碳排放", "PGDP", "tech_rd", "IS", "Grid"]:
    df[var] = winsorize_series(df[var])
    df["ln_总碳排放"] = np.log(df["总碳排放"])
    df["ln_PGDP"] = np.log(df["PGDP"])
    df["ln_Tech"] = np.log(df["tech_rd"])

# ========== 门槛检验函数 ==========
def check_threshold(data, name):
    print(f"稳健性检验：{name}")


    threshold_var = data["Grid"].values
    y = data["ln_总碳排放"].values
    x = data["净减排率"].values

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

    def f_stat(y, x, threshold_var, gamma):
        n = len(y)
        d = (threshold_var <= gamma).astype(int)
        X = np.column_stack([np.ones(n), x, x * d])
        X0 = np.column_stack([np.ones(n), x])
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        beta0 = np.linalg.inv(X0.T @ X0) @ X0.T @ y
        rss = np.sum((y - X @ beta)**2)
        rss0 = np.sum((y - X0 @ beta0)**2)
        return (rss0 - rss) / (rss / (n - X.shape[1]))

    candidates = np.percentile(threshold_var, np.arange(10, 91, 2))
    rss_dict = {g: calculate_rss(y, x, threshold_var, g) for g in candidates}
    best_gamma = min(rss_dict, key=rss_dict.get)
    f_true = f_stat(y, x, threshold_var, best_gamma)

    np.random.seed(123)
    f_boot = []
    X0 = np.column_stack([np.ones(len(y)), x])
    beta0 = np.linalg.inv(X0.T @ X0) @ X0.T @ y
    resid0 = y - X0 @ beta0
    for _ in range(300):
        resid_star = np.random.choice(resid0 - np.mean(resid0), size=len(y), replace=True)
        y_star = X0 @ beta0 + resid_star
        rss_star = {g: calculate_rss(y_star, x, threshold_var, g) for g in candidates}
        best_star = min(rss_star, key=rss_star.get)
        f_boot.append(f_stat(y_star, x, threshold_var, best_star))
    p_val = np.mean(np.array(f_boot) >= f_true)

    # 估计 β₁ β₂
    data["D_low"] = (data["Grid"] <= best_gamma).astype(int)
    data["D_high"] = (data["Grid"] > best_gamma).astype(int)
    data["Rate_low"] = data["净减排率"] * data["D_low"]
    data["Rate_high"] = data["净减排率"] * data["D_high"]

    formula = "ln_总碳排放 ~ Rate_low + Rate_high + ln_PGDP + IS + ln_Tech + C(年份)"
    model = sm.OLS.from_formula(formula, data=data).fit(cov_type='HC3')

    beta1 = model.params["Rate_low"]
    beta2 = model.params["Rate_high"]
    p1 = model.pvalues["Rate_low"]
    p2 = model.pvalues["Rate_high"]

    print(f"有效样本量：{len(data)}")
    print(f"最优门槛值：{best_gamma:.4f}")
    print(f"F 统计量：{f_true:.4f}")
    print(f"Bootstrap P 值：{p_val:.4f}")
    print(f"清洁区 β₁：{beta1:.4f} (P值：{p1:.4f})")
    print(f"煤电区 β₂：{beta2:.4f} (P值：{p2:.4f})")
    if p_val < 0.05:
        print("结论：存在显著门槛效应")
    else:
        print("结论：不存在显著门槛效应")

    return best_gamma, f_true, p_val, beta1, p1, beta2, p2

# ========== 运行三个检验 ==========
results = {}

# 1. 基准模型（全样本）
g1, f1, pv1, b1_1, pb1, b2_1, pb2 = check_threshold(df, "基准模型（全样本）")
results["基准模型"] = (g1, f1, pv1, b1_1, pb1, b2_1, pb2)

# 2. 剔除 2020-2022 年
df_no_covid = df[~df["年份"].isin([2020, 2021, 2022])].copy()
g2, f2, pv2, b1_2, pb1_2, b2_2, pb2_2 = check_threshold(df_no_covid, "剔除2020-2022年")
results["剔除2020-2022"] = (g2, f2, pv2, b1_2, pb1_2, b2_2, pb2_2)

# 3. 剔除 ln_Tech
df_no_tech = df.drop(columns=["ln_Tech"]).copy()
def check_threshold_no_tech(data, name):
    print(f"稳健性检验：{name}")

    threshold_var = data["Grid"].values
    y = data["ln_总碳排放"].values
    x = data["净减排率"].values

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

    def f_stat(y, x, threshold_var, gamma):
        n = len(y)
        d = (threshold_var <= gamma).astype(int)
        X = np.column_stack([np.ones(n), x, x * d])
        X0 = np.column_stack([np.ones(n), x])
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        beta0 = np.linalg.inv(X0.T @ X0) @ X0.T @ y
        rss = np.sum((y - X @ beta)**2)
        rss0 = np.sum((y - X0 @ beta0)**2)
        return (rss0 - rss) / (rss / (n - X.shape[1]))

    candidates = np.percentile(threshold_var, np.arange(10, 91, 2))
    rss_dict = {g: calculate_rss(y, x, threshold_var, g) for g in candidates}
    best_gamma = min(rss_dict, key=rss_dict.get)
    f_true = f_stat(y, x, threshold_var, best_gamma)

    np.random.seed(123)
    f_boot = []
    X0 = np.column_stack([np.ones(len(y)), x])
    beta0 = np.linalg.inv(X0.T @ X0) @ X0.T @ y
    resid0 = y - X0 @ beta0
    for _ in range(300):
        resid_star = np.random.choice(resid0 - np.mean(resid0), size=len(y), replace=True)
        y_star = X0 @ beta0 + resid_star
        rss_star = {g: calculate_rss(y_star, x, threshold_var, g) for g in candidates}
        best_star = min(rss_star, key=rss_star.get)
        f_boot.append(f_stat(y_star, x, threshold_var, best_star))
    p_val = np.mean(np.array(f_boot) >= f_true)

    data["D_low"] = (data["Grid"] <= best_gamma).astype(int)
    data["D_high"] = (data["Grid"] > best_gamma).astype(int)
    data["Rate_low"] = data["净减排率"] * data["D_low"]
    data["Rate_high"] = data["净减排率"] * data["D_high"]

    # 没有 ln_Tech
    formula = "ln_总碳排放 ~ Rate_low + Rate_high + ln_PGDP + IS  + C(年份)"
    model = sm.OLS.from_formula(formula, data=data).fit(cov_type='HC3')

    beta1 = model.params["Rate_low"]
    beta2 = model.params["Rate_high"]
    p1 = model.pvalues["Rate_low"]
    p2 = model.pvalues["Rate_high"]

    print(f"有效样本量：{len(data)}")
    print(f"最优门槛值：{best_gamma:.4f}")
    print(f"F 统计量：{f_true:.4f}")
    print(f"Bootstrap P 值：{p_val:.4f}")
    print(f"清洁区 β₁：{beta1:.4f} (P值：{p1:.4f})")
    print(f"煤电区 β₂：{beta2:.4f} (P值：{p2:.4f})")
    if p_val < 0.05:
        print("结论：存在显著门槛效应")
    else:
        print("结论：不存在显著门槛效应")

    return best_gamma, f_true, p_val, beta1, p1, beta2, p2

g3, f3, pv3, b1_3, pb1_3, b2_3, pb2_3 = check_threshold_no_tech(df_no_tech, "剔除ln_Tech")
results["剔除ln_Tech"] = (g3, f3, pv3, b1_3, pb1_3, b2_3, pb2_3)
'''
# ========== 汇总表 ==========
print("\n" + "="*80)
print("稳健性检验汇总（模型三，年份固定，含UR）")
print("="*80)
print(f"{'检验':<20} {'门槛值':<10} {'F统计量':<10} {'P值':<10} {'β₁(清洁区)':<16} {'P值(β₁)':<10} {'β₂(煤电区)':<16} {'P值(β₂)'}")
print("-"*100)
for name, (g, f, pv, b1, pb1, b2, pb2) in results.items():
    print(f"{name:<20} {g:<10.4f} {f:<10.2f} {pv:<10.4f} {b1:<16.4f} {pb1:<10.4f} {b2:<16.4f} {pb2:.4f}")
'''
