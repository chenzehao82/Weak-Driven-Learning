import numpy as np
import pandas as pd
# ============================
def compute_adaboost_sampling_weights(entropy_df: pd.DataFrame,
                             alpha: float,
                             beta: float,
                             gamma: float) -> pd.DataFrame:
    """
    根据 entropy_0 / entropy_1 计算：
        delta_entropy = entropy_1 - entropy_0
        sampling_weight = p_i (归一化概率)
    返回带 'sampling_weight' 列的 DataFrame
    """
    df = entropy_df.copy()
    H0 = df["entropy_0"].to_numpy(dtype=np.float64)
    H1 = df["entropy_1"].to_numpy(dtype=np.float64)
    dH = H1 - H0
    df["delta_entropy"] = dH

    # Z1, Z2, Z3
    Z1 = H0.sum()
    Z2 = (-dH[dH < 0]).sum()
    Z3 = (dH[dH > 0]).sum()
    eps = 1e-12
    Z1 = Z1 if Z1 != 0 else eps
    Z2 = Z2 if Z2 != 0 else eps
    Z3 = Z3 if Z3 != 0 else eps

    p1 = alpha * (H0 / Z1)
    p2 = beta * ((-dH) / Z2) * (dH < 0)
    p3 = gamma * (dH / Z3) * (dH > 0)

    p = p1 + p2 + p3
    p = np.maximum(p, 0.0)
    p_sum = p.sum()
    p = p / (p_sum if p_sum > 0 else eps)

    df["sampling_weight"] = p
    return df


def compute_sampling_weights_brownboost_style(
    entropy_df: pd.DataFrame,
    alpha: float = 1.0,        # 当前难度权重
    beta: float = 1.0,         # 进步程度权重
    gamma: float = 1.0,        # 时间惩罚权重
    easy_quantile: float = 0.2,
    hard_quantile: float = 0.8,
    patience: int = 2,         # 对“极难”题的容忍轮数
    easy_patience: int = 2,    # 对“极简”题的容忍轮数
    lambda_time: float = 1.0,  # 极难题时间惩罚强度
    lambda_easy: float = 1.0,  # 极简题惩罚强度
) -> pd.DataFrame:
    """
    要求 entropy_df 至少包含列:
      - 'idx'
      - 'entropy_0', 'entropy_1', 'entropy_2'
    返回: 在原 DataFrame 上新增一列 'sampling_weight'
    """
    df = entropy_df.copy()

    e0 = df["entropy_0"].to_numpy()
    e1 = df["entropy_1"].to_numpy()
    e2 = df["entropy_2"].to_numpy()
    eps = 1e-8

    # -------- 1) 定义 easy / hard 阈值（基于当前 entropy_2 分布） --------
    easy_th = np.quantile(e2, easy_quantile)
    hard_th = np.quantile(e2, hard_quantile)

    # -------- 2) 当前难度因子 d_i：中~偏难区间权重最高 --------
    difficulty = (e2 - easy_th) / (hard_th - easy_th + eps)
    difficulty = np.clip(difficulty, 0.0, 1.0) + eps  # 避免全 0

    # -------- 3) 进步因子 trend_i：考虑 entropy_0 -> entropy_2 的变化 --------
    improve = e0 - e2  # >0 说明在变简单
    improve_std = np.std(improve) + eps
    k_trend = 0.5  # 你可以调大调小
    trend = 1.0 + k_trend * (improve / improve_std)
    trend = np.clip(trend, 0.1, 10.0)

    # -------- 4) BrownBoost 风格的时间惩罚：长期极难的题被“放弃” --------
    hard_rounds = (
        (e0 > hard_th).astype(int)
        + (e1 > hard_th).astype(int)
        + (e2 > hard_th).astype(int)
    )
    time_penalty = np.exp(-lambda_time * np.maximum(0, hard_rounds - patience))

    # -------- 5) 长期简单惩罚：多轮都很简单的题降权 --------
    easy_rounds = (
        (e0 < easy_th).astype(int)
        + (e1 < easy_th).astype(int)
        + (e2 < easy_th).astype(int)
    )
    easy_penalty = np.exp(-lambda_easy * np.maximum(0, easy_rounds - easy_patience))

    # -------- 6) 综合权重 --------
    raw_w = (difficulty ** alpha) * (trend ** beta) * (time_penalty ** gamma) * (easy_penalty)

    # 避免全 0
    raw_w = np.maximum(raw_w, eps)
    raw_w = raw_w / raw_w.sum()

    df["sampling_weight"] = raw_w
    return df
