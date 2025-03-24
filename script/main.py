# -*- coding: utf-8 -*-
"""
Created on 2025/03/24 09:55:43

@File -> main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 限制性立方样条
"""

from scipy.interpolate import LSQUnivariateSpline, splrep, splev
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from mod.dir_file_op import load_pickle

plt.rcParams["font.family"] = "SimSun"
plt.rcParams["axes.unicode_minus"] = False


class PiecewiseCubicSpline:
    """分段立方样条拟合"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.array(x).flatten()
        self.y = np.array(y).flatten()

        assert len(self.x) == len(self.y), ValueError("len(x) != len(y)")
        self.N = len(self.x)

        self.prepare_data()

    def _add_noise_to_x(self):
        """按照x的标准差添加噪声"""
        x_std = np.std(self.x)
        self.x += 1e-6 * x_std * np.random.randn(self.N)

    def _autoset_knot_points(self):
        """按照x的分位数自动自动设置分割点"""
        self.auto_knot_points = np.quantile(self.x, np.arange(0.1, 1, 0.1))

    def prepare_data(self):
        self._add_noise_to_x()
        self._autoset_knot_points()

    def bootstrap_fit(self, k: int = 3, bt_rounds: int = 100, knots: np.ndarray = None):
        """bootstrap拟合"""
        knots = self.auto_knot_points if knots is None else knots

        # 如果knots有重复值，则返回None
        if len(knots) != len(np.unique(knots)):
            return None

        spline_models = []
        for i in range(bt_rounds):
            idxs = np.random.choice(np.arange(self.N), self.N, replace=True)
            x_bt = self.x[idxs]
            y_bt = self.y[idxs]

            # 升序排列
            idxs = np.argsort(x_bt)
            x_bt = x_bt[idxs]
            y_bt = y_bt[idxs]

            # 拟合样条
            t_bt = np.random.choice(knots, len(knots)-3, replace=False)
            t_bt = np.sort(t_bt)
            spline_model = LSQUnivariateSpline(x_bt, y_bt, t=t_bt, k=k)
            spline_models.append(spline_model)
        return spline_models


if __name__ == "__main__":
    # 载入测试数据
    data_dict = load_pickle(f"{BASE_DIR}/data/test.pkl")

    x_col = "上甑平均蒸汽量"
    y_col = "糟醅水分"
    arr = data_dict[x_col]
    x = arr[:, 0]
    y = arr[:, 1]

    # 绘制散点图
    plt.figure()
    plt.scatter(x, y, c="b", s=6, marker="o", label="data", alpha=0.1)
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    # ---- 限制性立方样条 ----------------------------------------------------------------------------

    self = PiecewiseCubicSpline(x, y)
    
    # 拟合
    spline_models = self.bootstrap_fit(k=2, bt_rounds=100)

    # 绘制拟合结果
    x_grids = np.linspace(np.min(x), np.max(x), 1000)
    y_grids = np.zeros((len(spline_models), len(x_grids)))

    # 绘制均值和置信区间
    for i, spline_model in enumerate(spline_models):
        y_grids[i] = spline_model(x_grids)

    y_mean = np.mean(y_grids, axis=0)
    y_std = np.std(y_grids, axis=0)
    y_upper = y_mean + 1.96 * y_std
    y_lower = y_mean - 1.96 * y_std

    plt.plot(x_grids, y_mean, c="r", lw=2, label="mean")
    plt.fill_between(x_grids, y_lower, y_upper, color="r", alpha=0.2, label="95% CI")

    plt.ylim([-50, 50])
    plt.show()



    