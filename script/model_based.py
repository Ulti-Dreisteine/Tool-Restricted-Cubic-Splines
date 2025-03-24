# -*- coding: utf-8 -*-
"""
Created on 2025/03/24 12:02:01

@File -> model_based.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于模型的拟合
"""

# from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import numpy as np
import warnings
import sys
import os

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from mod.dir_file_op import load_pickle

plt.rcParams["font.family"] = "SimSun"
plt.rcParams["axes.unicode_minus"] = False


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
    plt.scatter(x, y, c="b", s=6, marker="o", label="data", alpha=0.01)
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    # ---- 拟合 -------------------------------------------------------------------------------------

    bt_rounds = 100
    models = {}

    for i in range(bt_rounds):
        idxs = np.random.choice(len(x), 100, replace=True)
        x_bt = x[idxs].reshape(-1, 1)
        y_bt = y[idxs]

        # 设置kernel
        
        kernel = RBF(length_scale=(np.max(x_bt) - np.min(x_bt))/10) + WhiteKernel() + ConstantKernel()
        model = GaussianProcessRegressor(kernel=kernel)

        model.fit(x_bt, y_bt)

        models[i] = model

    # ---- 绘制拟合结果 ------------------------------------------------------------------------------

    x_grids = np.linspace(np.min(x), np.max(x), 1000).reshape(-1, 1)
    y_grids = np.zeros((bt_rounds, len(x_grids)))

    for i in range(bt_rounds):
        model = models[i]
        y_grids[i] = model.predict(x_grids)

    # 绘制均值和置信区间
    y_mean = np.mean(y_grids, axis=0)
    y_std = np.std(y_grids, axis=0)

    y_upper = y_mean + 1.96 * y_std
    y_lower = y_mean - 1.96 * y_std

    plt.plot(x_grids, y_mean, c="r", lw=2, label="mean")
    plt.fill_between(x_grids.flatten(), y_upper, y_lower, color="r", alpha=0.2, label="95% CI")
    plt.ylim([-10, 10])
    plt.show()