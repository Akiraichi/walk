import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
import os
# グラデーション作成用
import matplotlib.cm as cm


def quantum_walk_1d(exp_code, T, P, Q, PSY, plots_t, plot_graph_num_by_axis_row, plot_graph_num_by_axis_col,
                    is_solid_axis_x=False, graph_type=""):
    """実験結果保存設定"""
    # ファイル保存フォルダ名
    save_folder_name = "img"
    # ファイル名＝実験コード-実験パラメータ-プロット設定
    folder_name = f"{save_folder_name}/{exp_code}"
    file_name = f"T={T},P={P},Q={Q},PSY[0,0]={PSY[0, 0]},{graph_type},plotする時刻t={plots_t},x軸の幅は固定しているか：{is_solid_axis_x}.png".replace(
        "\n", ",")
    # フォルダーがなければ作成
    os.makedirs(folder_name, exist_ok=True)

    """実験コード"""
    # 時間発展パート
    for t in range(T - 1):  # PSY[t+1,x]があるのでtの最大を-1しておく
        for x in range(-(T - 1), T - 1):  # PSY[t, x+1] + Q @ PSY[t, x-1]があるので-(T-1),T-1としておく
            PSY[t + 1, x] = P @ PSY[t, x + 1] + Q @ PSY[t, x - 1]

    """plotパート"""
    fig = plt.figure(figsize=(16,12),tight_layout=True)
    for i, plot_t in enumerate(plots_t):
        ax = fig.add_subplot(plot_graph_num_by_axis_row, plot_graph_num_by_axis_col, i + 1, xlabel="x", ylabel="p")
        # 横軸（距離）の設定。
        if is_solid_axis_x:
            x_axis = np.arange(-plots_t[-1], plots_t[-1] + 1, 1, int)
        else:
            # （実際は最大でも-plot_t〜plot_tにしかならないから）
            x_axis = np.arange(-plot_t, plot_t + 1, 1, int)
        y_axis = np.zeros(len(x_axis))
        for k, x in enumerate(x_axis):
            # L2ノルム（いわゆる距離と同じ）をとる。そして2乗
            y_axis[k] = np.linalg.norm(PSY[plot_t, x], ord=2)**2
        if graph_type == "棒グラフ":
            ax.bar(x_axis, y_axis, label=f"t={plot_t}")
        plt.legend()

    plt.savefig(f"{folder_name}/{file_name}", dpi=400, bbox_inches='tight')
    plt.show()
