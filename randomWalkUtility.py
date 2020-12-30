import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
import os
# グラデーション作成用
import matplotlib.cm as cm

def random_walk_1d(exp_code, T, p, q, psy, plots_t, plot_graph_num_by_axis_row, plot_graph_num_by_axis_col,
                   is_solid_axis_x=False, graph_type=""):
    """実験結果保存設定"""
    # ファイル保存フォルダ名
    save_folder_name = "img"
    # ファイル名＝実験コード-実験パラメータ-プロット設定
    folder_name = f"{save_folder_name}/{exp_code}"
    # 図のタイトルネーム
    fig_title_name = f"p={p},q={q},psy[0,0]={psy[0, 0]}"
    file_name = f"P={p},Q={q},PSY[0,0]={psy[0, 0]},{graph_type},plotする時刻t={plots_t}.png".replace(
        "\n", ",")
    # フォルダーがなければ作成
    os.makedirs(folder_name, exist_ok=True)

    """実験コード"""
    # 時間発展パート
    for t in range(T - 1):  # psy[t+1,x]があるのでtの最大を-1しておく
        for x in range(-(T - 1), T - 1):  # psy[t, x+1] + q * psy[t, x-1]があるので-(T-1),T-1としておく
            psy[t + 1, x] = p * psy[t, x + 1] + q * psy[t, x - 1]

    """plotパート"""
    fig = plt.figure(figsize=(16, 12), tight_layout=True)
    fig.suptitle(fig_title_name)
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
            y_axis[k] = psy[plot_t, x]
        if graph_type == "棒グラフ":
            ax.bar(x_axis, y_axis, label=f"t={plot_t}")
        plt.legend()

    plt.savefig(f"{folder_name}/{file_name}", dpi=400, bbox_inches='tight')
    plt.show()


def random_walk_1d_p_change(exp_code, T, p_list, psy, plots_t, plot_graph_num_by_axis_row,
                            plot_graph_num_by_axis_col, graph_type=""):
    """実験結果保存設定"""
    # ファイル保存フォルダ名
    save_folder_name = "img"
    # ファイル名＝実験コード-実験パラメータ-プロット設定
    folder_name = f"{save_folder_name}/{exp_code}"
    # 図のタイトルネーム
    fig_title_name = f"p={p_list},psy[0,0]={psy[0, 0]}"
    file_name = f"P={p_list},PSY[0,0]={psy[0, 0]},{graph_type},plotする時刻t={plots_t}.png".replace(
        "\n", ",")
    # フォルダーがなければ作成
    os.makedirs(folder_name, exist_ok=True)

    """実験コード"""
    # 時間発展パート
    psy_list = []
    for p in p_list:
        q = 1 - p
        for t in range(T):
            for x in range(-T, T + 1):  # -T <= x <= T
                psy[t + 1, x + T] = p * psy[t, (x + T + 1) % (2 * T + 1)] + q * psy[
                    t, x + T - 1]  # 周期的境界条件を適用させている。（0の左はT, Tの右は0）
        psy_list.append(psy[plots_t])
        # psyをリセットさせる
        psy = np.zeros([T + 1, 2 * T + 1], dtype=float)
        psy[0, 0 + T] = 1

    """plotパート"""
    fig = plt.figure(figsize=(16, 12), tight_layout=True)
    fig.suptitle(fig_title_name)
    for index, p in enumerate(p_list):
        ax = fig.add_subplot(plot_graph_num_by_axis_row, plot_graph_num_by_axis_col, index + 1, xlabel="x", ylabel="p")
        # 横軸（距離）の設定。固定値のみ
        x_axis = np.arange(-plots_t, plots_t + 1, 1, int)
        y_axis = psy_list[index]
        if graph_type == "棒グラフ":
            ax.bar(x_axis, y_axis, label=f"p={p}")
        plt.legend()

    plt.savefig(f"{folder_name}/{file_name}", dpi=400, bbox_inches='tight')
    plt.show()
