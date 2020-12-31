import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sympy
from mpl_toolkits.mplot3d import Axes3D


def save_setting(exp_code, exp_code_chapter, description):
    """実験結果保存設定"""
    # ファイル保存フォルダ名
    save_folder_name = "img"
    # ファイル名＝実験コード-実験パラメータ-プロット設定
    folder_name = f"{save_folder_name}/{exp_code}"
    # 図のタイトルネーム
    fig_title_name = f"{exp_code}-{exp_code_chapter}"
    file_name = f"{exp_code}-{exp_code_chapter}-{description}.png"
    # フォルダーがなければ作成
    os.makedirs(folder_name, exist_ok=True)
    return folder_name, fig_title_name, file_name


def do_plot(folder_name, fig_title_name, file_name, plots_t, plot_graph_num_by_axis_row,
            plot_graph_num_by_axis_col, is_solid_axis_x, graph_type, PSY):
    fig = plt.figure(figsize=(16, 12), tight_layout=True, dpi=800)
    fig.suptitle(fig_title_name)
    for i, plot_t in enumerate(plots_t):
        ax = fig.add_subplot(plot_graph_num_by_axis_row, plot_graph_num_by_axis_col, i + 1, xlabel="$x$", ylabel="$p$")
        # 横軸（距離）の設定。
        if is_solid_axis_x:
            x_axis = np.arange(-plots_t[-1], plots_t[-1] + 1, 1, int)
        else:
            # （実際は最大でも-plot_t〜plot_tにしかならないから）
            x_axis = np.arange(-plot_t, plot_t + 1, 1, int)
        y_axis = np.zeros(len(x_axis))
        for k, x in enumerate(x_axis):
            # L2ノルム（いわゆる距離と同じ）をとる。そして2乗
            y_axis[k] = np.linalg.norm(PSY[plot_t, x], ord=2) ** 2
        if graph_type == "棒グラフ":
            ax.bar(x_axis, y_axis, label=f"$t={plot_t}$")
        plt.legend()

    plt.savefig(f"{folder_name}/{file_name}", dpi=800, bbox_inches='tight')
    plt.show()


def do_plot_PSY_list(folder_name, fig_title_name, file_name, plot_t, plot_graph_num_by_axis_row,
                     plot_graph_num_by_axis_col, graph_type, PSY_list, theta_list, label=r"$\theta$"):
    fig = plt.figure(figsize=(16, 12), tight_layout=True, dpi=800)
    fig.suptitle(fig_title_name)
    for index, theta in enumerate(theta_list):
        ax = fig.add_subplot(plot_graph_num_by_axis_row, plot_graph_num_by_axis_col, index + 1, xlabel="$x$",
                             ylabel="$p$")
        # 横軸（距離）の設定。固定値のみ
        x_axis = np.arange(-plot_t, plot_t + 1, 1, int)
        y_axis = np.zeros(len(x_axis))
        for k, x in enumerate(x_axis):
            # L2ノルム（いわゆる距離と同じ）をとる。そして2乗
            y_axis[k] = np.linalg.norm(PSY_list[index][plot_t, x], ord=2) ** 2
        if graph_type == "棒グラフ":
            ax.bar(x_axis, y_axis, label=f"{label}$={theta}$")
        plt.legend()

    plt.savefig(f"{folder_name}/{file_name}", dpi=800, bbox_inches='tight')
    plt.show()


def quantum_walk_1d(T, P, Q, PSY):
    # 時間発展パート
    for t in range(T):
        for x in range(-T, T + 1):  # T=2とすると、-2,-1,0,1,2 で、領域の確保数が2(T+1)+1なので、リストのインデックスは5,6,0,1,2となる
            PSY[t + 1, x] = P @ PSY[t, x + 1] + Q @ PSY[t, x - 1]
    return PSY


def quantum_walk_1d_theta_change(T, P_list, Q_list, PSY, PSY_init, theta_list):
    # 時間発展パート
    PSY_list = []
    for P, Q in zip(P_list, Q_list):
        for t in range(T):
            for x in range(-T, T + 1):  # T=2とすると、-2,-1,0,1,2 で、領域の確保数が2(T+1)+1なので、リストのインデックスは5,6,0,1,2となる
                PSY[t + 1, x] = P @ PSY[t, x + 1] + Q @ PSY[t, x - 1]
        PSY_list.append(PSY)
        # psyをリセットさせる
        PSY = np.zeros([T + 1, 2 * (T + 1) + 1, 2], dtype=np.complex128)
        PSY[0, 0] = PSY_init
    return PSY_list


def quantum_walk_1d_2phase(T, P_list, Q_list, PSY):
    # 時間発展パート
    for t in range(T):
        for x in range(-T, T + 1):  # T=2とすると、-2,-1,0,1,2 で、領域の確保数が2(T+1)+1なので、リストのインデックスは5,6,0,1,2となる
            if t % 2 == 0:
                PSY[t + 1, x] = P_list[0] @ PSY[t, x + 1] + Q_list[0] @ PSY[t, x - 1]
            else:
                PSY[t + 1, x] = P_list[1] @ PSY[t, x + 1] + Q_list[1] @ PSY[t, x - 1]
    return PSY


def quantum_walk_1d_2phase_theta_2_change(T, P_list, Q_list, P_1, Q_1, PSY, PSY_init, theta_list):
    # 時間発展パート
    PSY_list = []
    for P, Q in zip(P_list, Q_list):
        for t in range(T):
            for x in range(-T, T + 1):  # T=2とすると、-2,-1,0,1,2 で、領域の確保数が2(T+1)+1なので、リストのインデックスは5,6,0,1,2となる
                if t % 2 == 0:
                    PSY[t + 1, x] = P_1 @ PSY[t, x + 1] + Q_1 @ PSY[t, x - 1]
                else:
                    PSY[t + 1, x] = P @ PSY[t, x + 1] + Q @ PSY[t, x - 1]
        PSY_list.append(PSY)
        # psyをリセットさせる
        PSY = np.zeros([T + 1, 2 * (T + 1) + 1, 2], dtype=np.complex128)
        PSY[0, 0] = PSY_init

    return PSY_list


def quantum_walk_1d_3phase(T, P_list, Q_list, PSY):
    # 時間発展パート
    for t in range(T):
        for x in range(-T, T + 1):  # T=2とすると、-2,-1,0,1,2 で、領域の確保数が2(T+1)+1なので、リストのインデックスは5,6,0,1,2となる
            if t % 3 == 0:
                PSY[t + 1, x] = P_list[0] @ PSY[t, x + 1] + Q_list[0] @ PSY[t, x - 1]
            elif t % 3 == 1:
                PSY[t + 1, x] = P_list[1] @ PSY[t, x + 1] + Q_list[1] @ PSY[t, x - 1]
            elif t % 3 == 2:
                PSY[t + 1, x] = P_list[2] @ PSY[t, x + 1] + Q_list[2] @ PSY[t, x - 1]
            else:
                print("something is wrong.")
                return
    return PSY


def quantum_walk_1d_3phase_theta_change(T, P_list, Q_list, P_3, Q_3, PSY, PSY_init, theta_list):
    # 時間発展パート
    PSY_list = []
    for P_1, Q_1 in zip(P_list, Q_list):
        P_2 = P_1
        Q_2 = Q_1
        for t in range(T):
            for x in range(-T, T + 1):  # T=2とすると、-2,-1,0,1,2 で、領域の確保数が2(T+1)+1なので、リストのインデックスは5,6,0,1,2となる
                if t % 3 == 0:
                    PSY[t + 1, x] = P_1 @ PSY[t, x + 1] + Q_1 @ PSY[t, x - 1]
                elif t % 3 == 1:
                    PSY[t + 1, x] = P_2 @ PSY[t, x + 1] + Q_2 @ PSY[t, x - 1]
                elif t % 3 == 2:
                    PSY[t + 1, x] = P_3 @ PSY[t, x + 1] + Q_3 @ PSY[t, x - 1]
                else:
                    print("something is wrong.")
                    return
        PSY_list.append(PSY)
        # psyをリセットさせる
        PSY = np.zeros([T + 1, 2 * (T + 1) + 1, 2], dtype=np.complex128)
        PSY[0, 0] = PSY_init
    return PSY_list


# 以下から初期確率振幅ベクトルを内包している関数となっている
def quantum_walk_1d_depend_x(T, P_0, Q_0, P_x, Q_x, PSY_init):
    # 初期確率振幅ベクトル
    PSY = np.zeros([T + 1, 2 * (T + 1) + 1, 2], dtype=np.complex128)
    PSY[0, 0] = PSY_init

    # 時間発展パート
    for t in range(T):
        for x in range(-T, T + 1):
            if x == 1:
                PSY[t + 1, x] = P_x @ PSY[t, x + 1] + Q_0 @ PSY[t, x - 1]
            elif x == -1:
                PSY[t + 1, x] = P_0 @ PSY[t, x + 1] + Q_x @ PSY[t, x - 1]
            else:
                PSY[t + 1, x] = P_x @ PSY[t, x + 1] + Q_x @ PSY[t, x - 1]
    return PSY


def quantum_walk_1d_depend_x_omega_change(T, P_0_list, Q_0_list, P_x, Q_x, PSY_init):
    PSY_list = []
    for P_0, Q_0 in zip(P_0_list, Q_0_list):
        # 初期確率振幅ベクトル
        PSY = np.zeros([T + 1, 2 * (T + 1) + 1, 2], dtype=np.complex128)
        PSY[0, 0] = PSY_init
        # 時間発展パート
        for t in range(T):
            for x in range(-T, T + 1):  # T=2とすると、-2,-1,0,1,2 で、領域の確保数が2(T+1)+1なので、リストのインデックスは5,6,0,1,2となる
                if x == 1:
                    PSY[t + 1, x] = P_x @ PSY[t, x + 1] + Q_0 @ PSY[t, x - 1]
                elif x == -1:
                    PSY[t + 1, x] = P_0 @ PSY[t, x + 1] + Q_x @ PSY[t, x - 1]
                else:
                    PSY[t + 1, x] = P_x @ PSY[t, x + 1] + Q_x @ PSY[t, x - 1]
        PSY_list.append(PSY)

    return PSY_list


def quantum_walk_1d_3component(T, P, Q, R, PSY_init):
    # 初期確率振幅ベクトル
    PSY = np.zeros([T + 1, 2 * (T + 1) + 1, 3], dtype=np.complex128)
    PSY[0, 0] = PSY_init

    # 時間発展パート
    for t in range(T):
        for x in range(-T, T + 1):  # T=2とすると、-2,-1,0,1,2 で、領域の確保数が2(T+1)+1なので、リストのインデックスは5,6,0,1,2となる
            PSY[t + 1, x] = P @ PSY[t, x + 1] + Q @ PSY[t, x] + R @ PSY[t, x - 1]
    return PSY


def quantum_walk_1d_3component_theta_change(T, P_list, Q_list, R_list, PSY_init):
    PSY_list = []
    for P, Q, R in zip(P_list, Q_list, R_list):
        # 初期確率振幅ベクトル
        PSY = np.zeros([T + 1, 2 * (T + 1) + 1, 3], dtype=np.complex128)
        PSY[0, 0] = PSY_init
        # 時間発展パート
        for t in range(T):
            for x in range(-T, T + 1):  # T=2とすると、-2,-1,0,1,2 で、領域の確保数が2(T+1)+1なので、リストのインデックスは5,6,0,1,2となる
                PSY[t + 1, x] = P @ PSY[t, x + 1] + Q @ PSY[t, x] + R @ PSY[t, x - 1]
        PSY_list.append(PSY)
    return PSY_list


def quantum_walk_1d_4component(T, P, Q, PSY_init):
    # 初期確率振幅ベクトル
    PSY = np.zeros([T + 1, 2 * (T + 1) + 1, 4], dtype=np.complex128)
    PSY[0, 0] = PSY_init

    # 時間発展パート
    for t in range(T):
        for x in range(-T, T + 1):
            PSY[t + 1, x] = P @ PSY[t, x + 1] + Q @ PSY[t, x - 1]
    return PSY


def quantum_walk_1d_4component_theta_change(T, P_list, Q_list, PSY_init):
    PSY_list = []
    for P, Q in zip(P_list, Q_list):
        # 初期確率振幅ベクトル
        PSY = np.zeros([T + 1, 2 * (T + 1) + 1, 4], dtype=np.complex128)
        PSY[0, 0] = PSY_init
        # 時間発展パート
        for t in range(T):
            for x in range(-T, T + 1):  # T=2とすると、-2,-1,0,1,2 で、領域の確保数が2(T+1)+1なので、リストのインデックスは5,6,0,1,2となる
                PSY[t + 1, x] = P @ PSY[t, x + 1] + Q @ PSY[t, x - 1]
        PSY_list.append(PSY)
    return PSY_list


def quantum_walk_2d(T, P, Q, R, S, PSY_init):
    # 初期確率振幅ベクトル
    PSY = np.zeros([T + 1, 2 * (T + 1) + 1, 2 * (T + 1) + 1, 4], dtype=np.complex128)
    PSY[0, 0] = PSY_init

    # 時間発展パート
    for t in range(T):
        for x in range(-T, T + 1):
            for y in range(-T, T + 1):
                PSY[t + 1, x, y] = P @ PSY[t, x + 1, y] + Q @ PSY[t, x - 1, y] + R @ PSY[t, x, y + 1] + S @ PSY[
                    t, x, y - 1]
    return PSY


# def do_plot_3d(PSY, plots_t, plot_graph_num_by_axis_row, plot_graph_num_by_axis_col):
#     # setup the figure and axes
#     fig = plt.figure(figsize=(16, 9), dpi=800)
#     # ax1 = fig.add_subplot(121, projection='3d')
#
#     for i, plot_t in enumerate(plots_t):
#         ax = fig.add_subplot(plot_graph_num_by_axis_row, plot_graph_num_by_axis_col, i + 1, projection="3d")
#         # 軸の設定。
#         x_axis = np.arange(-plot_t, plot_t + 1, 1, int)
#         y_axis = np.arange(-plot_t, plot_t + 1, 1, int)
#         _xx, _yy = np.meshgrid(x_axis, y_axis)
#         _x, _y = _xx.ravel(), _yy.ravel()
#
#         z_axis = np.zeros(len(_x))
#         count = 0
#         for y in y_axis:
#             for x in x_axis:
#                 # L2ノルム（いわゆる距離と同じ）をとる。そして2乗
#                 z_axis[count] = np.linalg.norm(PSY[plot_t, x, y], ord=2) ** 2
#                 count += 1
#         bottom = np.zeros_like(z_axis)
#         width = depth = 1
#         ax.bar3d(_x, _y, bottom, width, depth, z_axis, shade=True)
#         # plt.legend()
#
#     plt.show()

# def do_plot_3d(PSY, plots_t, plot_graph_num_by_axis_row, plot_graph_num_by_axis_col):
#     # setup the figure and axes
#     fig = plt.figure(figsize=(16, 9), dpi=800)
#
#     for i, plot_t in enumerate(plots_t):
#         ax = fig.add_subplot(plot_graph_num_by_axis_row, plot_graph_num_by_axis_col, i + 1, projection="3d")
#         # 軸の設定。
#         x_axis = np.arange(-plot_t, plot_t + 1, 1, int)
#         y_axis = np.arange(-plot_t, plot_t + 1, 1, int)
#
#         x_list = np.zeros(len(x_axis) * len(y_axis))
#         y_list = np.zeros(len(x_axis) * len(y_axis))
#         z_list = np.zeros(len(x_axis) * len(y_axis))
#
#         count = 0
#         for y in y_axis:
#             for x in x_axis:
#                 # L2ノルム（いわゆる距離と同じ）をとる。そして2乗
#                 z_list[count] = np.linalg.norm(PSY[plot_t, x, y], ord=2) ** 2
#                 x_list[count] = x
#                 y_list[count] = y
#                 count += 1
#
#         bottom = np.zeros_like(z_list)
#         width = depth = 1
#         ax.bar3d(x_list, y_list, bottom, width, depth, z_list, shade=True)
#         # plt.legend()

#     plt.show()
#
# def do_plot_3d(PSY, plots_t, plot_graph_num_by_axis_row, plot_graph_num_by_axis_col):
#     # setup the figure and axes
#     fig = plt.figure(figsize=(16, 9), dpi=800)
#
#     for i, plot_t in enumerate(plots_t):
#         ax = fig.add_subplot(plot_graph_num_by_axis_row, plot_graph_num_by_axis_col, i + 1, projection="3d")
#         # 軸の設定。
#         x_axis = np.arange(-plot_t, plot_t + 1, 1, int)
#         y_axis = np.arange(-plot_t, plot_t + 1, 1, int)
#
#         X, Y = np.meshgrid(x_axis, y_axis)
#
#         Z = np.zeros(len(x_axis) * len(y_axis))
#         count = 0
#         for y in y_axis:
#             for x in x_axis:
#                 # L2ノルム（いわゆる距離と同じ）をとる。そして2乗
#                 Z[count] = np.linalg.norm(PSY[plot_t, x, y], ord=2) ** 2
#                 count += 1
#
#         ax.plot_surface(X, Y, Z.reshape(X.shape), cmap='bwr', linewidth=0)
#         # plt.legend()
#
#     plt.show()
