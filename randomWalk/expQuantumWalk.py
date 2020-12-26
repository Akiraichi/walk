import numpy as np
import math
from randomWalk import exp_func

"""
実験名：実験6
実験内容（できるだけ分かりやすく文章で説明）：
"""

if True:
    """実験パラメータ"""
    # 最大時間発展T(t=0〜t=Tまで時間発展させる。t=Tを求めるにはT+1まで計算する必要があるため内部ではT+1まで計算している)
    T = 500
    # P,Q
    # P = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [0, 0]], dtype=np.complex128)
    # Q = np.array([[0, 0], [1 / np.sqrt(2), -1 / np.sqrt(2)]], dtype=np.complex128)
    P = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    Q = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    # 初期確率振幅ベクトル[時間×距離(-T〜T)×2次元ベクトル] x+1やx-1があるので、2つ余分に領域をとっておく
    PSY = np.zeros([T+1, 2 * (T + 1) + 1, 2], dtype=np.complex128)
    # PSY[0, 0] = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])
    PSY[0, 0] = np.array([1, 0])

    """プロット設定"""
    # plotしたい時間t。リストで指定。複数指定の場合の例：plots_t = [0, 1, 2, 3]
    plots_t = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500]
    # 1つのグラフ内に複数のグラフを表示させる場合、一列（行）に何個グラフを表示させるようにするか？
    # plot_graph_num_by_axis_row = int(math.ceil(len(plots_t) ** 0.5))
    # plot_graph_num_by_axis_col = int(math.ceil(len(plots_t) ** 0.5))
    plot_graph_num_by_axis_row = 3
    plot_graph_num_by_axis_col = 4
    # グラフの横軸をそれぞれのグラフで最後のx軸の幅に固定するか？
    is_solid_axis_x = False
    # グラフの種類
    graph_type = "棒グラフ"

    """実験"""
    exp_code = "exp_6"
    exp_func.quantum_walk_1d(exp_code, T, P, Q, PSY, plots_t, plot_graph_num_by_axis_row, plot_graph_num_by_axis_col,
                             is_solid_axis_x, graph_type)
