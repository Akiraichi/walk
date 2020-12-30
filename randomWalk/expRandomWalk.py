import numpy as np
import exp_func

if True:
    """実験パラメータ"""
    # 最大時間発展T(t=0〜t=Tまで時間発展させる)及びプロットさせたい点t
    T = 10
    p_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # 初期確率振幅ベクトル[時間×距離(-T〜T)×2次元ベクトル]
    psy = np.zeros([T + 1, 2 * T + 1], dtype=float)
    psy[0, 0+T] = 1

    """プロット設定"""
    # plotしたい時間t。
    plot_t = T
    # 1つのグラフ内に複数のグラフを表示させる場合、一列（行）に何個グラフを表示させるようにするか？
    plot_graph_num_by_axis_row = 3
    plot_graph_num_by_axis_col = 4
    # グラフの横軸をそれぞれのグラフで最後のx軸の幅に固定するか？
    is_solid_axis_x = False
    # グラフの種類
    graph_type = "棒グラフ"

    """実験"""
    exp_code = "exp_5"
    exp_func.random_walk_1d_p_change(exp_code, T, p_list, psy, plot_t, plot_graph_num_by_axis_row,
                                     plot_graph_num_by_axis_col, graph_type=graph_type)
