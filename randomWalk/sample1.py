import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
import os
# グラデーション作成用
import matplotlib.cm as cm
from randomWalk import exp_func


def random_walk(p):
    """
    確率pで１次元上で正の方向へ進む
    :param p: 確率pで正の方向へ進む
    :return:
    """

    if np.random.rand() > p:
        return 1  # １が出たとき１進む
    else:
        return -1  # 0が出たとき１戻る


# def random_walk_2d(position, is_enable_return=False):
#     random_num = np.random.rand()
#     if is_enable_return:
#
#     else:
#         if random_num < 0.25:
#             position.x += 1
#         elif random_num < 0.5:
#             position.y += 1
#         elif random_num < 0.75:
#             position.x -= 1
#         else:
#             position.y -= 1
#
#     return position


# def test_2(n, is_enable_return=False, filename):
#     """
#         次元：2次元
#         形状：2次元、グリッドにした
#         ルール：一度来た場所にも戻れる
#         確率：等確率でx,yに+1, -1
#         表現：カラーバーとグラデーション
#         n:　時間（100なら100回ウォークする）
#         is_enable_return: 元きた道に戻れるか（一段前の道に戻れるか？）
#         return: 時間と位置で作ったdf
#     """
#     postion = Position()
#
#     postion.x = 0
#     postion.y = 0
#     postion.before = None
#     position_list = []
#     for i in range(n):
#         # x_list.append(x)
#         # y_list.append(y)
#         x, y = random_walk_2d(postion, is_enable_return)
#
#         # plt.scatter(x, y, color=cm.jet(i / n))
#     #
#     cm = plt.cm.get_cmap('RdYlBu')
#     # 点(x,y)が持つ値。今回はtime
#     z = range(n)
#     plt.scatter(x_list, y_list, c=z, cmap=cm)
#     plt.colorbar()
#     plt.legend()
#     plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
#     plt.show()


def test_1(n, p, column):
    """
    次元：1次元
    形状：1次元なので、ただの線
    確率：50％で左（-1）、50％で右（＋1）
    表現：y軸：距離、x軸：時間

    n:　時間（100なら100回ウォークする）
    p: 正の方向へ進む確率
    column: 列名
    return: 時間と位置で作ったdf
    """

    r = 0  # 粒子の初期位置
    r_list = [r]  # 粒子の位置の時間変化を格納するリスト
    for i in range(n):
        r += random_walk(p)
        r_list.append(r)
    # pandasでデータフレームにする
    df = pd.DataFrame(r_list, columns=[column])
    return df

    # 以下プロット処理


def plot_axis(file_name, df):
    """3つの横軸でプロットする"""
    _, ax1 = plt.subplots()
    ax1.plot(df.index, df.values, "b-")
    ax1.set_ylabel("position")
    ax1.set_xlabel("time")
    plt.savefig(f'{file_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


# plotする
def plot_twin_x(df_list, filename):
    fig = plt.figure(figsize=(12, 4))
    for i in range(len(df_list.columns)):
        # 1回目だけsubplotの作成に当てる
        if i == 0:
            ax = fig.add_subplot()
            ax_list = [ax]
            continue
        ax = ax_list[0].twinx()
        ax_list.append(ax)
    i = 0
    # 折れ線グラフをそれぞれ異なる色に設定するためのカラーマップを設定。tab20は20色までしかないことに注意
    cmap = plt.get_cmap('tab20')
    for column_name, df in df_list.iteritems():
        ax_list[i].plot(df.index, df.values, label=column_name, c=cmap(i))
        ax_list[i].set_ylabel(column_name)
        ax_list[i].spines["right"].set_position(("axes", 1.2 + i * 0.2))
        i += 1
    fig.legend()
    # fig.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.show()


class Position_1d:
    def __init__(self, x=None, p=None, t=None):
        # 1次元上の位置
        self.x: int = x
        # 位置xにいる確率
        self.p: float = p
        # 時刻t
        self.t: int = t
        # 1時刻前のposition
        # self.before = None


def random_walker_achieve_probability(p, n):
    """
    時刻t+1において位置xにいる確率（1次元）
    ν_{t+1}(x) = pν_t(x+1) + qν_t(x-1)

    n: T
    p: +1進む確率
    :return:
    """
    q = 1 - p
    nu_list = [[1]]
    for _ in range(n):
        nu_t_plus_1 = [0] * (len(nu_list[-1]) + 2)
        for index, value in enumerate(nu_list[-1]):
            nu_t_plus_1[index] += value * q
            nu_t_plus_1[index + 2] += value * p
        nu_list.append(nu_t_plus_1)

    return nu_list


if __name__ == '__main__':
    # 実験1：p=0.1~0.9まで0.1とびにデータフレームを作成し、一つのデータフレームに結合する
    if False:
        df_list = pd.DataFrame()
        for p in range(1, 10, 1):
            p /= 10
            df = test_1(1000, p, f"position_p={p}")
            # dfをconnectしていく
            df_list = pd.concat([df_list, df], axis=1)
        filename = "一次元上の軸において、p=0.1〜0.9において、時間の軸は固定で、位置の軸をそれぞれで用意する"
        plot_twin_x(df_list, filename)

    # 実験2：次に、正の方向へ進む確率を0.1~0.9の9通りに変化させて図で表す
    if False:
        df_list = pd.DataFrame()
        for p in range(1, 10, 1):
            p /= 10
            df = test_1(1000, p, f"position_p={p}")
            df_list = pd.concat([df_list, df], axis=1)

        df_list.plot()
        file_name = "次に、正の方向へ進む確率を0.1~0.9の9通りに変化させて図で表す"
        plt.savefig(f'{file_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 実験3：2次元でランダムウォークする
    if False:
        T = 100
        test_2(T, f"2次元ランダムウォーク_{T}回")
        T = 1000
        test_2(T, f"2次元ランダムウォーク_{T}回")
        T = 10000
        test_2(T, f"2次元ランダムウォーク_{T}回")

    # 実験4：元の道に戻れない場合
    if False:
        T = 100
        test_2(T, f"2次元ランダムウォーク_{T}回")
        # n = 1000
        # test_2(n, f"2次元ランダムウォーク_{n}回")
        # n = 10000
        # test_2(n, f"2次元ランダムウォーク_{n}回")

    # 実験5：式2.1を利用したランダムウォークの確率計算とプロット
    if False:
        nu_list = random_walker_achieve_probability(p=0.5, n=100)

        # シフトさせる
        for i, v in enumerate(nu_list):
            shift = int((len(nu_list[-1]) - len(nu_list[i])) / 2)
            nu_list[i] = [0] * shift + nu_list[i] + [0] * shift

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel="x", ylabel="p")
        length = (len(nu_list[100]) - 1) / 2
        x = np.arange(-length, length + 1, 1, int)
        for i in [20, 40, 60, 80, 100]:
            ax.plot(x, nu_list[i], label=f"t={i}")
        plt.legend()
        # plt.savefig("式2.1を利用したランダムウォークの確率計算とプロット.png", dpi=300, bbox_inches='tight')
        plt.savefig("式2.1を利用したランダムウォークの確率計算とプロット-シフト.png", dpi=300, bbox_inches='tight')
        plt.show()

    """実験5-2
    """
    if False:
        T = 20
        nu_list = random_walker_achieve_probability(p=0.5, n=T)

        # シフトさせる
        for i, v in enumerate(nu_list):
            shift = int((len(nu_list[-1]) - len(nu_list[i])) / 2)
            nu_list[i] = [0] * shift + nu_list[i] + [0] * shift

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel="x", ylabel="p")
        length = (len(nu_list[-1]) - 1) / 2
        x = np.arange(-length, length + 1, 1, int)
        for i in range(T):
            ax.bar(x, nu_list[i], label=f"t={i}")
        plt.legend()
        plt.savefig(f"img/ex-5/式2.1を利用したランダムウォークの確率計算とプロット-シフト-t=0~{T - 1}-棒グラフ.png", dpi=300, bbox_inches='tight')
        plt.show()

    """実験5-3
        n=num**2専用のコード
        グラフ複数表示
        """
    if False:
        num = 3
        T = num ** 2
        p = 0.6
        nu_list = random_walker_achieve_probability(p=p, n=T)

        # シフトさせる
        for i, v in enumerate(nu_list):
            shift = int((len(nu_list[-1]) - len(nu_list[i])) / 2)
            nu_list[i] = [0] * shift + nu_list[i] + [0] * shift

        fig = plt.figure(tight_layout=True)
        length = (len(nu_list[-1]) - 1) / 2
        x = np.arange(-length, length + 1, 1, int)

        for i in range(T):
            ax = fig.add_subplot(num, num, i + 1, xlabel="x", ylabel="p")
            ax.bar(x, nu_list[i], label=f"t={i}")
            plt.legend()
        plt.savefig(f"img/ex-5/式2.1を利用したランダムウォークの確率計算とプロット-シフト-t=0~{T - 1}-棒グラフ-複数のグラフ表示p={p}.png", dpi=400,
                    bbox_inches='tight')
        plt.show()

    """実験5-3
        n=特定のコード
        グラフ複数表示
        折れ線グラフ
    """

    if False:
        plot_graph_num_by_axis = 2
        T = 1000
        p = 0.8
        plot_start_t = 10
        plot_end_t = 400
        graph_type = "棒グラフ"
        plot_step_t = int((-plot_start_t + plot_end_t) / plot_graph_num_by_axis ** 2) + 1

        nu_list = random_walker_achieve_probability(p=p, n=T)

        fig = plt.figure(tight_layout=True)

        for i, v in enumerate(range(plot_start_t, plot_end_t, plot_step_t)):
            ax = fig.add_subplot(plot_graph_num_by_axis, plot_graph_num_by_axis, i + 1, xlabel="x", ylabel="p")
            length = (len(nu_list[v]) - 1) / 2
            x = np.arange(-length, length + 1, 1, int)
            ax.bar(x, nu_list[v], label=f"t={v}")
            plt.legend()
        plt.savefig(
            f"img/ex-5/式2.1を利用したランダムウォークの確率計算-複数のグラフ表示-表示するグラフ数{plot_graph_num_by_axis ** 2}-{graph_type}-"
            f"t={plot_start_t}~{plot_end_t},step={plot_step_t}-p={p}.png",
            dpi=400,
            bbox_inches='tight')
        plt.show()
