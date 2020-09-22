
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from skimage.feature import hog
from skimage import measure


"""###################################
処理時間調整パラメータ
###################################"""
OUTPUT_TEMP_IMAGE = True  # デバッグ用のファイルを出力する
SCALE_PARAM = 2  # 処理画像のスケーリング（自然数のみ）
PLOT_WINDOW = 2  # プロット間隔

"""###################################
画像処理実装箇所
* scikit-image以下のHOG特徴とエッジ検出を用いて認識を行う
* ラップ判定が誤認識率が少し高い
###################################"""

# HOG特徴のハイパーパラメータ
PIXEL_PER_CELL = (5, 5)
CELL_PER_BLOCK = (2, 2)
PIXEL_PER_CELL_COIN = (5, 5)
CELL_PER_BLOCK_COIN = (2, 2)
PIXEL_PER_CELL_LAP = (4, 4)
CELL_PER_BLOCK_LAP = (2, 2)
USE_HOG_FOR_COIN = False  # コイン枚数検出にHOGを使用する場合のみTrue

# 画像ファイル名
LAP_LIST = ['c_3_1', 'c_3_2', 'c_3_3']
COIN_LIST = ['c_00', 'c_01', 'c_02', 'c_03',
             'c_04', 'c_05', 'c_06', 'c_07',
             'c_08', 'c_09', 'c_10']
RANK_LIST = ['c_1', 'c_2', 'c_3', 'c_4',
             'c_5', 'c_6', 'c_7', 'c_8',
             'c_9', 'c_10', 'c_11', 'c_12']

# 画像からのデータ取得処理
LAP_NAME = [ln for ln in os.listdir('img/dataset/lap/') if ln.startswith('c_')]
LAP_BI_IMGS = [cv2.imread(os.path.join('img/dataset/lap/', ln), 0) for ln in LAP_NAME]
LAP_2V_IMGS = []
COIN_NAME = [ln + ".png" for ln in COIN_LIST]
COIN_BI_IMGS = [cv2.imread(os.path.join('img/dataset/coin/', ln), 0) for ln in COIN_NAME]
COIN_2V_IMGS = []
RANK_NAME = [ln for ln in os.listdir('img/dataset/rank/') if ln.startswith('c_')]
RANK_BI_IMGS = [cv2.imread(os.path.join('img/dataset/rank/', ln), 0) for ln in RANK_NAME]
RANK_2V_IMGS = []

for bim in LAP_BI_IMGS:
    hog_vector = hog(bim[:, 15:-12], block_norm="L2", pixels_per_cell=PIXEL_PER_CELL_LAP,
                     cells_per_block=CELL_PER_BLOCK_LAP, orientations=9)
    LAP_2V_IMGS.append(hog_vector)

for bim in COIN_BI_IMGS:
    hog_vector = hog(bim[:, 4:-1], block_norm="L2", pixels_per_cell=PIXEL_PER_CELL_COIN,
                     cells_per_block=CELL_PER_BLOCK_COIN, orientations=9)
    COIN_2V_IMGS.append(hog_vector)

for bim in RANK_BI_IMGS:
    hog_vector = hog(bim, block_norm="L1-sqrt", pixels_per_cell=PIXEL_PER_CELL, cells_per_block=CELL_PER_BLOCK,)
    RANK_2V_IMGS.append(hog_vector)



"""###################################
順位・コイン認識
* あらかじめ用意したテンプレートからもっとも類似度が高い画像を見つける
###################################"""
def get_is_racing(image):
    """レース中判定のフラグ

    Args:
        image: グレースケール画像

    Returns:
        bool: レース中フラグ
    """
    return True


def get_lap(image):
    """画像から現在のラップ数を取得する

    Args:
        image: グレースケール画像

    Returns:
        判定可否：falseならば領域の取得に失敗している可能性が高い
        ラップ名
    """
    lap_area = image[273:290, 64:77]  # 配信時に調整すること
    lap_area_hog = hog(lap_area, block_norm="L2", pixels_per_cell=PIXEL_PER_CELL_LAP,
                       cells_per_block=CELL_PER_BLOCK_LAP, orientations=9)
    scores = [np.mean(np.abs(lap_area_hog - lp)) for lp in LAP_2V_IMGS]

    res = np.argmin(scores)
    min_score = np.min(scores)

    if OUTPUT_TEMP_IMAGE:
        cv2.imwrite("temp_lap_.png", lap_area)
        logging.info("[lap_ min score] %s " % min_score)
    return min_score < 0.12, LAP_NAME[res]


def get_coinnum(image, im_before_coin):
    """画像から現在のコイン数を取得する

    Args:
        image: グレースケール画像
        im_before_coin: 直前のコイン数

    Returns:
        判定可否：falseならば領域の取得に失敗している可能性が高い
        コイン枚数名
    """
    coin_area = image[273:290, 29:47]  # 配信時に調整すること
    scores = []
    coin_area_hog = hog(coin_area, block_norm="L2", pixels_per_cell=PIXEL_PER_CELL_COIN,
                        cells_per_block=CELL_PER_BLOCK_COIN, orientations=9)

    for cn, bi in zip(COIN_2V_IMGS, COIN_BI_IMGS):
        hog_score = np.mean(np.abs(coin_area_hog - cn))
        ssim_score = measure.compare_ssim(bi[:, 4:-1], coin_area)
        scores.append(hog_score * (1 - ssim_score))

    # コインは急に１０増加したりはしないので、直前のコイン枚数に近い箇所だけ選ばれやすくする
    if im_before_coin <= 3:
        for i in [4, 5, 6, 7, 8, 9]:
            cname_index = COIN_NAME.index("c_0%s.png" % (i))
            scores[cname_index] = scores[cname_index] * 1.02
    elif 3 < im_before_coin < 8:
        for i in [-2, -1, 0, 1, 2]:
            cname_index = COIN_NAME.index("c_0%s.png" % (im_before_coin - i))
            scores[cname_index] = scores[cname_index] * 0.98

    res = np.argmin(scores)
    min_score = np.min(scores)

    if OUTPUT_TEMP_IMAGE:
        cv2.imwrite("temp_coin.png", coin_area)
        logging.info("[coin min score] %s " % min_score)
    return min_score < 0.105, COIN_NAME[res]


def get_rank(image):
    """画像から現在の順位を取得する

    Args:
        image: グレースケール画像

    Returns:
        判定可否：falseならば領域の取得に失敗している可能性が高い
        順位名
    """
    rank_area = image[243:290, 340:380]
    rank_area_hog = hog(rank_area, block_norm="L1-sqrt", pixels_per_cell=PIXEL_PER_CELL,
                        cells_per_block=CELL_PER_BLOCK,)
    scores = [np.mean(np.abs(rank_area_hog - cn)) for cn in RANK_2V_IMGS]
    res = np.argmin(scores)
    min_score = np.min(scores)

    if OUTPUT_TEMP_IMAGE:
        cv2.imwrite("temp_rank.png", rank_area)
        logging.info("[rank min score] %s " % min_score)
    return min_score < 0.105, RANK_NAME[res]


"""###################################
出力用ファイル作成
* OBSに出力するための画像ファイルを作成する
###################################"""
def plot_lap(fig, lap_index, draw_lapline=False):
    """ラップの区切りを描画する

    Args:
        fig : グラフ
        lap_index (list of int): ラップ変化点のインデックス
        draw_lapline (bool, optional): Trueの時のみラップ変化を描画. Defaults to SHOW_LAP.
    """
    if draw_lapline:
        # ラップの区切りをプロットする
        for lapnum, lap_time in enumerate(lap_index):
            if lapnum + 2 > 3:
                break
            fig.axvline(x=lap_time, ymin=-1, ymax=20, color="gold", linewidth=13.5, alpha=0.5)
            fig.text(lap_time + 1.0, 0, "lap%s" % (lapnum + 2), size=16, color="gold")


def output_race_status(cl, ch, rh, is_racing_flag, is_racing_now, lap_index, draw_lapline=False):
    """OBSに出力するための画像を作成する

    Args:
        cl (list of int): 記録されているラップのリスト
        ch (list of int): 記録されているコインのリスト
        rh (list of int): 記録されている順位のリスト
        is_racing_flag (bool): レース判定フラグ、レースと判定されている時True
        is_racing_now (bool): レース中フラグ、レース中のみTrue
        lap_index (list of int): ラップ切り替え時点のインデックス
    """
    logging.info("[is_racing_flag] %s" % is_racing_flag)
    logging.info("   > ラップ： %s " %  cl)
    logging.info("   > コイン： %s " %  ch[-1:])
    logging.info("   > 順　位： %s " %  rh[-1:])

    if not is_racing_now:
        return

    # コインのプロット用データを作成する
    ch_for_plot = []
    if len(ch) < PLOT_WINDOW + 2:
        ch_for_plot = ch
    else:
        for i in range(int(len(ch) / PLOT_WINDOW)):
            ch_for_plot.append(np.mean(ch[i * PLOT_WINDOW:(i + 1) * PLOT_WINDOW]))

    data_length = len(ch_for_plot)

    # コイングラフの作成
    if is_racing_now:
        coin = pd.DataFrame([ch_for_plot[1:]]).T
    else:
        coin = pd.DataFrame([[-2]]).T

    coin.columns = ["coin"]
    plt.figure(figsize=(10, 2), dpi=100)
    plt.tick_params(labelbottom=False)
    plt.hlines(y=0, xmin=-1, xmax=data_length, colors='gray', linewidths=1)
    plt.hlines(y=5, xmin=-1, xmax=data_length, colors='gray', linewidths=1)
    plt.hlines(y=10, xmin=-1, xmax=data_length, colors='gray', linewidths=1)
    plt.yticks([0, 5, 10], [0, 5, 10], fontsize=18)
    plt.ylim(-2, 12)
    plot_lap(plt, lap_index, draw_lapline=draw_lapline)

    try:
        g = sns.lineplot(data=coin, palette="tab10", linewidth=0, marker="o", markersize=10)
        g.legend_.remove()
        plt.savefig("img/obs_coin.png", transparent=True)
        plt.close()
    except AttributeError:
        plt.close()
        logging.info("描画失敗")
        return

    # ランキング推移の作成
    rh_for_plot = []
    if len(rh) < PLOT_WINDOW + 2:
        rh_for_plot = rh
    else:
        for i in range(int(len(rh) / PLOT_WINDOW)):
            rh_for_plot.append(np.mean(rh[i * PLOT_WINDOW:(i + 1) * PLOT_WINDOW]))

    rank = pd.DataFrame([[13 - r for r in rh_for_plot[1:]]]).T
    rank.columns = ["rank"]

    plt.figure(figsize=(10, 2), dpi=100)
    plt.tick_params(labelbottom=False)
    plt.hlines(y=1, xmin=-1, xmax=data_length, colors='gray', linewidths=1)
    plt.hlines(y=4, xmin=-1, xmax=data_length, colors='gray', linewidths=1)
    plt.hlines(y=10, xmin=-1, xmax=data_length, colors='brown', linewidths=1)
    plt.hlines(y=12, xmin=-1, xmax=data_length, colors='gold', linewidths=1)
    plt.ylim(-2, 14)
    plt.yticks([4, 12], [9, 1], fontsize=18)
    plot_lap(plt, lap_index, draw_lapline=draw_lapline)

    try:
        g = sns.lineplot(data=rank, palette="tab10", linewidth=0, marker="o", markersize=10)
        g.legend_.remove()
        plt.savefig("img/obs_rank.png", transparent=True)
        plt.close()
    except AttributeError:
        plt.close()
        logging.info("描画失敗")
        return


"""###################################
一時ファイル作成・削除
###################################"""
def delete_temp_file():
    """一時ファイルを削除する
    """
    if os.path.exists("img/obs_coin.png"):
        os.remove("img/obs_coin.png")
    if os.path.exists("img/obs_rank.png"):
        os.remove("img/obs_rank.png")

    if os.path.exists("temp_lap_.png"):
        os.remove("temp_lap_.png")
    if os.path.exists("temp_coin.png"):
        os.remove("temp_coin.png")
    if os.path.exists("temp_rank.png"):
        os.remove("temp_rank.png")
