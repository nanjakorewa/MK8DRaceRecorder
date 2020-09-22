from utils import *

import logging
import os
import time
import warnings

import subprocess
from subprocess import PIPE

formatter = '%(levelname)s : %(asctime)s : %(message)s'
logging.basicConfig(filename='mk8d.log', level=logging.INFO, format=formatter)
warnings.simplefilter('ignore')


"""###################################
パラメータ
###################################"""
SC_COMMAND = "screenshot OBS -t OBS -f "  # スクリーンショット用コマンド
TEMP_IMG_FILENAME = "temp.png"  # キャプチャ結果の保存先
WAIT_SECOND = 0.2  # 処理間の待機時間（秒）
WAITTIME_BEFORE_DELETE = 6  # 画像を消すまでに猶予を持たせる
IS_RACING_CHECK_LENGTH = 4  # xWAIT_SECONDの間レース判定が出ない場合は処理をリセットする
IS_RACING_CHECK_LENGTH_RACEEND = 3  # レース終わり確認
DRAW_LAPLINE = False # ラップの区切りを見せる


def run_server():
    frame_num = 0
    is_racing_flag_list = [False for _ in range(IS_RACING_CHECK_LENGTH + 1)]
    is_racing_now = False

    curent_lap = 1
    lap_history = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    lap_index = []
    coin_history = [0, ]
    rank_history = [12, ]
    im_before_coin = 0  # 直前の時刻のコイン

    while(True):
        logging.info("[log] is_racing_now==%s" % is_racing_now)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        # OBSのSCを取得
        im_before_coin = coin_history[-1]
        frame_num += 1
        res = subprocess.run("screenshot OBS -t OBS -f temp_raw.png", shell=True, stdout=PIPE, stderr=PIPE, text=True)
        time.sleep(WAIT_SECOND)
        if not res:
            continue

        frame_gray = cv2.imread("temp_raw.png", 0)
        frame_gray = frame_gray[70:1230:, 90:2180]
        frame_gray = cv2.resize(frame_gray, (400, 300))
        cv2.imwrite(TEMP_IMG_FILENAME, frame_gray)

        # 画像の認識結果をcsvに書き出し
        frame_gray = cv2.imread(TEMP_IMG_FILENAME, 0)
        ret1, lp = get_lap(frame_gray)
        ret2, cn = get_coinnum(frame_gray, im_before_coin)
        ret3, rk = get_rank(frame_gray)
        is_racing_flag = ret1 and ret2 and ret3
        is_racing_flag_list.append(is_racing_flag)

        # 現在の状態を更新
        lap = lp.replace('.png', '')
        coin = cn.replace('.png', '')
        rank = rk.replace('.png', '')

        # レース判定が降りない場合は手前の時刻の結果を再利用する
        if not is_racing_flag:
            lap = lap_history[-1]
            coin = coin_history[-1]
            rank = rank_history[-1]

        logging.info("lap:%s coin:%s rank:%s is_racing_flag==%s" % (lap, coin, rank, is_racing_flag))

        if lap in LAP_LIST:
            lap_number = LAP_LIST.index(lap) + 1
            lap_history.append(lap_number)

            lap_stat_mode = 1
            lap_2_count = lap_history[-6:].count(2)
            lap_3_count = lap_history[-6:].count(3)
            logging.info("[lap_history] %s " % lap_history[-6:])
            if curent_lap == 1 and lap_2_count > 3 and len(lap_history) > 20:
                lap_stat_mode = 2
            elif curent_lap == 2 and lap_3_count > 4 and len(lap_history) > 40:
                lap_stat_mode = 3

            # ラップが更新された場合はそのインデックスを記録する
            if lap_stat_mode > curent_lap:
                curent_lap = lap_stat_mode
                lap_index.append((len(lap_history) - 10) / PLOT_WINDOW - 2.0)
        else:
            curent_lap = 1

        if coin in COIN_LIST:
            coin_history.append(COIN_LIST.index(coin))
        elif is_racing_now:
            coin_history.append(coin_history[-1])
        else:
            coin_history.append(-2)

        if rank in RANK_LIST:
            rank_history.append(RANK_LIST.index(rank) + 1)
        elif is_racing_now:
            rank_history.append(rank_history[-1] + 1)
        else:
            rank_history.append(-2)

        # 3週以上は無視
        if len(lap_index) > 3:
            curent_lap = 3
            lap_index = lap_index[:3]

        if not is_racing_now and all(np.array(is_racing_flag_list[-IS_RACING_CHECK_LENGTH:]) == 1):
            # レースしていない状態から続けてレース判定が降りた場合はレース処理に移行する
            is_racing_now = True
            curent_lap = 1
            lap_history = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            lap_index = []
            coin_history = [0, ] + coin_history[-2:]
            rank_history = [12, ] + rank_history[-2:]
            output_race_status(curent_lap, coin_history, rank_history, is_racing_flag, is_racing_now, lap_index, draw_lapline=DRAW_LAPLINE)
            logging.info("レースを開始")
            continue
        elif is_racing_now and all(np.array(is_racing_flag_list[-IS_RACING_CHECK_LENGTH_RACEEND:]) == 0):
            # 指定時間レース判定が起きない場合は一時ファイルをリセットし、レース終了処理に移行する
            is_racing_now = False
            curent_lap = 1
            lap_history = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            lap_index = []
            coin_history = [0]
            rank_history = [12]

            time.sleep(WAITTIME_BEFORE_DELETE)  # 画像を消すまでに猶予を持たせる
            output_race_status(curent_lap, coin_history, rank_history, is_racing_flag, is_racing_now, lap_index, draw_lapline=DRAW_LAPLINE)

            delete_temp_file()
            logging.info("レースを終了")
            continue
        elif not is_racing_flag:
            # レースフラグ判定が降りない場合は一旦プロットしない
            continue
        elif is_racing_now:
            # レース中はグラフを出す
            output_race_status(curent_lap, coin_history, rank_history, is_racing_flag, is_racing_now, lap_index, draw_lapline=DRAW_LAPLINE)
    logging.info("Finish!!!!")


if __name__ == '__main__':
    delete_temp_file()
    run_server()
