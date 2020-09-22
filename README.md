# MK8D Record
配信時にコイン枚数・ラップ・順位を記録します。

# 使用例

[![](http://img.youtube.com/vi/QT2UY8TPNiE/0.jpg)](http://www.youtube.com/watch?v=QT2UY8TPNiE "")

# セットアップ
mac OSX Catalina上にて動作確認しています。

1. `conda env create -f environment.yml` にて必要なライブラリをインストールした環境を作成
2. `source activate mk8d-record-env`
3. `conda install -c conda-forge opencv` # opencvのインストール
4. `python server.py`
5. OBSを起動して、順位判定が正しく行われるようにOBSのウィンドウサイズを調整する

# 補足
スクリーンショット取得には以下のライブラリを使用しています。
The following libraries are used to get screenshots.
[alexdelorenzo/screenshot](https://github.com/alexdelorenzo/screenshot)

また、`img/` フォルダ以下の各静止画の著作権は任天堂に帰属します。
静止画は[ネットワークサービスにおける任天堂の著作物の利用に関するガイドライン](https://www.nintendo.co.jp/networkservice_guideline/ja/index.html)における "個人であるお客様は、任天堂のゲーム著作物を利用した動画や静止画等を、営利を目的としない場合に限り、投稿することができます。"と指定された範囲内でのみ利用しています。