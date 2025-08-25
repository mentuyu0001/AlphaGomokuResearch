# 変更の可能性があるパラメータ
BOARD_SIZE = 9          # 盤面
PV_EVALUATE_COUNT = 250 # 1推論あたりのシミュレーション回数（本家囲碁は1600回）
SP_GAME_COUNT = 200    # セルフプレイゲーム数
# 学習パラメータ
PATIENCE_EPOCHS = 10 # 何エポック学習が向上しなかったらあきらめるか
RN_EPOCHS = 500 # 最大エポック数
LOAD_FILES = 300 # ロードするファイル数
BATCH_SIZE = 512 # バッチサイズ
C_PUCT = 4.0 # モンテカルロ木探索の定数
AUGMENTATION_PROBABILITY = 0.1  # 盤面複製確率
# 変更の可能性があるパラメータ（以上）

# ボードサイズ
WIN_COUNT = 5            # 5つ並べで勝ち
BOARD_LEN = BOARD_SIZE * BOARD_SIZE

# セルフプレイ部パラメータ
SP_TEMPERATURE = 1.0      # 温度パラメータ（学習時1、実行時0）

# 畳み込みパラメータ
DN_FILTERS = 256 # 畳み込み層のカーネル数
DN_RESIDUAL_NUM = 5 # 残差ブロックの数（本家囲碁は19）
DN_INPUT_SHAPE = (2, 9, 9)  # PyTorch: (C, H, W)
DN_OUTPUT_SIZE = 81 # 行動数(配置先(15*15))