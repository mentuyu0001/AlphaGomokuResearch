# ====================
# 学習サイクルの実行
# ====================

from DualNetwork import AlphaGomokuNet, save_model
from SelfPlay import self_play
from TrainNetwork import train_network
import os
import LearningParameters

def dual_network():
    model_path = './model/AlphaGomoku.pth'
    if not os.path.exists(model_path):
        model = AlphaGomokuNet()
        save_model(model, path=model_path)
        del model

if __name__ == '__main__':
        dual_network()

        cnt = 0

        folder_path = 'C:\\Users\\sudok\\Desktop\\master_research_Miyazaki\\gomoku\\AlphaGomoku9X9\\Losses'
        # 指定されたパスがディレクトリ（フォルダー）として存在するか確認
        if os.path.isdir(folder_path):
            # フォルダー内の全てのアイテム（ファイルとサブフォルダー）のリストを取得
            for item in os.listdir(folder_path):
                # アイテムのフルパスを作成
                item_path = os.path.join(folder_path, item)
                # それがファイルであり、サブフォルダーではないことを確認
                if os.path.isfile(item_path):
                    cnt += 1
        
        while True:
            cnt += 1

            if (cnt <= 5):
                 LearningParameters.PV_EVALUATE_COUNT = 10
            elif (cnt <= 10):
                 LearningParameters.PV_EVALUATE_COUNT = 50
            elif (cnt <= 15):
                 LearningParameters.PV_EVALUATE_COUNT = 100
            elif (cnt <= 20):
                 LearningParameters.PV_EVALUATE_COUNT = 200
            else:
                 LearningParameters.PV_EVALUATE_COUNT = 250

            print(f'Train {cnt} ====================')
            print(f"Board size: {LearningParameters.BOARD_SIZE}")
            print(f"PV Evaluate count: {LearningParameters.PV_EVALUATE_COUNT}")
            print(f"SP Game count: {LearningParameters.SP_GAME_COUNT}")
            print(f"Load files: {LearningParameters.LOAD_FILES}")

            # セルフプレイ
            self_play()

            # ネットワーク学習
            train_network(cnt)
