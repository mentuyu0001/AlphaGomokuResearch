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

        print(f"Board size: {LearningParameters.BOARD_SIZE}")
        print(f"PV Evaluate count: {LearningParameters.PV_EVALUATE_COUNT}")
        print(f"SP Game count: {LearningParameters.SP_GAME_COUNT}")
        print(f"Load files: {LearningParameters.LOAD_FILES}")

        cnt = 0
        while True:
            cnt += 1
            print(f'Train {cnt} ====================')

            # ネットワーク学習
            train_network(cnt)
