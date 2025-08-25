# ====================
# 学習サイクルの実行
# ====================

from DualNetwork import AlphaGomokuNet, save_model
from SelfPlay import self_play
from TrainNetwork import train_network
import os

def dual_network():
    model_path = './model/AlphaGomoku.pth'
    if not os.path.exists(model_path):
        model = AlphaGomokuNet()
        save_model(model, path=model_path)
        del model

if __name__ == '__main__':
        dual_network()

        print("atodeyaruyatu")

        cnt = 0
        while True:
            cnt += 1
            print(f'Train {cnt} ====================')
            # セルフプレイ
            self_play()

            # ネットワーク学習
            train_network(cnt)
