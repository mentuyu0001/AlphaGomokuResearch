# ====================
# セルフプレイ部
# ====================

from GomokuGame import State, create_special_initial_state
from PVmcts import pv_mcts_scores
from LearningParameters import DN_OUTPUT_SIZE
from datetime import datetime
import numpy as np
import pickle
import os
import torch
from pathlib import Path
import LearningParameters
from DualNetwork import AlphaGomokuNet
import uuid

# パラメータ
sp_game_count = LearningParameters.SP_GAME_COUNT
sp_tempreature = LearningParameters.SP_TEMPERATURE

# 先手プレイヤーの価値計算（勝ち=1、引き分け=0.5、負け=0）
def first_player_value(ended_state):
    if ended_state.is_lose():
        # 先手負けなら0、先手勝ちなら1
        return 1 if ended_state.is_first_player() else 0 # 修正のために0, 1を逆転させたよ！
    elif ended_state.is_draw():
        return 0.5
    else:
        # ゲーム終了していない場合は想定外
        return 0.5

# 学習データ保存関数
def write_data(history):
    now = datetime.now()
    
    # このスクリプトファイルがあるディレクトリの絶対パスを取得
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # 保存先となる 'data' フォルダの絶対パスを作成
    save_dir = os.path.join(script_directory, 'data')
    
    
    # ディレクトリが存在しない場合は再帰的に作成
    os.makedirs(save_dir, exist_ok=True)
        
    # 保存するファイル名を生成
    now = datetime.now()
    file_name = f'{now:%Y%m%d%H%M%S}_{uuid.uuid4()}.history'
        
    # ファイルの完全な絶対パスを生成
    full_path = os.path.join(save_dir, file_name)
    print(full_path)
    with open(full_path, 'wb') as f:
        pickle.dump(history, f)

# 1ゲームのセルフプレイ実行
def play(model, device):
    history = []
    state = create_special_initial_state()

    while True:
        if state.is_done():
            break
        
        scores = pv_mcts_scores(model, state, sp_tempreature)

        # policies配列はモデルの出力次元と同じ (9*9=81)
        policies = np.zeros(DN_OUTPUT_SIZE, dtype=np.float32)
        legal_actions = state.legal_actions()

        # MCTSのスコアを、81マスのポリシー配列の合法手の位置に格納
        if len(scores) > 0:
            policies[legal_actions] = scores

        history.append([state.to_tensor(), policies, None])

        # scoresの確率分布に従って次の一手を選択
        action = np.random.choice(legal_actions, p=scores)
        state = state.next(action)

    value = first_player_value(state)
    # 価値を交互に割り当て：先手手番には value、後手手番には 1 - value
    for i in range(len(history)):
        history[i][2] = value if i % 2 == 0 else 1 - value

    return history

# セルフプレイ複数回実行
def self_play():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = sorted(Path('C:/Users/sudok/Desktop/master_research_Miyazaki/gomoku/AlphaGomoku9X9/model').glob('*.pth'))[-1]
    model = AlphaGomokuNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_history = []
    for i in range(sp_game_count):
        h = play(model, device)
        all_history.extend(h)
        print(f'\rSelfPlay {i+1}/{sp_game_count}', end='')
    print()

    write_data(all_history)

if __name__ == '__main__':
    self_play()
