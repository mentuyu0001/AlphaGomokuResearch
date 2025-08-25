import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
import pickle
from DualNetwork import AlphaGomokuNet
import matplotlib.pyplot as plt
import datetime
import os
import LearningParameters
import torch.nn.functional as F

# 学習パラメータ
patience_epochs = LearningParameters.PATIENCE_EPOCHS
default_batch_size = LearningParameters.BATCH_SIZE
load_files = LearningParameters.LOAD_FILES

input_shape = LearningParameters.DN_INPUT_SHAPE

max_epochs = LearningParameters.RN_EPOCHS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_multiple_histories(n_latest=load_files):

    # 1. Pathオブジェクトとしてスクリプトのあるディレクトリを取得
    script_dir = Path(__file__).resolve().parent
    
    # 2. '/' を使って 'data' ディレクトリへのパスを作成 (結果もPathオブジェクト)
    data_dir = script_dir / 'data'

    history_files = sorted(data_dir.glob('*.history'), reverse=True)
    latest_files = history_files[:n_latest]

    all_data = []
    for file_path in latest_files:
        with file_path.open('rb') as f:
            data = pickle.load(f)
            all_data.extend(data)

    # 古いファイル削除（最新5個以外）
    files_to_delete = history_files[n_latest:]
    for old_file in files_to_delete:
        try:
            old_file.unlink()
            print(f"Deleted old history file: {old_file.name}")
        except Exception as e:
            print(f"Failed to delete {old_file.name}: {e}")

    return all_data

# 学習率スケジューラ
def get_lr(epoch):
    """
    if epoch < 200:
        return 0.001
    elif epoch < 400:
        return 0.0005
    else:
        return 0.0002
    """
    return 0.0002

def train_network(train_cycle):

    # --- ログ保存用の設定を変更 ---
    LOG_DIR = './log'
    # ★★★ ファイル名を .txt に変更 ★★★
    LOG_FILE = os.path.join(LOG_DIR, 'training_log.txt') 
    os.makedirs(LOG_DIR, exist_ok=True)

    # ファイルがなければヘッダーを書き込む
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f:
            # ヘッダーもカンマ区切りで書き込みます
            f.write('timestamp,train_cycle,epoch,learning_rate,total_loss,policy_loss,value_loss\n')
    # --- ログ設定ここまで ---
    # --- ★★★ グローバルエポック数を決定するロジックを追加 ★★★ ---
    global_epoch_start_num = 1
    try:
        with open(LOG_FILE, 'r') as f:
            # 最終行を読み込む
            lines = f.readlines()
            if len(lines) > 1: # ヘッダー行以外にデータがある場合
                last_line = lines[-1]
                # 最終行からグローバルエポック数を取得
                last_epoch_num = int(last_line.split(',')[0])
                global_epoch_start_num = last_epoch_num + 1
    except (FileNotFoundError, IndexError):
        # ファイルが存在しない、または空の場合は、新しいヘッダーを書き込む
        with open(LOG_FILE, 'w') as f:
            f.write('global_epoch,timestamp,learning_rate,total_loss,policy_loss,value_loss\n')
    # --- グローバルエポック数決定ロジックここまで ---

    # データ読み込み
    history = load_multiple_histories()
    xs, y_policies, y_values = zip(*history)

    # ndarray化・形状変換 (N, C, H, W)
    c, a, b = input_shape  # 11, 15, 15
    xs = np.array(xs, dtype=np.float32).reshape(len(xs), c, a, b)
    y_policies = np.array(y_policies, dtype=np.float32)
    y_values = np.array(y_values, dtype=np.float32)

    # PyTorch Tensorに変換
    xs_tensor = torch.tensor(xs)
    y_policies_tensor = torch.tensor(y_policies)
    y_values_tensor = torch.tensor(y_values).unsqueeze(1)  # (N,1)に変換

    dataset = TensorDataset(xs_tensor, y_policies_tensor, y_values_tensor)
    dataloader = DataLoader(dataset, batch_size=default_batch_size, shuffle=True)

    # モデル初期化
    model = AlphaGomokuNet().to(DEVICE)

    # 最良モデル(best.pth)があればロード（なければ初期モデルを使用）
    best_model_path = Path('./model/AlphaGomoku.pth')
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))

    # 損失関数と最適化
    #criterion_policy = nn.NLLLoss()  # y_policiesは確率分布なので要調整
    criterion_value = nn.functional.binary_cross_entropy_with_logits
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    learning_rates = []
    total_losses = []
    policy_losses = []
    value_losses = []

    # 損失の初期値の設定
    best_loss = float('inf')
    patience_counter = 0
    epoch = 0

    while epoch < max_epochs:
        # 学習率の更新
        lr = get_lr(train_cycle)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        total_loss = 0
        total_loss_policy = 0
        total_loss_value = 0

        for x_batch, y_policy_batch, y_value_batch in dataloader:
            x_batch = x_batch.to(DEVICE)
            y_policy_batch = y_policy_batch.to(DEVICE)
            y_value_batch = y_value_batch.to(DEVICE)
            #y_policy_labels_batch = torch.argmax(y_policy_batch, dim=1)

            optimizer.zero_grad()
            pred_policy, pred_value = model(x_batch)

            """修正前コード
            loss_policy = criterion_policy(pred_policy, y_policy_labels_batch)
            loss_value = criterion_value(pred_value, y_value_batch)
            loss = loss_policy + loss_value

            loss.backward()
            optimizer.step()
            """

            # 修正後コード
            # Policy Lossを正しく計算する
            # pred_policy (ロジット) にLogSoftmaxを適用し、ターゲット確率分布とのクロスエントロピーを計算
            loss_policy = -torch.sum(y_policy_batch * F.log_softmax(pred_policy, dim=1), dim=1).mean()

            loss_value = criterion_value(pred_value, y_value_batch)
            loss = loss_policy + loss_value

            loss.backward()
            optimizer.step()

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_loss_policy += loss_policy.item() * batch_size
            total_loss_value += loss_value.item() * batch_size

            # ★★★ グローバルエポック数を計算 ★★★
            global_epoch = global_epoch_start_num + epoch

        avg_loss = total_loss / len(dataset)
        avg_policy_loss = total_loss_policy / len(dataset)
        avg_value_loss = total_loss_value / len(dataset)

        learning_rates.append(lr)
        total_losses.append(avg_loss)
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)

        print(f"Epoch {epoch+1}/{max_epochs} LR:{lr:.6f} "
              f"Total Loss:{avg_loss:.6f} Policy Loss:{avg_policy_loss:.6f} Value Loss:{avg_value_loss:.6f}")
        
        # ★★★ ログファイルへの追記形式を変更 ★★★
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = (f"{global_epoch},{timestamp},{lr:.6f},"
                     f"{avg_loss:.6f},{avg_policy_loss:.6f},{avg_value_loss:.6f}\n")
        
        with open(LOG_FILE, 'a') as f:
            f.write(log_entry)
        
        # 学習が進んでいないなら、次のデータを生成する
        if (avg_loss < best_loss):
            best_loss = avg_loss
            patience_counter = 0

            # 学習済みモデルの保存
            torch.save(model.state_dict(), './model/AlphaGomoku.pth')
        else:
            patience_counter += 1
        
        if (patience_counter >= patience_epochs):
            break
        
        epoch += 1

    # グラフ描画＆保存
    actual_epochs = range(1, len(total_losses) + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1,3,1)
    plt.plot(actual_epochs, total_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(actual_epochs, policy_losses, label='Policy Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Policy Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(actual_epochs, value_losses, label='Value Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Value Loss')
    plt.grid(True)
    plt.legend()

    # 保存先ディレクトリ作成
    save_dir = './Losses'
    os.makedirs(save_dir, exist_ok=True)

    # タイムスタンプ付きファイル名
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = os.path.join(save_dir, f'train_loss_{timestamp}.png')

    # 画像保存
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    train_network()
