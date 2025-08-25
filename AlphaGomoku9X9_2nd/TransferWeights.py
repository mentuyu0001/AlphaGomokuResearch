# transfer_weights.py

import torch
from DualNetwork import AlphaGomokuNet, save_model
import LearningParameters # 9x9の設定を読み込む

# --- 設定項目 ---
# 15x15学習済みモデルのパス
OLD_MODEL_PATH = 'C:/Users/sudok/Desktop/master_research_Miyazaki/gomoku/AlphaGomoku15X15/model/AlphaGomoku.pth' # 古いモデルのパスを正しく指定してください
# 9x9転移学習モデルの保存パス
NEW_MODEL_PATH = 'C:/Users/sudok/Desktop/master_research_Miyazaki/gomoku/AlphaGomoku9X9/model/AlphaGomoku.pth'
# -----------------

print("9x9用の新しいモデルを初期化します...")
# LearningParameters.pyに基づき、9x9用のモデルが作成される
new_model = AlphaGomokuNet()
new_state_dict = new_model.state_dict()

print(f"15x15用の学習済みモデル {OLD_MODEL_PATH} を読み込みます...")
old_state_dict = torch.load(OLD_MODEL_PATH, map_location='cpu')

# 新しいstate_dictに、古いモデルから重みをコピーしていく
for key in old_state_dict:
    # 新しいモデルにも同じ名前の層があり、かつ形状が一致する場合に重みをコピー
    if key in new_state_dict and old_state_dict[key].shape == new_state_dict[key].shape:
        new_state_dict[key] = old_state_dict[key]
        print(f"  ✅ 重みをコピー: {key}")

# 特別に処理が必要な最初の畳み込み層
input_conv_key = 'input_conv.0.weight'
if input_conv_key in old_state_dict and input_conv_key in new_state_dict:
    old_weight = old_state_dict[input_conv_key] # shape: (256, 3, 5, 5)
    new_weight = new_state_dict[input_conv_key]   # shape: (256, 2, 5, 5)
    
    # 形状が異なることを確認
    if old_weight.shape != new_weight.shape:
        print(f"  📝 入力層の重みを修正: {input_conv_key}")
        # 古い重みのうち、自分・相手の石に対応する2チャンネル分だけをコピー
        new_state_dict[input_conv_key] = old_weight[:, 0:2, :, :]

print("\n--- 転移サマリー ---")
print("✅ 転移できた層: 残差ブロック、畳み込み層など")
print("❌ 転移できなかった層 (再初期化): Policy/Valueヘッドの全結合層")
print("---------------------\n")

# 組み立てた新しい重みをモデルに読み込ませる
new_model.load_state_dict(new_state_dict)

# 新しいモデルを保存
save_model(new_model, path=NEW_MODEL_PATH)

print(f"転移学習モデルの保存が完了しました: {NEW_MODEL_PATH}")