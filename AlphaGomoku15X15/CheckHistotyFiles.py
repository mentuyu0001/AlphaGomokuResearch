import pickle
import numpy as np

# 読み込み対象ファイル
path = "data/20250701143108.history"

with open(path, "rb") as f:
    history = pickle.load(f)  # history = List[List[state_tensor, pi, z]]

# 先頭から5手分だけ表示
for i in range(len(history)):
    s, pi, z = history[i]

    print(f"=== Move {i+1} ===")
    print("z:", z)
    print("pi:", np.round(pi, 3))  # policy配列
    print("state_tensor shape:", s.shape)
    print()
    #print(history)
