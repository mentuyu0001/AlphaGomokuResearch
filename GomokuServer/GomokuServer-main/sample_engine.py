"""
着手可能位置からランダムに手を選ぶサンプルエンジン
"""

import random
import time

empties = set()

while True:
    cmd = input().split()

    if cmd[0] == "quit":
        break

    if cmd[0] == "pos":
        board, side = cmd[1:]
        
        empties.clear()
        for coord, s in enumerate(board):
            if s == '-':
                empties.add(coord)

    if cmd[0] == "move":
        move = int(cmd[1])
        empties.discard(move)

    if cmd[0] == "go":
        time_limit = int(cmd[1])
        time.sleep(time_limit / 1000)  # 特に意味もなく待つ

        move = random.choice(list(empties))
        empties.discard(move)
        print(f"move {move}")