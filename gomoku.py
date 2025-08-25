import tkinter as tk
from tkinter import messagebox
import numpy as np

BOARD_SIZE = 8
WIN_COUNT = 5

class GomokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("8×8 五目並べ")
        self.current_player = 1  # 1: 先手(X), -1: 後手(O)
        self.raw_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.buttons = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.create_board()

    def create_board(self):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                button = tk.Button(
                    self.root,
                    text="",
                    width=4,
                    height=2,
                    font=("Helvetica", 20),
                    command=lambda r=row, c=col: self.handle_click(r, c)
                )
                button.grid(row=row, column=col)
                self.buttons[row][col] = button

    def handle_click(self, row, col):
        if self.raw_board[row][col] != 0:
            return  # すでに置かれている

        # 内部盤面に反映
        self.raw_board[row][col] = self.current_player
        symbol = "X" if self.current_player == 1 else "O"
        self.buttons[row][col].config(text=symbol)

        if self.check_win(self.current_player, row, col):
            winner = "X" if self.current_player == 1 else "O"
            messagebox.showinfo("勝利！", f"プレイヤー {winner} の勝ちです！")
            self.reset_game()
            return

        if np.all(self.raw_board != 0):
            messagebox.showinfo("引き分け", "引き分けです。")
            self.reset_game()
            return

        # 特徴量を表示（必要に応じて推論に使う）
        feature = self.create_feature_board(self.raw_board)
        print("特徴量配列（8x8x3）:")
        print("先手:\n", feature[:, :, 0])
        print("後手:\n", feature[:, :, 1])
        print("全体:\n", feature[:, :, 2])

        # ターン交代
        self.current_player *= -1

    def check_win(self, player, row, col):
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            for d in [1, -1]:
                r, c = row, col
                while True:
                    r += d * dr
                    c += d * dc
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.raw_board[r][c] == player:
                        count += 1
                    else:
                        break
            if count >= WIN_COUNT:
                return True
        return False

    def create_feature_board(self, board):
        """AlphaZero形式の8x8x3配列に変換"""
        feature_board = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=int)
        feature_board[:, :, 0] = (board == 1).astype(int)   # 先手
        feature_board[:, :, 1] = (board == -1).astype(int)  # 後手
        feature_board[:, :, 2] = (board != 0).astype(int)   # 石がある
        return feature_board

    def reset_game(self):
        self.current_player = 1
        self.raw_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                self.buttons[row][col].config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    game = GomokuGUI(root)
    root.mainloop()
