import torch
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
from GomokuGame import State
from DualNetwork import AlphaGomokuNet
from PVmcts import pv_mcts_action
import LearningParameters

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
board_size = LearningParameters.BOARD_SIZE

class GomokuGUI:
    def __init__(self, master):
        self.master = master

        # モード選択
        mode = simpledialog.askstring("モード選択", "モードを選んでください:\n'h' = Human vs AI（人間が先手）\n'a' = AI vs Human（AIが先手）\n'aa' = AI vs AI")
        if mode == 'h':
            self.human_turn = True
        elif mode == 'a':
            self.human_turn = False
        elif mode == 'aa':
            self.human_turn = None  # AI vs AI モード
        else:
            messagebox.showerror("エラー", "無効な入力。プログラムを終了します。")
            self.master.destroy()
            return

        self.master.title(f"Play AlphaGomoku ({board_size}x{board_size})")

        # モデル読み込み
        model_path = './model/AlphaGomoku.pth'
        self.model = AlphaGomokuNet().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        self.model.eval()
        self.mcts_player = pv_mcts_action(self.model, temperature=0)

        self.state = State()

        # 盤面作成
        self.buttons = []
        for y in range(board_size):
            row = []
            for x in range(board_size):
                btn = tk.Button(master, text='-', width=3, height=1,
                                command=lambda pos=x + y * board_size: self.on_click(pos),
                                font=('Arial', 16))
                btn.grid(row=y, column=x)
                row.append(btn)
            self.buttons.append(row)

        self.update_board()

        # 対局開始
        if self.human_turn is None:
            self.master.after(500, self.ai_vs_ai)
        elif not self.human_turn:
            self.master.after(500, self.ai_move)

    def on_click(self, pos):
        if not self.human_turn or self.state.is_done():
            return
        if pos not in self.state.legal_actions():
            messagebox.showwarning("Illegal Move", "その手はできません。")
            return

        self.state = self.state.next(pos)
        self.human_turn = False
        self.update_board()

        if self.state.is_done():
            self.show_result()
        else:
            self.master.after(100, self.ai_move)

    def ai_move(self):
        if self.state.is_done():
            self.show_result()
            return

        action = self.mcts_player(self.state)
        self.state = self.state.next(action)
        self.human_turn = True
        self.update_board()

        if self.state.is_done():
            self.show_result()

    def ai_vs_ai(self):
        if self.state.is_done():
            self.show_result()
            return

        action = self.mcts_player(self.state)
        self.state = self.state.next(action)
        self.update_board()

        # 次の手を0.5秒後に実行
        self.master.after(500, self.ai_vs_ai)

    def update_board(self):
        for y in range(board_size):
            for x in range(board_size):
                i = x + y * board_size
                if self.state.pieces[i] == 1:
                    char = 'o' if self.state.is_first_player() else 'x'
                elif self.state.enemy_pieces[i] == 1:
                    char = 'x' if self.state.is_first_player() else 'o'
                else:
                    char = '-'
                self.buttons[y][x].config(text=char)

    def show_result(self):
        if self.state.is_lose():
            if self.human_turn is None:
                msg = "AI後手の勝ちです！" if self.state.is_first_player() else "AI先手の勝ちです！"
            elif self.human_turn:
                msg = "AlphaGomokuの勝ちです！"
            else:
                msg = "あなたの勝ちです！"
        else:
            msg = "引き分けです！"
        messagebox.showinfo("ゲーム終了", msg)
        self.master.quit()

def main():
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
