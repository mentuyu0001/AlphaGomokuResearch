import torch
import tkinter as tk
from tkinter import messagebox, simpledialog
from pathlib import Path
import os
import numpy as np

# 必要な自作モジュールをインポート
from GomokuGame import State, create_special_initial_state
from DualNetwork import AlphaGomokuNet
from PVmcts import pv_mcts_scores
import LearningParameters

# パラメータ読み込み
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BOARD_SIZE = LearningParameters.BOARD_SIZE

class GomokuGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Gomoku AI")
        # --- モデルの読み込み (パス解決を修正) ---
        try:
            script_dir = Path(__file__).parent.resolve()
            model_dir = script_dir / 'model'
            if not model_dir.is_dir():
                raise FileNotFoundError(f"モデルディレクトリが見つかりません: {model_dir}")
            
            model_paths = sorted(model_dir.glob('AlphaGomoku.pth'))
            if not model_paths:
                raise FileNotFoundError(f"モデルファイル(.pth)がディレクトリ内に見つかりません: {model_dir}")

            latest_model_path = model_paths[-1]
            self.model = AlphaGomokuNet().to(DEVICE)
            self.model.load_state_dict(torch.load(latest_model_path, map_location=DEVICE, weights_only=True))
            self.model.eval()
            print(f"AIモデル '{latest_model_path.name}' をロードしました。")

        except (FileNotFoundError, IndexError) as e:
            messagebox.showerror("エラー", f"モデルの読み込みに失敗しました。\n詳細: {e}")
            self.master.destroy()
            return
        
        # --- モード選択 ---
        self.human_is_black = messagebox.askyesno("手番選択", "あなたが先手(黒)でプレイしますか？\n(「いいえ」の場合は後手(白)になります)")
        
        self.state = create_special_initial_state() # 特殊な初期盤面で開始
        self.is_human_turn = self.human_is_black

        # --- 盤面の作成 ---
        self.cell_size = 50
        canvas_size = BOARD_SIZE * self.cell_size
        self.canvas = tk.Canvas(master, width=canvas_size, height=canvas_size, bg='#D2B48C') # 木目調の色
        self.canvas.pack(padx=10, pady=10)
        self.draw_board()

        self.draw_pieces() # 初期配置の石を描画
        
        self.canvas.bind('<Button-1>', self.on_click)

        self.update_title()

        # AIが先手の場合、最初のAIの手をスケジュール
        if not self.is_human_turn:
            self.master.after(500, self.ai_move)

    def draw_board(self):
        """盤面の格子線を描画する"""
        margin = self.cell_size // 2
        for i in range(BOARD_SIZE):
            # 縦線
            self.canvas.create_line(margin + i * self.cell_size, margin, margin + i * self.cell_size, margin + (BOARD_SIZE-1) * self.cell_size)
            # 横線
            self.canvas.create_line(margin, margin + i * self.cell_size, margin + (BOARD_SIZE-1) * self.cell_size, margin + i * self.cell_size)

    def on_click(self, event):
        """盤面がクリックされたときの処理"""
        if not self.is_human_turn or self.state.is_done():
            return
        
        # クリックされた座標からマス目を計算
        x = int((event.x - self.cell_size // 4) / self.cell_size)
        y = int((event.y - self.cell_size // 4) / self.cell_size)

        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return

        action = y * BOARD_SIZE + x

        if action in self.state.legal_actions():
            self.state = self.state.next(action)
            self.draw_pieces()
            self.is_human_turn = False
            self.update_title()
            
            if self.check_game_over(): return
            
            # AIの思考時間を少し待ってから実行
            self.master.after(200, self.ai_move)
        else:
            messagebox.showwarning("不正な手", "その場所には置けません。")

    def ai_move(self):
        """AIの手番処理 (最弱AIロジックに修正)"""
        if self.state.is_done(): return

        # MCTSで思考
        scores = pv_mcts_scores(self.model, self.state, temperature=1.0)
        
        if scores.size > 0:
            # 最も評価の低い手（負けそうな手）を選ぶ
            min_prob = np.max(scores)
            min_indices = np.where(scores == min_prob)[0]
            chosen_index = np.random.choice(min_indices)
            action = self.state.legal_actions()[chosen_index]
        else:
            # 探索結果がない場合はランダム
            action = np.random.choice(self.state.legal_actions())

        self.state = self.state.next(action)
        self.draw_pieces()
        self.is_human_turn = True
        self.update_title()

        self.check_game_over()

    def draw_pieces(self):
        """盤面の石を描画する (ロジックを修正)"""
        self.canvas.delete("pieces")
        
        # is_first_player() は「黒の手番か？」を意味する
        if self.state.is_first_player():
            black_pieces = self.state.pieces
            white_pieces = self.state.enemy_pieces
        else:
            white_pieces = self.state.pieces
            black_pieces = self.state.enemy_pieces

        self.draw_stone_set(black_pieces, 'black')
        self.draw_stone_set(white_pieces, 'white')

    def draw_stone_set(self, pieces, color):
        """指定された色の石を描画するヘルパー関数"""
        r = self.cell_size * 0.4
        margin = self.cell_size // 2
        for i in range(len(pieces)):
            if pieces[i] == 1:
                y, x = divmod(i, BOARD_SIZE)
                cx = margin + x * self.cell_size
                cy = margin + y * self.cell_size
                self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=color, outline='gray', tags="pieces")

    def check_game_over(self):
        """ゲーム終了をチェックし、メッセージを表示する"""
        if self.state.is_done():
            winner_msg = ""
            if self.state.is_lose():
                # is_lose() は手番プレイヤーの負け
                # is_human_turn は次の手番が人間かを示す
                if self.is_human_turn: # AIが打った後なので、AIの負け = 人間の勝ち
                    winner_msg = "AIの勝ちです！"
                else: # 人間が打った後なので、人間の負け = AIの勝ち
                    winner_msg = "あなたの勝ちです！"
            else: # 引き分け
                winner_msg = "引き分けです！"
            
            messagebox.showinfo("ゲーム終了", winner_msg)
            self.master.after(1000, self.master.destroy)
            return True
        return False
        
    def update_title(self):
        """ウィンドウのタイトルを更新する"""
        turn_text = "あなたの番" if self.is_human_turn else "AIの番"
        
        # 自分の手番か、かつ自分が黒か？ or 相手の手番か、かつ相手が黒か？
        is_black_turn = (self.is_human_turn and self.human_is_black) or \
                        (not self.is_human_turn and not self.human_is_black)
        color_text = "(黒)" if is_black_turn else "(白)"
        
        self.master.title(f"Gomoku (最弱AI) - {turn_text} {color_text}")

if __name__ == '__main__':
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()
