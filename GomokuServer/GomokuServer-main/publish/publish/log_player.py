#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五目並べの棋譜再生GUI (修正版)
1桁と2桁が混在する棋譜ログに対応
"""

import tkinter as tk
from tkinter import messagebox

class GomokuLogViewer:
    def __init__(self, initial_board_size=9):
        # --- モデルデータ ---
        self.board_size = initial_board_size
        self.initial_board_state = ['-'] * (self.board_size * self.board_size)
        self.moves = []
        self.current_move_index = -1 # -1は初期盤面を示す

        # --- GUIの初期化 ---
        self.root = tk.Tk()
        self.root.title("Gomoku Log Player")
        self.root.geometry("600x750")

        # --- コントロールフレーム ---
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        log_label = tk.Label(control_frame, text="棋譜ログ:")
        log_label.pack(side=tk.LEFT, padx=(0, 5))

        self.log_entry = tk.Entry(control_frame, width=50)
        self.log_entry.pack(side=tk.LEFT, padx=5)

        self.load_button = tk.Button(control_frame, text="ログ読込", command=self.load_log)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # --- 再生コントロールフレーム ---
        playback_frame = tk.Frame(self.root)
        playback_frame.pack(pady=5)

        self.prev_button = tk.Button(playback_frame, text="< Prev", command=self.prev_move, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.move_label = tk.Label(playback_frame, text="Initial State", font=("Arial", 14), width=20)
        self.move_label.pack(side=tk.LEFT)

        self.next_button = tk.Button(playback_frame, text="Next >", command=self.next_move, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=10)
        
        # --- キャンバスの作成 ---
        self.canvas = tk.Canvas(self.root, bg="burlywood", width=580, height=580)
        self.canvas.pack(padx=10, pady=10, expand=True, fill="both")
        self.canvas.bind("<Configure>", lambda e: self.draw_board())
        
        # --- 初期盤面描画 ---
        self.draw_board()

    def index_to_coordinate(self, index):
        """インデックスを座標(row, col)に変換"""
        row = index // self.board_size
        col = index % self.board_size
        return row, col

    def parse_moves(self, moves_part):
        """1桁・2桁混在の棋譜文字列を解析する"""
        moves = []
        i = 0
        while i < len(moves_part):
            # 2桁の数値を先に試す
            if i + 1 < len(moves_part):
                num_str = moves_part[i:i+2]
                num = int(num_str)
                # 2桁の数値が有効な座標(0-80)の場合
                if 0 <= num <= 80:
                    moves.append(num)
                    i += 2
                    continue
            
            # 2桁が有効でない場合、または文字列の末尾の場合、1桁で試す
            num_str = moves_part[i:i+1]
            num = int(num_str)
            moves.append(num)
            i += 1
        return moves

    def load_log(self):
        """入力されたログをパースして読み込む"""
        log_string = self.log_entry.get().strip()
        if not log_string:
            messagebox.showerror("Error", "ログが入力されていません。")
            return
        
        try:
            parts = log_string.split()
            if len(parts) != 2:
                raise ValueError("ログは'[盤面] [棋譜]'の形式である必要があります。")

            board_part, moves_part = parts
            
            if len(board_part) != self.board_size * self.board_size:
                raise ValueError(f"盤面文字列の長さが{self.board_size * self.board_size}ではありません。")

            # --- ★ここが修正点★ ---
            # 新しい解析ロジックを呼び出す
            self.moves = self.parse_moves(moves_part)
            
            # データの初期化
            self.initial_board_state = list(board_part)
            self.current_move_index = -1
            
            # GUIの更新
            self.update_board_display()
            messagebox.showinfo("Success", f"ログを読み込みました。棋譜: {len(self.moves)}手")

        except (ValueError, IndexError) as e:
            messagebox.showerror("Error", f"ログの形式が正しくありません。\n{e}")
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)

    def update_board_display(self):
        """現在のインデックスに基づいて盤面を更新して表示"""
        # ボタンの状態更新
        self.prev_button.config(state=tk.NORMAL if self.current_move_index >= 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_move_index < len(self.moves) - 1 else tk.DISABLED)

        # 手数ラベルの更新
        if self.current_move_index == -1:
            self.move_label.config(text="Initial State")
        else:
            player = "先手(X)" if self.current_move_index % 2 == 0 else "後手(O)"
            move_pos = self.moves[self.current_move_index]
            r, c = self.index_to_coordinate(move_pos)
            self.move_label.config(text=f"Move {self.current_move_index + 1}: {player} at ({r+1}, {c+1})")
        
        # 盤面の再描画
        self.draw_board()

    def get_current_board_state(self):
        """現在のインデックスまでの棋譜を反映した盤面状態を返す"""
        board = list(self.initial_board_state)
        # 初期盤面にある石(X, O)はそのまま描画する
        for i in range(self.current_move_index + 1):
            move_pos = self.moves[i]
            # 棋譜の手番は初期盤面の手番に関係なく、0手目=先手, 1手目=後手..とする
            # ここでは初期盤面にXが含まれる場合を考慮し、棋譜の0手目は後手(O)かもしれない
            # ただし、提供されたログでは初期盤面にXがないため、0手目=先手(X)で固定する
            player = 'X' if i % 2 == 0 else 'O'
            if 0 <= move_pos < len(board):
                 board[move_pos] = player
        return board

    def next_move(self):
        """一手進める"""
        if self.current_move_index < len(self.moves) - 1:
            self.current_move_index += 1
            self.update_board_display()

    def prev_move(self):
        """一手戻す"""
        if self.current_move_index >= -1:
            self.current_move_index -= 1
            self.update_board_display()

    def draw_board(self):
        """碁盤と石を描画"""
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1: return
        
        margin = 40
        board_size_pixels = min(canvas_width - 2 * margin, canvas_height - 2 * margin)
        start_x = (canvas_width - board_size_pixels) / 2
        start_y = (canvas_height - board_size_pixels) / 2
        cell_size = board_size_pixels / (self.board_size - 1)
        
        # 盤面の線を描画
        for i in range(self.board_size):
            x = start_x + i * cell_size
            self.canvas.create_line(x, start_y, x, start_y + board_size_pixels, fill="black")
            y = start_y + i * cell_size
            self.canvas.create_line(start_x, y, start_x + board_size_pixels, y, fill="black")
            
            # 座標ラベル
            self.canvas.create_text(start_x + i * cell_size, start_y - 15, text=str(i + 1), fill="black")
            self.canvas.create_text(start_x - 15, start_y + i * cell_size, text=str(i + 1), fill="black")
        
        # 星の描画
        center = self.board_size // 2
        star_positions = [(2, 2), (2, 6), (6, 2), (6, 6), (center, center)]
        for r, c in star_positions:
            x = start_x + c * cell_size
            y = start_y + r * cell_size
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="black")
        
        # 石の描画
        board_state = self.get_current_board_state()
        stone_radius = cell_size * 0.45
        for i, state in enumerate(board_state):
            if state != '-':
                row, col = self.index_to_coordinate(i)
                x = start_x + col * cell_size
                y = start_y + row * cell_size
                color = "black" if state == 'X' else "white"
                self.canvas.create_oval(x - stone_radius, y - stone_radius,
                                      x + stone_radius, y + stone_radius,
                                      fill=color, outline="black", width=1)
        
        # 最新の石にマークをつける
        if self.current_move_index != -1:
            move_pos = self.moves[self.current_move_index]
            row, col = self.index_to_coordinate(move_pos)
            x = start_x + col * cell_size
            y = start_y + row * cell_size
            self.canvas.create_rectangle(x-5, y-5, x+5, y+5, outline='red', width=2)


    def run(self):
        """GUIを開始"""
        self.root.mainloop()

def main():
    viewer = GomokuLogViewer(initial_board_size=9)
    test_log = ""
    viewer.log_entry.insert(0, test_log)
    viewer.run()

if __name__ == "__main__":
    main()
