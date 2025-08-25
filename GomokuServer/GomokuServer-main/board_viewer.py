#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5目並べの碁盤描画GUI
Tkinterを使用した実装
"""

import sys
import threading
import tkinter as tk
from tkinter import messagebox
import math


class GomokuBoardViewer:
    def __init__(self, initial_board_size=9):
        self.board_size = initial_board_size
        self.board_state = ['-'] * (initial_board_size * initial_board_size)
        self.current_player = '-'
        
        # GUIの初期化
        self.root = tk.Tk()
        self.root.title("5目並べ")
        self.root.geometry("600x650")
        
        # 現在の手番表示ラベル
        self.player_label = tk.Label(self.root, text="Current Player: None", 
                                   font=("Arial", 16), bg="lightgray")
        self.player_label.pack(pady=10)
        
        # キャンバスの作成
        self.canvas = tk.Canvas(self.root, bg="brown", width=580, height=580)
        self.canvas.pack(padx=10, pady=10, expand=True, fill="both")
        
        # リサイズイベントのバインド
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        # 初期描画
        self.draw_board()
        
        # 標準入力を読み込むスレッドを開始
        self.input_thread = threading.Thread(target=self.read_input, daemon=True)
        self.input_thread.start()
    
    def coordinate_to_index(self, row, col):
        """座標をインデックスに変換"""
        return row * self.board_size + col
    
    def index_to_coordinate(self, index):
        """インデックスを座標に変換"""
        row = index // self.board_size
        col = index % self.board_size
        return row, col
    
    def on_canvas_resize(self, event):
        """キャンバスがリサイズされた時の処理"""
        self.draw_board()
    
    def draw_board(self):
        """碁盤を描画"""
        self.canvas.delete("all")
        
        # キャンバスサイズの取得
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # 描画エリアのマージン
        margin = 40
        
        # 実際の描画エリアサイズ
        draw_width = canvas_width - 2 * margin
        draw_height = canvas_height - 2 * margin
        
        # 正方形にするため、小さい方に合わせる
        board_size_pixels = min(draw_width, draw_height)
        
        # 開始位置の計算（中央配置）
        start_x = (canvas_width - board_size_pixels) // 2
        start_y = (canvas_height - board_size_pixels) // 2
        
        # 線の間隔
        cell_size = board_size_pixels / (self.board_size - 1)
        
        # 背景を茶色で塗りつぶし
        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, 
                                   fill="burlywood", outline="")
        
        # 縦線の描画
        for i in range(self.board_size):
            x = start_x + i * cell_size
            self.canvas.create_line(x, start_y, 
                                  x, start_y + board_size_pixels, 
                                  fill="black", width=1)
        
        # 横線の描画
        for i in range(self.board_size):
            y = start_y + i * cell_size
            self.canvas.create_line(start_x, y, 
                                  start_x + board_size_pixels, y, 
                                  fill="black", width=1)
        
        # 星の描画（9x9以上の場合）
        if self.board_size >= 9:
            star_positions = []
            if self.board_size == 9:
                # 9路盤の星：四隅の三々と天元
                star_positions.extend([(2, 2), (2, 6), (6, 2), (6, 6)])
                # 天元（中央）
                center = self.board_size // 2
                star_positions.append((center, center))
            elif self.board_size >= 13:
                # 13路盤以上の星：四隅の三々と天元
                star_positions.extend([(3, 3), (3, self.board_size-4), 
                                     (self.board_size-4, 3), (self.board_size-4, self.board_size-4)])
                # 中央の星（奇数サイズの場合）
                if self.board_size % 2 == 1:
                    center = self.board_size // 2
                    star_positions.append((center, center))
            
            for row, col in star_positions:
                x = start_x + col * cell_size
                y = start_y + row * cell_size
                self.canvas.create_oval(x-3, y-3, x+3, y+3, 
                                      fill="black", outline="black")
        
        # 石の描画
        stone_radius = cell_size * 0.4
        for i, state in enumerate(self.board_state):
            if state != '-':
                row, col = self.index_to_coordinate(i)
                x = start_x + col * cell_size
                y = start_y + row * cell_size
                
                color = "black" if state == 'X' else "white"
                outline_color = "black"
                
                self.canvas.create_oval(x - stone_radius, y - stone_radius,
                                      x + stone_radius, y + stone_radius,
                                      fill=color, outline=outline_color, width=2)
    
    def update_board(self, board_string, current_player):
        """盤面を更新"""
        # 盤面サイズを文字列の長さから自動検知
        board_length = len(board_string)
        new_board_size = int(board_length ** 0.5)
        
        # 盤面サイズが変わった場合、リサイズ
        if new_board_size * new_board_size == board_length and new_board_size != self.board_size:
            self.board_size = new_board_size
            self.root.title(f"5目並べ ({self.board_size}x{self.board_size})")
        
        # 盤面状態と手番を更新
        self.board_state = list(board_string)
        self.current_player = current_player
        
        # メインスレッドで更新を実行
        self.root.after(0, self._update_gui)
    
    def _update_gui(self):
        """GUI更新（メインスレッドで実行）"""
        # 手番表示の更新
        player_text = "Current Player: "
        if self.current_player == 'X':
            player_text += "Black (X)"
        elif self.current_player == 'O':
            player_text += "White (O)"
        else:
            player_text += "None"
        
        self.player_label.config(text=player_text)
        
        # 盤面の再描画
        self.draw_board()
    
    def show_game_result(self, winner):
        """ゲーム結果を表示"""
        if winner == "none":
            message = "Draw.\nPress enter to go next."
        else:
            message = f"{winner} wins.\nPress enter to go next."
        
        # メインスレッドでメッセージボックスを表示
        self.root.after(0, lambda: self._show_message_box(message))
    
    def _show_message_box(self, message):
        """メッセージボックス表示（メインスレッドで実行）"""
        # canvasをリフレッシュ
        self.canvas.update_idletasks()
        messagebox.showinfo("Game Result", message)
        print("ok", flush=True)
    
    def read_input(self):
        """標準入力を読み込むスレッド"""
        try:
            while True:
                line = sys.stdin.readline().strip()
                if not line:
                    continue
                
                if line == "quit":
                    self.root.after(0, self.root.quit)
                    break
                elif line.startswith("winner "):
                    winner = line.split()[1]
                    self.show_game_result(winner)
                else:
                    # 局面の更新
                    parts = line.split()
                    if parts[0] == "pos" and len(parts) == 3:
                        board_string = parts[1]
                        current_player = parts[2]
                        # 盤面サイズチェックは update_board 内で行う
                        self.update_board(board_string, current_player)
        except EOFError:
            pass
        except Exception as e:
            print(f"Error reading input: {e}", file=sys.stderr)
    
    def run(self):
        """GUIを開始"""
        self.root.mainloop()


def main():
    # 初期盤面サイズ（後で動的に変更される）
    initial_board_size = 9
    if len(sys.argv) == 2:
        try:
            initial_board_size = int(sys.argv[1])
            if initial_board_size < 3:
                print("Board size must be at least 3", file=sys.stderr)
                sys.exit(1)
        except ValueError:
            print("Board size must be an integer", file=sys.stderr)
            sys.exit(1)
    
    # GUIを作成・実行
    viewer = GomokuBoardViewer(initial_board_size)
    viewer.run()


if __name__ == "__main__":
    main()