import tkinter as tk

# グラフィックUIの定義
class GraphicUI(tk.Frame):
    # 初期化
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        # タイトルの表示
        self.master.title('グラフィックの描画')

        # キャンバスの生成
        self.c = tk.Canvas(self, width = 240, height = 240, highlightthickness = 0)
        self.c.pack()

        # 描画の更新
        self.on_draw()

    # 描画の更新
    def on_draw(self):
        # 描画のクリア
        self.c.delete('all')

        # ラインの描画
        self.c.create_line(10, 30, 230, 30, width = 2.0, fill = '#FF0000')

        # 円の描画
        self.c.create_oval(10, 70, 50, 110, width = 2.0, outline = '#00FF00')

        # 円の塗り潰し
        self.c.create_oval(70, 70, 110, 110, width = 0.0, fill = '#00FF00')

        # 矩形の描画
        self.c.create_rectangle(10, 130, 50, 170, width = 2.0, outline = '#00A0FF')

        # 矩形の塗り潰し
        self.c.create_rectangle(70, 130, 110, 170, width = 0.0, fill = '#00A0FF')

        # 文字列の表示
        self.c.create_text(10, 200, text = 'Hello World', font='courier 20', anchor = tk.NW)

# グラフィックUIの実行
f = GraphicUI()
f.pack()
f.mainloop()