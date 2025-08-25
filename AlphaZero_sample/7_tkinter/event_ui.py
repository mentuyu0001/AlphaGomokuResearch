import tkinter as tk
from PIL import Image, ImageTk

# イベントUIの定義
class EventUI(tk.Frame):
    # 初期化
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        # タイトルの表示
        self.master.title('イベント処理')

        # クリック位置
        self.x = 0
        self.y = 0

        # キャンバスの生成
        self.c = tk.Canvas(self, width = 240, height = 240, highlightthickness = 0)
        self.c.bind('<Button-1>', self.on_click) # クリック判定の追加
        self.c.pack()

        # 描画の更新
        self.on_draw()

    # クリック時に呼ばれる
    def on_click(self, event):
        self.x = event.x
        self.y = event.y
        self.on_draw()

    # 描画の更新
    def on_draw(self):
        # 描画のクリア
        self.c.delete('all')

        # 文字列の表示
        str = 'クリック位置 {},{}'.format(self.x, self.y)
        self.c.create_text(10, 10, text = str, font='courier 16', anchor = tk.NW)

# イベントUIの実行
f = EventUI()
f.pack()
f.mainloop()