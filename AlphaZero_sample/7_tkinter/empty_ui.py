import tkinter as tk

# 空UIの定義
class EmptyUI(tk.Frame):
    # 初期化
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        # タイトルの表示
        self.master.title('Hello World')

        # キャンバスの生成
        self.c = tk.Canvas(self, width = 240, height = 240, highlightthickness = 0)
        self.c.pack()

# 空UIの実行
f = EmptyUI()
f.pack()
f.mainloop()