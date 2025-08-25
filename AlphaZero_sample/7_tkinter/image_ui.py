import tkinter as tk
from PIL import Image, ImageTk

# イメージUIの定義
class ImageUI(tk.Frame):
    # 初期化
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        # タイトルの表示
        self.master.title('イメージの描画')

        # イメージの読み込み
        image = Image.open('sample.png')
        self.images = []
        self.images.append(ImageTk.PhotoImage(image))
        self.images.append(ImageTk.PhotoImage(image.rotate(180)))

        # キャンバスの生成
        self.c = tk.Canvas(self, width = 240, height = 240, highlightthickness = 0)
        self.c.pack()

        # 描画の更新
        self.on_draw()


    # 描画の更新
    def on_draw(self):
        # 描画のクリア
        self.c.delete('all')

        # イメージの描画
        self.c.create_image(10, 10, image=self.images[0],  anchor=tk.NW)

        # 反転イメージの描画
        self.c.create_image(10, 100, image=self.images[1],  anchor=tk.NW)

# イメージUIの実行
f = ImageUI()
f.pack()
f.mainloop()