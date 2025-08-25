import os

# ここに'Losses'フォルダーの絶対パスを指定してください
# 例：
# Windows: 'C:\\Users\\YourUser\\Documents\\Losses'
# macOS/Linux: '/Users/YourUser/Documents/Losses'
folder_path = 'C:\\Users\\sudok\\Desktop\\master_research_Miyazaki\\gomoku\\AlphaGomoku9X9\\Losses' 

# ファイル数を初期化
file_count = 0

# 指定されたパスがディレクトリ（フォルダー）として存在するか確認
if os.path.isdir(folder_path):
    # フォルダー内の全てのアイテム（ファイルとサブフォルダー）のリストを取得
    for item in os.listdir(folder_path):
        # アイテムのフルパスを作成
        item_path = os.path.join(folder_path, item)
        # それがファイルであり、サブフォルダーではないことを確認
        if os.path.isfile(item_path):
            file_count += 1
            
    print(f"'{folder_path}' 内のファイル数: {file_count}")

else:
    print(f"エラー: 指定されたパス '{folder_path}' が見つからないか、フォルダーではありません。")