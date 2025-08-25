# control.py
import subprocess
import time
import os

# --- 設定 ---
# 実行したいPythonスクリプトのファイル名
SCRIPT_A = "testA.py"
SCRIPT_B = "testB.py"

# スクリプトBを同時に実行する数
NUM_B_INSTANCES = 4
# --- ここまで ---

# スクリプトへの絶対パスを取得（予期せぬエラーを防ぐため）
script_a_path = os.path.abspath(SCRIPT_A)
script_b_path = os.path.abspath(SCRIPT_B)

# 実行サイクルをカウントする変数
loop_count = 0

try:
    # このループを繰り返すことで、処理を何度も実行します
    while True:
        loop_count += 1
        print(f"--- 実行サイクル #{loop_count} を開始します ---")

        # 起動したプロセスの情報を保存するリスト
        processes = []

        # PopenのcreationflagsにCREATE_NEW_CONSOLEを指定すると、
        # プロセスごとに新しいウィンドウが作成されます。
        new_window_flag = subprocess.CREATE_NEW_CONSOLE

        # --- A.pyを1つ起動 ---
        print(f"起動中: {script_a_path}")
        # PowerShellを起動し、その中でpythonコマンドを実行
        command_a = ['powershell', '-Command', 'python', script_a_path]
        proc_a = subprocess.Popen(command_a, creationflags=new_window_flag)
        processes.append(proc_a)

        # --- B.pyを4つ起動 ---
        for i in range(NUM_B_INSTANCES):
            instance_id = i + 1
            print(f"起動中: {script_b_path} (インスタンス #{instance_id})")
            # B.pyに引数としてインスタンス番号を渡すことも可能
            command_b = ['powershell', '-Command', 'python', script_b_path, str(instance_id)]
            proc_b = subprocess.Popen(command_b, creationflags=new_window_flag)
            processes.append(proc_b)

        print("\n>> 全てのスクリプトを起動しました。完了を待っています...")

        # --- 全てのプロセスの完了を待機 ---
        # processesリストに入っている全てのプロセスについて、wait()を呼び出す
        # wait()は、そのプロセスが完了するまで次の処理に進むのをブロックします
        for p in processes:
            p.wait()

        print(f"\n--- 実行サイクル #{loop_count} が完了しました ---")

        # 次のサイクルを開始する前に少し待機（任意）
        print("5秒後に次のサイクルを開始します...")
        time.sleep(5)

except FileNotFoundError:
    print("\nエラー: A.py または B.py が見つかりません。control.pyと同じフォルダに配置してください。")
except KeyboardInterrupt:
    # Ctrl+C が押されたら、ループを抜けてプログラムを終了
    print("\nスクリプトの実行がユーザーによって中断されました。")