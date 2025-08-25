# B.py
import time
import sys
import random

# control.pyから渡されたインスタンス番号を取得
instance_id = sys.argv[1] if len(sys.argv) > 1 else "N/A"

print(f"--- プロセスB (インスタンス #{instance_id}) を開始 ---")
# 5秒から10秒のランダムな時間、処理を模倣して待機
sleep_time = random.randint(5, 10)
print(f"インスタンス #{instance_id}: {sleep_time}秒間、処理を実行します...")
time.sleep(sleep_time)
print(f"--- プロセスB (インスタンス #{instance_id}) が完了 ---")