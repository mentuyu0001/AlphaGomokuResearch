# A.py
import time
import random

print("--- プロセスAを開始 ---")
# 10秒から15秒のランダムな時間、処理を模倣して待機
sleep_time = random.randint(10, 15)
print(f"{sleep_time}秒間、処理を実行します...")
time.sleep(sleep_time)
print("--- プロセスAが完了 ---")