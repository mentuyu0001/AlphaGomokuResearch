import torch
import numpy as np
import sys
from pathlib import Path
import os

# 必要な自作モジュールをインポート
try:
    from GomokuGame import State # 修正対象のStateクラスを読み込む
    from DualNetwork import AlphaGomokuNet
    from PVmcts import pv_mcts_scores_by_time
    import LearningParameters
except ImportError as e:
    print(f"エラー: 必要なモジュールが見つかりません: {e}", file=sys.stderr)
    print("GomokuGame.py, DualNetwork.py, PVmcts.py, LearningParameters.py が同じディレクトリにあるか確認してください。", file=sys.stderr)
    sys.exit(1)

# --- グローバルパラメータ ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BOARD_SIZE = LearningParameters.BOARD_SIZE

# --- AIの思考部 ---
# (ai_player, _find_critical_move, rule_based_player 関数は変更ないため、ここでは省略します)
# (もし必要であれば、前回の回答からコピーしてください)

def ai_player(model):
    """
    MCTSに基づいて最適な手を判断するAIプレイヤー。
    思考時間を受け取れるように修正。
    """
    def get_action(state, time_limit_ms): # time_limit_ms を引数に追加
        # 時間ベースのMCTS関数を呼び出す
        scores = pv_mcts_scores_by_time(model, state, time_limit_ms, temperature=0)
        
        # スコア（訪問回数）が最も高い手を選択
        if scores.size == 0:
            # 時間切れなどで探索ができなかった場合、合法手からランダムに選択
            return {'action': np.random.choice(state.legal_actions())}
            
        best_action_index = np.argmax(scores)
        action = state.legal_actions()[best_action_index]

        return {'action': action}
    
    get_action.__name__ = 'ai_player'
    return get_action

def _find_critical_move(board_to_check, legal_actions, n):
    """指定された盤面上で、石を置くとn個の石が連なる「決定的な手」を探すヘルパー関数。"""
    if legal_actions.size == 0: return None
    for action in legal_actions:
        y, x = divmod(action, BOARD_SIZE)
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in [1, -1]:
                for i in range(1, n):
                    ny, nx = y + i * sign * dy, x + i * sign * dx
                    if 0 <= ny < BOARD_SIZE and 0 <= nx < BOARD_SIZE and board_to_check[ny * BOARD_SIZE + nx] == 1:
                        count += 1
                    else: break
            if count >= n: return action
    return None

def rule_based_player():
    """モデルが読み込めなかった場合に使用する、ルールベースのフォールバックAI。"""
    # ★★★ 修正点: get_actionが思考時間の引数を受け取るようにする ★★★
    def get_action(state, time_limit_ms=None): # time_limit_ms を引数に追加（ただし使用はしない）
        legal_actions = state.legal_actions()
        if legal_actions.size == 0: return {'action': -1}
        
        # 思考ロジックは変更なし
        win_move = _find_critical_move(state.pieces, legal_actions, 5)
        if win_move is not None: return {'action': win_move}
        block_win_move = _find_critical_move(state.enemy_pieces, legal_actions, 5)
        if block_win_move is not None: return {'action': block_win_move}
        reach_move = _find_critical_move(state.pieces, legal_actions, 4)
        if reach_move is not None: return {'action': reach_move}
        block_reach_move = _find_critical_move(state.enemy_pieces, legal_actions, 4)
        if block_reach_move is not None: return {'action': block_reach_move}
        
        return {'action': np.random.choice(legal_actions)}
        
    get_action.__name__ = 'rule_based_player'
    return get_action

# --- サーバー通信のコア機能 ---

def create_state_from_pos(s_board, turn_char):
    """
    サーバーから送られてきた 'pos S x' コマンドの情報から、
    提供されたStateクラスに適合したオブジェクトを生成する。（★★ 修正版 ★★）
    """
    # Stateクラスの仕様に基づき、手番プレイヤーの石を`pieces`に、
    # 相手の石を`enemy_pieces`に割り当てる。
    pieces = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.int8)
    enemy_pieces = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.int8)

    my_stone_char = turn_char
    enemy_stone_char = 'O' if turn_char == 'X' else 'X'

    for i, char in enumerate(s_board):
        if char == my_stone_char:
            pieces[i] = 1
        elif char == enemy_stone_char:
            enemy_pieces[i] = 1
    
    # Stateクラスのコンストラクタ`__init__(self, pieces=None, enemy_pieces=None)`を
    # 正しい引数で呼び出す。
    return State(pieces=pieces, enemy_pieces=enemy_pieces)


def engine_loop(player_ai, log_func):
    """
    サーバーからのコマンドを待ち受け、思考結果を返すエンジンのメインループ。
    """
    state = None
    log_func("エンジンが起動しました。サーバーからのコマンドを待っています...")

    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                continue

            log_func(f"受信: {line}")
            parts = line.split()
            command = parts[0]

            if command == "pos":
                s_board = parts[1]
                turn_char = parts[2]
                state = create_state_from_pos(s_board, turn_char)
                log_func(f"盤面を初期化しました。手番: {turn_char}")

            elif command == "move":
                action = int(parts[1])
                if state:
                    state = state.next(action)
                    log_func(f"相手の手: {action} を盤面に反映しました。")
                else:
                    log_func("エラー: 'pos'コマンドで盤面が初期化されていません。")

            elif command == "go":
                # `go <time>` コマンドで思考を開始
                time_left_ms = int(parts[1])
                log_func(f"思考開始。残り時間: {time_left_ms}ms")
                
                if state:
                    # player_aiに思考時間を渡す
                    result = player_ai(state, time_left_ms)
                    my_action = result['action']

                    print(f"move {my_action}", flush=True)
                    log_func(f"送信: move {my_action}")
                    
                    state = state.next(my_action)
                else:
                    log_func("エラー: 'pos'コマンドで盤面が初期化されておらず、思考できません。")

            elif command == "quit":
                log_func("終了コマンドを受信。エンジンを停止します。")
                break
            
            else:
                log_func(f"未定義のコマンドです: {command}")

        except Exception as e:
            log_func(f"致命的なエラーが発生しました: {e}")
            import traceback
            log_func(traceback.format_exc()) # 詳細なエラー内容をログに出力
            break


# --- メイン処理 ---
if __name__ == '__main__':
    log_file = open("engine_log.txt", "w", encoding='utf-8')
    def log(message):
        print(message, file=log_file, flush=True)
        print(message, file=sys.stderr, flush=True)

    ai_agent = None
    try:
        # --- モデル読み込み処理 (★★★ ここからが修正箇所 ★★★) ---

        # 1. このスクリプトファイル自身の場所を取得
        script_dir = Path(__file__).parent.resolve()
        
        # 2. スクリプトと同じ階層にある 'learnedModel' フォルダのパスを正しく作成
        model_dir = script_dir / "learnedModel"
        
        # 3. フォルダが存在するか確認
        if not model_dir.is_dir():
            raise FileNotFoundError(f"モデルが保存されているディレクトリが見つかりません: {model_dir}")

        # 4. 'learnedModel' フォルダの中から.pthファイルを検索
        model_paths = sorted(model_dir.glob('*.pth'))
        if not model_paths:
            raise FileNotFoundError(f"モデルファイル (.pth) がディレクトリ内に見つかりません: {model_dir}")

        # 5. 最新のモデルファイルを取得
        latest_model_path = model_paths[-1]
        
        # 6. モデルを読み込む
        model = AlphaGomokuNet().to(DEVICE)
        model.load_state_dict(torch.load(latest_model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        
        ai_agent = ai_player(model)
        log(f"AIモデル '{latest_model_path.name}' を正常にロードしました。")
        # (★★★ 修正箇所はここまで ★★★)

    except Exception as e:
        # ★★★ ここからが修正箇所 ★★★
        # モデルの読み込みに失敗した場合、エラーログを出力してプログラムを終了する
        log(f"致命的エラー: AIモデルの読み込みに失敗しました。プログラムを終了します。")
        log(f"エラー詳細: {e}")
        
        import traceback
        log(traceback.format_exc()) # デバッグ用に詳細なエラー情報をログに出力
        
        sys.exit(1) # エラーコード1でプログラムを終了

    # モデルの読み込みが成功した場合のみ、以下のエンジンループが実行される
    engine_loop(ai_agent, log)
    
    log_file.close()