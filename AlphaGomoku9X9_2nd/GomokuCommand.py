import torch
import numpy as np
import time
from pathlib import Path
import sys
import os

# 必要な自作モジュールをインポート
from GomokuGame import State, create_special_initial_state
from DualNetwork import AlphaGomokuNet
from PVmcts import pv_mcts_scores
import LearningParameters

# パラメータ読み込み
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BOARD_SIZE = LearningParameters.BOARD_SIZE

def print_policy_grid(title, legal_actions, policy_probs):
    """AIの思考をグリッド形式で表示する関数"""
    print(f"\n[AI思考情報] {title}:")
    policy_grid = np.full((BOARD_SIZE, BOARD_SIZE), " --- ")

    if legal_actions.size > 0 and policy_probs.size > 0:
        for act, prob in zip(legal_actions, policy_probs):
            y, x = divmod(act, BOARD_SIZE)
            policy_grid[y, x] = f"{int(prob * 100):>3d}%"

    header = "   " + "".join([f"  {i:^3} " for i in range(BOARD_SIZE)])
    print(header)
    print("  " + "------" * BOARD_SIZE + "-")
    for y in range(BOARD_SIZE):
        row_str = f"{y} |" + "".join(policy_grid[y]) + " |"
        print(row_str)
    print("  " + "------" * BOARD_SIZE + "-")

# --- プレイヤーの定義 ---

def ai_player(model):
    """MCTSの探索結果から、最も確率の高い手を選ぶAI"""
    def get_action(state):
        # MCTSを実行して方策（各手の訪問回数の割合）を取得
        mcts_policy = pv_mcts_scores(model, state, temperature=0) # temperature=1.0で多様な手を考慮

        # 合法手がない、または探索結果がない場合はランダムな手を選ぶ
        if mcts_policy.size == 0 or state.legal_actions().size == 0:
            return {'action': np.random.choice(state.legal_actions())}

        # (1) MCTS方策の中から最大の確率値を見つける
        min_prob = np.max(mcts_policy)

        # (2) 最小の確率値を持つ全ての手のインデックスを取得する
        min_indices = np.where(mcts_policy == min_prob)[0]

        # (3) 最大値を持つインデックスの中から、ランダムに1つを選ぶ
        chosen_index = np.random.choice(min_indices)

        # (4) 選ばれたインデックスを使って、実際のアクションを決定する
        action = state.legal_actions()[chosen_index]

        # デバッグや表示用に情報を辞書で返す
        return {
            'action': action,
            'mcts_policy': mcts_policy,
        }
    get_action.__name__ = 'ai_player'
    return get_action

# --- ここからルールベースAIの実装 ---

def rule_based_player():
    """
    シンプルなルールベースのAI。
    「勝利手」のみを探索し、それ以外はランダムな手を選択する。
    """
    def get_action(state):
        legal_actions = state.legal_actions()

        # 優先度1: 自分が次に打てば勝てる手（五目）を探す
        if legal_actions.size > 0:
            for action in legal_actions:
                # 自分が action に打ったと仮定した次の状態を生成
                temp_state = state.next(action)
                # is_lose()は「現在の（相手）手番プレイヤー」の負けを判定する。
                # つまり、自分が手を打った結果、相手が負ける＝自分の勝ち。
                if temp_state.is_lose():
                    return {'action': action}

        # 優先度2: 勝利手がない場合、ランダムな手を選ぶ
        if legal_actions.size > 0:
            return {'action': np.random.choice(legal_actions)}
        else:
            # ゲーム終了などで合法手がない場合
            return {'action': -1}

    get_action.__name__ = 'Rule-based'
    return get_action

def random_player():
    """ランダムな手を打つプレイヤー"""
    def get_action(state):
        return {'action': np.random.choice(state.legal_actions())}
    get_action.__name__ = 'random_player'
    return get_action

# --- ここからルールベースAIの実装 ---

# ルールベースAIのためのヘルパー関数
def _find_critical_move(board_to_check, legal_actions, n):
    """
    指定された盤面(board_to_check)上で、合法手(legal_actions)の中から
    石を置くとちょうどn個のラインができる手を探す。
    """
    board_size = BOARD_SIZE
    # board_to_check: state.pieces または state.enemy_pieces を想定
    if legal_actions.size == 0:
        return None

    for action in legal_actions:
        y, x = divmod(action, board_size)
        # 4方向（横、縦、右下がり、右上がり）をチェック
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1 # actionに石を置くので1からスタート
            # 正の方向をチェック
            for i in range(1, n):
                ny, nx = y + i * dy, x + i * dx
                if not (0 <= ny < board_size and 0 <= nx < board_size and board_to_check[ny * board_size + nx] == 1):
                    break
                count += 1
            # 負の方向をチェック
            for i in range(1, n):
                ny, nx = y - i * dy, x - i * dx
                if not (0 <= ny < board_size and 0 <= nx < board_size and board_to_check[ny * board_size + nx] == 1):
                    break
                count += 1
            
            # n個以上のラインができていれば、その手を返す
            if count >= n:
                return action
    return None


def rule_based_player():
    """
    ルールベースAI（GomokuGame.py対応の完成版）。
    五目・四目を判断して、より賢い手を打つ。
    """
    def get_action(state):
        legal_actions = state.legal_actions()
        if legal_actions.size == 0:
            return {'action': -1}

        # 優先度1: 自分が勝てる手（五目）を探す
        win_move = _find_critical_move(state.pieces, legal_actions, 5)
        if win_move is not None:
            return {'action': win_move}

        # 優先度2: 相手が勝つ手（五目）を防ぐ
        block_win_move = _find_critical_move(state.enemy_pieces, legal_actions, 5)
        if block_win_move is not None:
            return {'action': block_win_move}

        # 優先度3: 自分がリーチをかける手（四目）を探す
        reach_move = _find_critical_move(state.pieces, legal_actions, 4)
        if reach_move is not None:
            return {'action': reach_move}

        # 優先度4: 相手のリーチを防ぐ手（四目）を探す
        block_reach_move = _find_critical_move(state.enemy_pieces, legal_actions, 4)
        if block_reach_move is not None:
            return {'action': block_reach_move}
            
        # 優先度5: 上記に当てはまらない場合、ランダムな手を選ぶ
        return {'action': np.random.choice(legal_actions)}

    get_action.__name__ = 'Rule-based'
    return get_action

# --- ゲーム進行のメインループ ---
def game_loop(player1, player2, verbose=True):
    state = create_special_initial_state() # ここを変更
    turn = 0
    while not state.is_done():
        turn += 1
        current_player = player1 if state.is_first_player() else player2

        if verbose:
            # os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Turn {turn} ---")
            print(f"手番: {'先手(o)' if state.is_first_player() else '後手(x)'} ({current_player.__name__})")
            print(state)

        result = current_player(state)
        action = result['action']

        if verbose and 'mcts_policy' in result:
            #print_policy_grid("MCTS探索後の方策 (AIが参照した確率)", state.legal_actions(), result['mcts_policy'])
            y, x = divmod(action, BOARD_SIZE)
            print(f"\nAIの選択: ({y}, {x})")

        state = state.next(action)
        if verbose: time.sleep(1)

    if verbose:
        # os.system('cls' if os.name == 'nt' else 'clear')
        print(f"\n--- ゲーム終了 (Turn {turn}) ---")
        print(state)
        winner = "引き分け"
        if state.is_lose():
            # is_lose()は手番プレイヤーの負けを意味する
            winner = "後手(x)" if state.is_first_player() else "先手(o)"
        print(f"結果: {winner} の勝ち")

    return state

def calculate_win_rate(player1_func, player2_func, num_games=50):
    """汎用的な勝率計算関数"""
    p1_name = player1_func.__name__
    p2_name = player2_func.__name__
    
    p1_wins = 0
    p2_wins = 0
    draws = 0

    print(f"\n--- {p1_name}(先手) vs {p2_name}(後手) を{num_games}回対戦 ---")
    for i in range(num_games):
        final_state = game_loop(player1_func, player2_func, verbose=False)
        if final_state.is_lose() and not final_state.is_first_player(): # player1(先手)の勝ち
            p1_wins += 1
        elif final_state.is_lose() and final_state.is_first_player(): # player2(後手)の勝ち
            p2_wins += 1
        else:
            draws += 1
        sys.stdout.write(f'\r対局中... {i + 1}/{num_games}')
        sys.stdout.flush()
    print()

    # 先手・後手入れ替え
    print(f"\n--- {p2_name}(先手) vs {p1_name}(後手) を{num_games}回対戦 ---")
    for i in range(num_games):
        final_state = game_loop(player2_func, player1_func, verbose=False)
        if final_state.is_lose() and not final_state.is_first_player(): # player2(先手)の勝ち
            p2_wins += 1
        elif final_state.is_lose() and final_state.is_first_player(): # player1(後手)の勝ち
            p1_wins += 1
        else:
            draws += 1
        sys.stdout.write(f'\r対局中... {i + 1}/{num_games}')
        sys.stdout.flush()
    print()

    total_games = num_games * 2
    p1_win_rate = (p1_wins / total_games) * 100
    p2_win_rate = (p2_wins / total_games) * 100
    draw_rate = (draws / total_games) * 100

    print("\n---------- 勝率計算結果 ----------")
    print(f"総対戦数: {total_games}回")
    print(f" {p1_name} の勝利数: {p1_wins}回 ({p1_win_rate:.2f}%)")
    print(f" {p2_name} の勝利数: {p2_wins}回 ({p2_win_rate:.2f}%)")
    print(f" 引き分け: {draws}回 ({draw_rate:.2f}%)")
    print("----------------------------------")

# --- メイン処理 ---
if __name__ == '__main__':
    ai = None
    try:
        # --- パス解決のロジックを修正 ---
        # このスクリプト自身の場所を基準にする
        script_dir = Path(__file__).parent.resolve()
        model_dir = script_dir / 'model'

        if not model_dir.is_dir():
            raise FileNotFoundError(f"モデルディレクトリが見つかりません: {model_dir}")

        # modelディレクトリ内の.pthファイルを探す
        model_paths = sorted(model_dir.glob('AlphaGomoku.pth'))
        if not model_paths:
            raise FileNotFoundError(f"モデルファイル(.pth)がディレクトリ内に見つかりません: {model_dir}")

        # 最新のモデルを選択
        latest_model_path = model_paths[-1]

        model = AlphaGomokuNet().to(DEVICE)
        model.load_state_dict(torch.load(latest_model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        ai = ai_player(model)
        print(f"AIモデル '{latest_model_path.name}' をロードしました。")

    except (FileNotFoundError, IndexError) as e:
        print(f"\n警告: AIモデルの読み込みに失敗しました。AIを含むモードは選択できません。")
        print(f"エラー詳細: {e}")
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}")

    while True:
        command = input().split()
        
        if (command[0] == 'pos'):
            if (command[2])
        elif (command[0] == 'move'):
        elif (command[0] == 'go'):
        elif (command[0] == 'quit'):
            break