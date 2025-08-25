# ====================
# モンテカルロ木探索の作成
# ====================

# パッケージのインポート
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from GomokuGame import State
from DualNetwork import AlphaGomokuNet
import os
from math import sqrt
import LearningParameters
import numba


# シミュレーション回数
pv_evaluate_count = LearningParameters.PV_EVALUATE_COUNT

# 推論関数
def predict(model, state):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = state.to_tensor()  # shape (11, 15, 15), np.ndarray
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 11, 15, 15)

    model.eval()
    with torch.no_grad():
        policy, value = model(x_tensor)
        policy = torch.softmax(policy, dim=1)
        value = torch.sigmoid(value)

    policy = policy[0].cpu().numpy()
    value = value.cpu().item()

    """ 修正前コード

    legal = list(state.legal_actions())
    policies = policy[legal]
    """

    # 修正後コード
    board_size = LearningParameters.BOARD_SIZE # 7*7を15*15にパディングする作業
    center_offset = 15 // 2 - board_size // 2 # 7*7を15*15にパディングする作業

    legal_actions_7 = state.legal_actions() # [0..48] のリスト
    legal_actions_15 = []
    for action_7 in legal_actions_7:
        y7, x7 = divmod(action_7, board_size)
        y15, x15 = y7 + center_offset, x7 + center_offset
        legal_actions_15.append(y15 * 15 + x15)

    policies = policy[legal_actions_15] # 15x15空間のインデックスでスライス
    # ここまで修正

    policies /= np.sum(policies) if np.sum(policies) > 0 else 1

    # デバッグ用
    # print(f"Legal actions (7x7): {legal_actions_7}")
    # print(f"Legal actions (15x15): {legal_actions_15}")
    # print(f"Policy (filtered): {np.round(policies, 3)}")


    return policies, value


# ノード -> スコア配列変換
def nodes_to_scores(nodes):
    return [c.n for c in nodes]

# Numbaで高速化するPUCBスコア計算関数を新たに追加
@numba.jit(nopython=True, fastmath=True)
def find_best_child_jit(w_np, n_np, p_np, c_puct, t_sqrt):
    """
    NumbaでコンパイルされるPUCB計算のホットループ
    w_np, n_np, p_np: 子ノードのw, n, pをまとめたNumPy配列
    """
    # 最初の1手を選ぶ際(t_sqrtが0)は、純粋に方策pが最も高い手を選ぶ
    if t_sqrt == 0:
        return np.argmax(p_np)
    
    pucb_values = np.zeros(len(w_np), dtype=np.float32)
    for i in range(len(w_np)):
        q = w_np[i] / n_np[i] if n_np[i] > 0 else 0.0
        u = c_puct * p_np[i] * t_sqrt / (1 + n_np[i])
        pucb_values[i] = q + u
    
    # 最もスコアが高い子ノードのインデックスを返す
    return np.argmax(pucb_values)

# モンテカルロ木探索のスコア取得
def pv_mcts_scores(model, state, temperature):
    # ★★★ このチェックを追加 ★★★
    # ゲーム終了時は、探索を行わず空のスコアを返す
    if state.is_done():
        return []
    
    class Node:
        def __init__(self, state, p, parent=None):
            self.state = state
            self.p = p
            self.w = 0
            self.n = 0
            self.parent = parent
            self.child_nodes = None
        
        def select_child(self):
            # PUCBスコアが最大の子ノードを選択
            c_puct = LearningParameters.C_PUCT
            w_np = np.array([child.w for child in self.child_nodes], dtype=np.float32)
            n_np = np.array([child.n for child in self.child_nodes], dtype=np.float32)
            p_np = np.array([child.p for child in self.child_nodes], dtype=np.float32)
            t_sqrt = sqrt(self.n)
            best_child_index = find_best_child_jit(w_np, n_np, p_np, c_puct, t_sqrt)
            return self.child_nodes[best_child_index]

        def evaluate(self): # なんかよくわからないけど最弱のAIができてしまったので、勝ち負けの価値を反転させます
            if self.state.is_done():
                if self.state.is_lose():
                    value = 1  # 負け
                elif self.state.is_draw():
                    value = 0.5  # 引き分け
                else:
                    value = 0  # 勝ち
                self.w += value
                self.n += 1
                return value

            if not self.child_nodes:
                policies, value = predict(model, self.state)
                self.w += value
                self.n += 1
                self.child_nodes = [Node(self.state.next(a), p) for a, p in zip(self.state.legal_actions(), policies)]
                return value

            value = 1-self.next_child_node().evaluate() # 相手視点のvalueなため、1からvalueを引いてる
            self.w += value
            self.n += 1
            return value
        
        # numba 使用後
        def next_child_node(self):
            # ★★★ next_child_node を修正 ★★★
            c_puct = LearningParameters.C_PUCT
            
            # (1) 子ノードの情報をNumPy配列にまとめる
            w_list = [child.w for child in self.child_nodes]
            n_list = [child.n for child in self.child_nodes]
            p_list = [child.p for child in self.child_nodes]
            w_np = np.array(w_list, dtype=np.float32)
            n_np = np.array(n_list, dtype=np.float32)
            p_np = np.array(p_list, dtype=np.float32)

            t = np.sum(n_np)
            t_sqrt = sqrt(t) if t > 0 else 0

            # (2) Numbaで高速化した関数を呼び出して、最善手の子のインデックスを取得
            best_child_index = find_best_child_jit(w_np, n_np, p_np, c_puct, t_sqrt)
            
            # (3) インデックスを使って子ノードを返す
            return self.child_nodes[best_child_index]


    """修正前コード    
    # --- ここからが修正・追加箇所 ---
    # (1) ルートノードの作成
    root_node = Node(state, 0)

    # (2) ルートノードで1度推論を実行し、子ノードを展開する
    policies, _ = predict(model, root_node.state)

    # (3) ディリクレノイズを方策に加える ★★★ここが最重要★★★
    if temperature > 0: # 学習時のみノイズを加える
        # AlphaGoの論文に基づいたパラメータ (alpha=0.3, epsilon=0.25)
        alpha = 0.3
        epsilon = 0.25
        noise = np.random.dirichlet([alpha] * len(policies))
        policies = (1 - epsilon) * policies + epsilon * noise

    root_node.child_nodes = [Node(root_node.state.next(a), p) for a, p in zip(root_node.state.legal_actions(), policies)]

    # (4) MCTSシミュレーションの実行（ルートノードの子から探索を始める）
    for _ in range(pv_evaluate_count):
        # root_node.evaluate() ではなく、子ノードから評価を開始する
        root_node.next_child_node().evaluate()

    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:
        if not scores: return np.array([]) # 安全策
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        if not scores: return np.array([]) # 安全策

        # ★★★ ここでリストをNumPy配列に変換 ★★★
        scores_np = np.array(scores)
        scores = boltzmann(scores_np, temperature)
    return scores
    """
    """修正後コード"""
    # --- MCTSのメイン処理 ---
    # (1) ルートノードの作成
    root_node = Node(state, 0)
    
    # (2) シミュレーションを指定回数実行
    for _ in range(pv_evaluate_count):
        node = root_node
        
        # Selection: 葉ノードまで選択を繰り返す
        while node.child_nodes is not None:
            node = node.select_child()

        # Expansion & Evaluation: 葉ノードを展開し、NNで評価
        # ゲームが終了していない場合
        if not node.state.is_done():
            # NNで方策と価値を予測
            policies, value = predict(model, node.state)
            
            # 展開したノードの価値をバックアップの起点とする
            node.child_nodes = []
            legal_actions = node.state.legal_actions()
            
            # ルートノードの展開時のみディリクレノイズを加える
            if node.parent is None and temperature > 0:
                alpha = 0.3
                epsilon = 0.25
                if policies.size > 0:
                    noise = np.random.dirichlet([alpha] * len(policies))
                    policies = (1 - epsilon) * policies + epsilon * noise

            for action, p in zip(legal_actions, policies):
                node.child_nodes.append(Node(node.state.next(action), p, parent=node))
        
        # ゲームが終了している場合
        else: # なんかよくわからないけど最弱のAIができてしまったので、勝ち負けの価値を反転させます
            if node.state.is_lose(): value = 1
            elif node.state.is_draw(): value = 0.5
            else: value = 0

        # Backup: 価値をルートまで逆伝播させる
        while node is not None:
            # 自分の手番から見た価値に変換して加算
            node.w += value
            node.n += 1
            node = node.parent
            # 親の視点に価値を反転
            value = 1 - value
            
    # --- 探索結果から方策(訪問回数の比率)を計算 ---
    if not root_node.child_nodes:
        return np.array([])
        
    visit_counts = np.array([child.n for child in root_node.child_nodes])
    if np.sum(visit_counts) == 0:
        return np.array([])

    return visit_counts / np.sum(visit_counts)

# アクション選択関数
def pv_mcts_action(model, temperature=0):
    def act(state):
        scores = pv_mcts_scores(model, state, temperature)
        # ★★★ ここに修正を反映 ★★★
        # scoresが空(ゲーム終了局面)の場合は、合法手の中からランダムに手を選ぶ
        # (ただし、ゲーム終了局面で合法手は無いはずなので、これは主にエラー防止)
        if len(scores) == 0:
            # 安全のため、合法手があるかチェック
            legal = state.legal_actions()
            if not legal:
                return None # または適切なエラー処理
            return np.random.choice(legal)
        return np.random.choice(state.legal_actions(), p=scores)
    return act

# ボルツマン分布
def boltzmann(scores_np, temperature):
    # scores_npはNumPy配列を想定
    scores_np = scores_np.astype(np.float64)
    logits = scores_np / temperature
    e_x = np.exp(logits - np.max(logits)) # オーバーフロー防止
    return e_x / np.sum(e_x)

# 動作確認用
if __name__ == '__main__':
    model_path = sorted(Path('./model').glob('*.pth'))[-1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlphaGomokuNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    state = State()
    next_action = pv_mcts_action(model, temperature=1.0)  # 学習時は1.0

    while not state.is_done():
        action = next_action(state)
        state = state.next(action)
        print(state)