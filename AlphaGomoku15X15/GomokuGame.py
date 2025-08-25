import random
import numpy as np
import LearningParameters
import numba

# 定数定義
board_size = LearningParameters.BOARD_SIZE
win_count = LearningParameters.WIN_COUNT
board_len = LearningParameters.BOARD_LEN

# 勝利判定（引数: 石が置かれているマスのリスト）
@numba.jit(nopython=True, fastmath=True) # Numbaで高速化
def is_win(pieces_np):
    # pieces_np は 1次元のNumPy配列を想定
    for y in range(board_size):
        for x in range(board_size):
            # is_lineを内部で定義
            # 横方向
            if x <= board_size - win_count:
                if np.sum(pieces_np[y * board_size + x : y * board_size + x + win_count]) == win_count:
                    return True
            # 縦方向
            if y <= board_size - win_count:
                if np.sum(pieces_np[y * board_size + x : (y + win_count) * board_size + x : board_size]) == win_count:
                    return True
            # 右下斜め
            if x <= board_size - win_count and y <= board_size - win_count:
                diag = np.zeros(win_count, dtype=np.int8)
                for i in range(win_count):
                    diag[i] = pieces_np[(y + i) * board_size + (x + i)]
                if np.sum(diag) == win_count:
                    return True
            # 右上斜め
            if x <= board_size - win_count and y >= win_count - 1:
                diag = np.zeros(win_count, dtype=np.int8)
                for i in range(win_count):
                    diag[i] = pieces_np[(y - i) * board_size + (x + i)]
                if np.sum(diag) == win_count:
                    return True
    return False

# ゲーム状態クラス
class State:
    def __init__(self, pieces=None, enemy_pieces=None, history=None):
        self.pieces = pieces if pieces is not None else np.zeros(board_len, dtype=np.int8)
        self.enemy_pieces = enemy_pieces if enemy_pieces is not None else np.zeros(board_len, dtype=np.int8)

    def piece_count(self, pieces):
        return self.pieces.sum()

    def is_lose(self):
        # 相手が勝利条件を満たしているか
        return is_win(self.enemy_pieces)

    def is_draw(self):
        # 両者の石を合わせて盤面が埋まっていたら引き分け
        return self.pieces.sum() + self.enemy_pieces.sum() == board_len

    def is_done(self):
        return self.is_lose() or self.is_draw()

    def next(self, action):
         # ★★★ 履歴の引き継ぎをやめ、よりシンプルに ★★★
        new_pieces = self.pieces.copy()
        new_pieces[action] = 1
        return State(self.enemy_pieces, new_pieces)

    def legal_actions(self):
        # ★★★ NumPyのブロードキャストで合法手を高速に取得 ★★★
        # (self.pieces == 0) と (self.enemy_pieces == 0) の両方を満たすインデックスを返す
        return np.where((self.pieces == 0) & (self.enemy_pieces == 0))[0]

    def is_first_player(self):
        my_stones = self.pieces.sum()
        enemy_stones = self.enemy_pieces.sum()

        # 自分の石が相手より少ない場合、自分は先手（最初に多く置かれた側）
        if my_stones < enemy_stones:
            return True
        # 自分の石が相手より多い場合、自分は後手
        elif my_stones > enemy_stones:
            return False
        # 石の数が同数の場合、通常の五目並べと同様に先手番
        else:
            return True

    def __str__(self):
        # 盤面を文字列で表示（'o' と 'x' の交互）
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        s = ''
        for y in range(board_size):
            for x in range(board_size):
                i = x + y * board_size
                if self.pieces[i] == 1:
                    s += ox[0]
                elif self.enemy_pieces[i] == 1:
                    s += ox[1]
                else:
                    s += '-'
                s += " "
            s += '\n'
        return s
    
    def to_tensor(self):
        # (3, 15, 15) のテンソルを用意
        tensor = np.zeros(LearningParameters.DN_INPUT_SHAPE, dtype=np.float32)
        
        board_size = LearningParameters.BOARD_SIZE
        center_offset = 15 // 2 - board_size // 2

        # チャンネル0: 自分の石 (現在の手番のプレイヤー)
        my_plane = self.pieces.reshape(board_size, board_size)
        tensor[0, center_offset:center_offset+board_size, center_offset:center_offset+board_size] = my_plane

        # チャンネル 1: 相手の石
        enemy_plane = self.enemy_pieces.reshape(board_size, board_size)
        tensor[1, center_offset:center_offset+board_size, center_offset:center_offset+board_size] = enemy_plane
        
        # チャンネル 2: 外枠 (プレイエリア外は1、プレイエリア内は0)
        # まず15x15を全て1で埋める
        tensor[2, :, :] = 1.0
        # 中央の7x7のプレイエリアを0にする
        tensor[2, center_offset:center_offset+board_size, center_offset:center_offset+board_size] = 0.0
        
        return tensor



    
def random_action(state):
    return random.choice(state.legal_actions())

def playout(state):
    if state.is_lose():
        return -1
    if state.is_draw():
        return 0
    return -playout(state.next(random_action(state)))

def argmax(collection):
    return collection.index(max(collection))

# 特殊な初期盤面を生成する関数
def create_special_initial_state():
    pieces = np.zeros(board_len, dtype=np.int8)       # 先手 (自分) の石
    enemy_pieces = np.zeros(board_len, dtype=np.int8) # 後手 (相手) の石

    # 後手の石を配置 (4つ)
    # 各コーナーの6x6区画 (0-5, 6-8, 9-14行/列)
    corners = [
        (0, 6, 0, 6),   # 左上 (y_start, y_end, x_start, x_end)
        (0, 6, 9, 15),  # 右上
        (9, 15, 0, 6),  # 左下
        (9, 15, 9, 15)  # 右下
    ]

    # 後手の石を各コーナーに1つずつランダムに配置
    for y_start, y_end, x_start, x_end in corners:
        while True:
            y = random.randint(y_start, y_end - 1)
            x = random.randint(x_start, x_end - 1)
            action = y * board_size + x
            if pieces[action] == 0 and enemy_pieces[action] == 0:
                enemy_pieces[action] = 1
                break

    # 先手の石を配置 (1つ)
    # 十字型のエリア（四隅の6x6を除いたエリア）の座標リストを作成
    center_area_actions = []
    all_actions = set(range(board_len))
    corner_actions = set()
    for y_start, y_end, x_start, x_end in corners:
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                corner_actions.add(y * board_size + x)
    
    center_area_actions = list(all_actions - corner_actions)
    
    # 十字型エリアからランダムに1マス選び、先手の石を配置
    action = random.choice(center_area_actions)
    pieces[action] = 1
            
    # 初期状態のStateオブジェクトを生成して返す
    # is_first_playerの新しいロジックにより、この状態は先手番と判定される
    return State(pieces=pieces, enemy_pieces=enemy_pieces)

# 動作確認
if __name__ == '__main__':
    state = State()
    while True:
        if state.is_done():
            break
        state = state.next(random_action(state))
        print(state)
        print()