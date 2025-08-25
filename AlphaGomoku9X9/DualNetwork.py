import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import LearningParameters

# パラメータ
filters = LearningParameters.DN_FILTERS
residual_num = LearningParameters.DN_RESIDUAL_NUM
input_shape = LearningParameters.DN_INPUT_SHAPE
output_size = LearningParameters.DN_OUTPUT_SIZE

# 残差ブロック定義（3x3 conv ×2 + shortcut）
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Add shortcut
        return F.relu(out)

# 全体モデル定義
class AlphaGomokuNet(nn.Module):
    def __init__(self, in_channels=input_shape[0], channels=filters):
        super().__init__()

        # 5x5 畳み込み層（最初）
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # 残差ブロック × 5
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(residual_num)]
        )

        # --------------------
        # Policy head
        # Conv1x1 (256 → 2) → BN → ReLU → Flatten → FC(15x15x2 → 225)
        # Softmaxは出力しない（MCTS中に適用）
        # --------------------
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),  # (B,2,15,15)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),                              # → (B, 2*15*15)
            nn.Linear(2 * input_shape[1] * input_shape[2], output_size)     # → (B, 225)
        )

        # --------------------
        # Value head
        # Conv1x1 (256 → 1) → BN → ReLU → Flatten → FC(15x15 → 256) → ReLU → FC(256→1) → Tanh
        # --------------------
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),  # (B,1,15,15)
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),                              # → (B, 15*15)
            nn.Linear(input_shape[1] * input_shape[2], channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)

        policy = self.policy_head(x)  # ロジット出力（Softmaxなし）
        value = self.value_head(x)    # [0,1] の価値

        return policy, value


# モデル保存関数
def save_model(model, path='./model/AlphaGomoku.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

# モデル作成と保存
if __name__ == '__main__':
    if not os.path.exists('./model/AlphaGomoku.pth'):
        model = AlphaGomokuNet()
        save_model(model)
        del model
