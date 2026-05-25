import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

# Ladda ned från: https://drive.google.com/drive/folders/1h6edewgRUTJPzI81Mn0eSsqItnk9RMeO
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model-weights")
ATTENTIONHTR_MODEL_PATHS = {
    "attnHTR-general": os.path.join(MODEL_DIR, "AttentionHTR-General.pth"),
    "attnHTR-general-sensitive": os.path.join(
        MODEL_DIR, "AttentionHTR-General-sensitive.pth"
    ),
    "attnHTR-iam": os.path.join(MODEL_DIR, "AttentionHTR-IAM.pth"),
    "attnHTR-iam-sensitive": os.path.join(MODEL_DIR, "AttentionHTR-IAM-sensitive.pth"),
    "attnHTR-imgur5k": os.path.join(MODEL_DIR, "AttentionHTR-Imgur5K.pth"),
    "attnHTR-imgur5k-sensitive": os.path.join(
        MODEL_DIR, "AttentionHTR-Imgur5K-sensitive.pth"
    ),
}
MAX_IMAGES = 10

IMG_H = 32
IMG_W = 100
INPUT_CHANNEL = 1
OUTPUT_CHANNEL = 512
HIDDEN_SIZE = 256
NUM_FIDUCIAL = 20
attentionhtr_transforms = Compose(
    [
        Grayscale(num_output_channels=1),
        Resize((IMG_H, IMG_W)),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,)),
    ]
)

# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------


class LocalizationNetwork(nn.Module):
    def __init__(self, F, I_channel_num):
        super().__init__()
        self.F = F
        self.conv = nn.Sequential(
            nn.Conv2d(I_channel_num, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, F * 2)
        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_top = np.stack(
            [ctrl_pts_x, np.linspace(0.0, -1.0, int(F / 2))], axis=1
        )
        ctrl_pts_bottom = np.stack(
            [ctrl_pts_x, np.linspace(1.0, 0.0, int(F / 2))], axis=1
        )
        self.localization_fc2.bias.data = (
            torch.from_numpy(np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0))
            .float()
            .view(-1)
        )

    def forward(self, batch_I):
        B = batch_I.size(0)
        return self.localization_fc2(
            self.localization_fc1(self.conv(batch_I).view(B, -1))
        ).view(B, self.F, 2)


class GridGenerator(nn.Module):
    eps = 1e-6

    def __init__(self, F, I_r_size):
        super().__init__()
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        C = self._build_C(F)
        P = self._build_P(self.I_r_width, self.I_r_height)
        self.register_buffer(
            "inv_delta_C", torch.tensor(self._build_inv_delta_C(F, C)).float()
        )
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(F, C, P)).float())

    def _build_C(self, F):
        x = np.linspace(-1.0, 1.0, int(F / 2))
        return np.concatenate(
            [
                np.stack([x, -np.ones(int(F / 2))], axis=1),
                np.stack([x, np.ones(int(F / 2))], axis=1),
            ]
        )

    def _build_inv_delta_C(self, F, C):
        hat_C = np.zeros((F, F))
        for i in range(F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = hat_C**2 * np.log(hat_C)
        delta_C = np.concatenate(
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),
                np.concatenate([np.zeros((2, 3)), C.T], axis=1),
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1),
            ]
        )
        return np.linalg.inv(delta_C)

    def _build_P(self, W, H):
        gx = (np.arange(-W, W, 2) + 1.0) / W
        gy = (np.arange(-H, H, 2) + 1.0) / H
        return np.stack(np.meshgrid(gx, gy), axis=2).reshape(-1, 2)

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]
        rbf_norm = np.linalg.norm(P[:, None, :] - C[None, :, :], ord=2, axis=2)
        rbf = rbf_norm**2 * np.log(rbf_norm + self.eps)
        return np.concatenate([np.ones((n, 1)), P, rbf], axis=1)

    def build_P_prime(self, batch_C_prime):
        B = batch_C_prime.size(0)
        batch_C_prime_with_zeros = torch.cat(
            [batch_C_prime, torch.zeros(B, 3, 2, device=batch_C_prime.device)], dim=1
        )
        batch_T = torch.bmm(self.inv_delta_C.repeat(B, 1, 1), batch_C_prime_with_zeros)
        return torch.bmm(self.P_hat.repeat(B, 1, 1), batch_T)


class TPS_SpatialTransformerNetwork(nn.Module):
    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        super().__init__()
        self.I_r_size = I_r_size
        self.LocalizationNetwork = LocalizationNetwork(F, I_channel_num)
        self.GridGenerator = GridGenerator(F, I_r_size)

    def forward(self, batch_I):
        P_prime = self.GridGenerator.build_P_prime(self.LocalizationNetwork(batch_I))
        grid = P_prime.reshape([P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        return F.grid_sample(batch_I, grid, padding_mode="border", align_corners=True)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers):
        super().__init__()
        ch = [output_channel // 4, output_channel // 2, output_channel, output_channel]
        self.inplanes = output_channel // 8
        self.conv0_1 = nn.Conv2d(
            input_channel, output_channel // 16, 3, 1, 1, bias=False
        )
        self.bn0_1 = nn.BatchNorm2d(output_channel // 16)
        self.conv0_2 = nn.Conv2d(
            output_channel // 16, self.inplanes, 3, 1, 1, bias=False
        )
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.layer1 = self._make_layer(block, ch[0], layers[0])
        self.conv1 = nn.Conv2d(ch[0], ch[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch[0])
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.layer2 = self._make_layer(block, ch[1], layers[1])
        self.conv2 = nn.Conv2d(ch[1], ch[1], 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch[1])
        self.maxpool3 = nn.MaxPool2d(2, (2, 1), (0, 1))
        self.layer3 = self._make_layer(block, ch[2], layers[2])
        self.conv3 = nn.Conv2d(ch[2], ch[2], 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ch[2])
        self.layer4 = self._make_layer(block, ch[3], layers[3])
        self.conv4_1 = nn.Conv2d(ch[3], ch[3], 2, (2, 1), (0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(ch[3])
        self.conv4_2 = nn.Conv2d(ch[3], ch[3], 2, 1, 0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(ch[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion, 1, stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn0_1(self.conv0_1(x)))
        x = self.relu(self.bn0_2(self.conv0_2(x)))
        x = self.maxpool1(x)
        x = self.relu(self.bn1(self.conv1(self.layer1(x))))
        x = self.maxpool2(x)
        x = self.relu(self.bn2(self.conv2(self.layer2(x))))
        x = self.maxpool3(x)
        x = self.relu(self.bn3(self.conv3(self.layer3(x))))
        x = self.layer4(x)
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        return x


class ResNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=512):
        super().__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, x):
        return self.ConvNet(x)


# ---------------------------------------------------------------------------
# Sequence modeling
# ---------------------------------------------------------------------------


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(x)
        return self.linear(recurrent)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class AttentionHTR(nn.Module):
    def __init__(self):
        super().__init__()
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=NUM_FIDUCIAL,
            I_size=(IMG_H, IMG_W),
            I_r_size=(IMG_H, IMG_W),
            I_channel_num=INPUT_CHANNEL,
        )
        self.FeatureExtraction = ResNet_FeatureExtractor(INPUT_CHANNEL, OUTPUT_CHANNEL)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(OUTPUT_CHANNEL, HIDDEN_SIZE, HIDDEN_SIZE),
            BidirectionalLSTM(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
        )

    def forward(self, x):
        x = self.Transformation(x)
        x = self.FeatureExtraction(x)
        x = self.AdaptiveAvgPool(x.permute(0, 3, 1, 2)).squeeze(3)
        return self.SequenceModeling(x)  # (B, T, HIDDEN_SIZE)


def load_model(model_path=None, device="cpu"):
    model = AttentionHTR()
    state = torch.load(model_path, map_location=device)
    # state_dict may be wrapped in DataParallel — strip the "module." prefix if present
    if any(k.startswith("module.") for k in state):
        state = {k[len("module.") :]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(device)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def get_features(images, model, device="cpu"):
    """
    Extract AttentionHTR contextual features for a list of PIL images.

    Args:
        images: List of PIL images.
        model:  Pre-loaded AttentionHTR model.
        device: Torch device string.

    Returns:
        Tensor of shape (B, T, D) — sequential contextual features.
    """
    batch = torch.stack([attentionhtr_transforms(img) for img in images]).to(device)
    with torch.no_grad():
        return model(batch)  # (B, T, D)
        


def pool_features(features):
    """
    Mean-pool sequential features over time positions.

    Args:
        features: Tensor of shape (B, T, D).

    Returns:
        Tensor of shape (B, D).
    """
    return features.mean(dim=1)

def _extract_from_paths(
    paths: list[str],
    model: AttentionHTR,
    device: str,
    batch_size: int,
) -> np.ndarray:
    all_features: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        feats = get_features(images, model, device=device)
        all_features.append(feats.cpu().numpy())
    return np.concatenate(all_features, axis=0)


