
from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as nd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from nibabel.affines import apply_affine


# ============================================================
# CONFIG
# ============================================================
DATASET_NAME = "CT-Based Cochlear Implant Planning Dataset"
REPORT_DIR = os.getenv("REPORT_DIR", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

LANDMARK_NAMES = ["RW", "C", "A", "OW", "V"]
SUMMARY_LANDMARKS = ["RW", "C", "OW"]
REPORT_LANDMARKS = ["RW", "C", "OW"]

TARGET_SHAPE = (96, 128, 128)   # (D, H, W) for landmark model
SEG_INPUT_SIZE = (128, 128)     # for 2.5D segmentation model

RISK_FEATURES = [
    "dist_C_RW",
    "dist_C_OW",
    "dist_RW_OW",
    "landmark_span",
    "compactness",
    "num_components",
]

TRAIN_SUBJECTS = [f"F{i:02d}" for i in range(1, 11)] + [f"T{i:02d}" for i in range(1, 6)]
VAL_SUBJECTS   = [f"F{i:02d}" for i in range(11, 14)]
TEST_SUBJECTS  = [f"F{i:02d}" for i in range(14, 16)] + [f"T{i:02d}" for i in range(6, 9)]


# ============================================================
# MODELS
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.pool = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv2 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x3 = self.down2(x2)
        x4 = self.pool(x3)
        x5 = self.down3(x4)

        x = self.up1(x5)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)

        return self.out(x)


class DoubleConv3D(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 5, base: int = 16):
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3D(base * 2, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base * 2, base)

        self.out = nn.Conv3d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.pool1(x1)

        x3 = self.enc2(x2)
        x4 = self.pool2(x3)

        x5 = self.bottleneck(x4)

        x = self.up2(x5)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        return self.out(x)


# ============================================================
# HELPERS
# ============================================================
def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    return (vol - vol.mean()) / (vol.std() + 1e-8)


def resize_volume_3d(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    vol: numpy array with shape (D, H, W)
    returns torch tensor with shape (1, D, H, W)
    """
    t = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)
    return t.squeeze(0)


def resize_heatmap_3d(hm: np.ndarray, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    t = torch.tensor(hm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0)


def decode_heatmap_argmax(hm: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    hm: torch tensor or numpy array with shape (D, H, W)
    returns (d, h, w)
    """
    if torch.is_tensor(hm):
        idx = torch.argmax(hm).item()
        D, H, W = hm.shape
    else:
        idx = int(np.argmax(hm))
        D, H, W = hm.shape

    d = idx // (H * W)
    rem = idx % (H * W)
    h = rem // W
    w = rem % W
    return np.array([d, h, w], dtype=np.float32)


def net_to_voxel_xyz(v_net: np.ndarray) -> np.ndarray:
    # (D, H, W) -> (X, Y, Z)
    d, h, w = v_net
    return np.array([h, w, d], dtype=np.float32)


def voxel_xyz_to_net(v_xyz: np.ndarray) -> np.ndarray:
    # CT array is (X, Y, Z)
    # network uses (D, H, W) = (Z, X, Y)
    x, y, z = v_xyz
    return np.array([z, x, y], dtype=np.float32)


def world_to_voxel(world_coord: Sequence[float], affine: np.ndarray) -> np.ndarray:
    inv_affine = np.linalg.inv(affine)
    return apply_affine(inv_affine, world_coord)


def load_desc(desc_path: str) -> Dict[str, Any]:
    with open(desc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    landmarks = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in data["LANDMARKS"].items()
    }

    return {
        "id": data.get("ID", ""),
        "side": data.get("SIDE", ""),
        "landmarks": landmarks,
        "origin": np.asarray(data["COORD"]["ORIGIN"], dtype=np.float32),
        "xaxis": np.asarray(data["COORD"]["XAXIS"], dtype=np.float32),
        "yaxis": np.asarray(data["COORD"]["YAXIS"], dtype=np.float32),
        "zaxis": np.asarray(data["COORD"]["ZAXIS"], dtype=np.float32),
    }


def find_first_file(folder: str, includes: Optional[Sequence[str]] = None, suffixes: Optional[Sequence[str]] = None) -> Optional[str]:
    if not os.path.isdir(folder):
        return None

    includes = includes or []
    suffixes = suffixes or []

    for f in os.listdir(folder):
        low = f.lower()
        ok_inc = all(s.lower() in low for s in includes)
        ok_suf = any(low.endswith(s.lower()) for s in suffixes) if suffixes else True
        if ok_inc and ok_suf:
            return os.path.join(folder, f)
    return None


def find_mask_file(subject_path: str) -> Optional[str]:
    candidates = [
        os.path.join(subject_path, "UCT"),
        os.path.join(subject_path, "CT"),
        subject_path,
    ]

    for folder in candidates:
        if not os.path.isdir(folder):
            continue

        for key in ("labels", "mask", "seg"):
            hit = find_first_file(folder, includes=[key], suffixes=[".nii", ".nii.gz"])
            if hit:
                return hit

    for root, _, files in os.walk(subject_path):
        for f in files:
            low = f.lower()
            if ("labels" in low or "mask" in low or "seg" in low) and (
                low.endswith(".nii") or low.endswith(".nii.gz")
            ):
                return os.path.join(root, f)

    return None


def build_dataset(root_path: str, subjects: Sequence[str]) -> List[Dict[str, str]]:
    data_list = []

    for subject in subjects:
        subject_path = os.path.join(root_path, subject)
        ct_path = os.path.join(subject_path, "CT")

        ct_file = None
        desc_file = None
        mask_file = None

        if os.path.isdir(ct_path):
            for f in os.listdir(ct_path):
                if "CT_RAW" in f and (f.endswith(".nii") or f.endswith(".nii.gz")):
                    ct_file = os.path.join(ct_path, f)
                    break
            mask_file = find_mask_file(subject_path)

        if os.path.isdir(subject_path):
            for f in os.listdir(subject_path):
                if f.endswith("_DESC.json"):
                    desc_file = os.path.join(subject_path, f)
                    break

        if ct_file and desc_file:
            item = {
                "subject": subject,
                "ct": ct_file,
                "desc": desc_file,
            }
            if mask_file:
                item["mask"] = mask_file
            data_list.append(item)

    return data_list


def compute_binary_dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = np.sum(a * b)
    return float((2.0 * inter + eps) / (np.sum(a) + np.sum(b) + eps))


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_bin = (mask > 0).astype(np.uint8)
    labeled, num = nd.label(mask_bin)

    if num == 0:
        return mask_bin

    sizes = nd.sum(mask_bin, labeled, range(1, num + 1))
    largest_label = 1 + int(np.argmax(sizes))
    return (labeled == largest_label).astype(np.uint8)


def extract_landmark_concepts_from_predictions(pred_world: Dict[str, np.ndarray], pred_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    def dist(a: str, b: str) -> float:
        return float(np.linalg.norm(pred_world[a] - pred_world[b]))

    feats: Dict[str, float] = {
        "dist_C_RW": dist("C", "RW"),
        "dist_C_OW": dist("C", "OW"),
        "dist_RW_OW": dist("RW", "OW"),
    }

    feats["landmark_span"] = max(
        feats["dist_C_RW"],
        feats["dist_C_OW"],
        feats["dist_RW_OW"],
    )

    if pred_mask is not None:
        coords = np.argwhere(pred_mask > 0)

        if coords.size == 0:
            feats.update({
                "compactness": 0.0,
                "num_components": 0,
                "volume_voxels": 0,
                "bbox_x": 0,
                "bbox_y": 0,
                "bbox_z": 0,
            })
        else:
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0) + 1
            bbox_dims = maxs - mins
            bbox_volume = int(np.prod(bbox_dims))
            volume_voxels = int(coords.shape[0])
            compactness = float(volume_voxels / (bbox_volume + 1e-8))
            _, num_components = nd.label(pred_mask > 0)

            feats.update({
                "compactness": compactness,
                "num_components": int(num_components),
                "volume_voxels": volume_voxels,
                "bbox_x": int(bbox_dims[0]),
                "bbox_y": int(bbox_dims[1]),
                "bbox_z": int(bbox_dims[2]),
            })
    else:
        feats.update({
            "compactness": np.nan,
            "num_components": np.nan,
            "volume_voxels": np.nan,
            "bbox_x": np.nan,
            "bbox_y": np.nan,
            "bbox_z": np.nan,
        })

    return feats


def fit_risk_profile(train_df: pd.DataFrame) -> Dict[str, Any]:
    mu = train_df[RISK_FEATURES].mean()
    sigma = train_df[RISK_FEATURES].std(ddof=0).replace(0, 1.0)

    z = (train_df[RISK_FEATURES] - mu) / sigma

    risk_score = (
        0.35 * z["dist_C_RW"] +
        0.35 * z["dist_C_OW"] +
        0.15 * z["dist_RW_OW"] +
        0.10 * z["num_components"] -
        0.10 * z["compactness"] +
        0.15 * z["landmark_span"]
    )

    q_safe, q_mod = np.quantile(risk_score, [0.33, 0.66])

    return {
        "mu": mu,
        "sigma": sigma,
        "q_safe": float(q_safe),
        "q_mod": float(q_mod),
    }


def predict_risk_from_features(features: Dict[str, float], risk_profile: Dict[str, Any]) -> Tuple[float, str, List[Tuple[str, float]]]:
    x = pd.Series(features)
    z = (x[RISK_FEATURES] - risk_profile["mu"]) / risk_profile["sigma"]

    score = (
        0.35 * z["dist_C_RW"] +
        0.35 * z["dist_C_OW"] +
        0.15 * z["dist_RW_OW"] +
        0.10 * z["num_components"] -
        0.10 * z["compactness"] +
        0.15 * z["landmark_span"]
    )

    if score <= risk_profile["q_safe"]:
        label = "SAFE"
    elif score <= risk_profile["q_mod"]:
        label = "MODERATE"
    else:
        label = "HIGH RISK"

    contributions = {
        "dist_C_RW": float(0.35 * z["dist_C_RW"]),
        "dist_C_OW": float(0.35 * z["dist_C_OW"]),
        "dist_RW_OW": float(0.15 * z["dist_RW_OW"]),
        "num_components": float(0.10 * z["num_components"]),
        "compactness": float(-0.10 * z["compactness"]),
        "landmark_span": float(0.15 * z["landmark_span"]),
    }

    top_reasons = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    return float(score), label, top_reasons


def predict_subject_mask(segmentation_model: nn.Module, ct_path: str, device: torch.device | str, input_size: Tuple[int, int] = SEG_INPUT_SIZE) -> np.ndarray:
    ct_img = nib.load(ct_path)
    ct = ct_img.get_fdata()
    X, Y, Z = ct.shape

    pred_vol = np.zeros((X, Y, Z), dtype=np.uint8)

    segmentation_model.eval()
    with torch.no_grad():
        for i in range(Z):
            i0 = max(i - 1, 0)
            i1 = i
            i2 = min(i + 1, Z - 1)

            slices = []
            for j in (i0, i1, i2):
                s = ct[:, :, j].astype(np.float32)
                s = (s - s.mean()) / (s.std() + 1e-8)

                t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                t = F.interpolate(t, size=input_size, mode="bilinear", align_corners=False)
                slices.append(t.squeeze(0).squeeze(0))

            x = torch.stack(slices, dim=0).unsqueeze(0).to(device)

            logits = segmentation_model(x)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()

            pred_2d = F.interpolate(pred, size=(X, Y), mode="nearest").squeeze().cpu().numpy().astype(np.uint8)
            pred_vol[:, :, i] = pred_2d

    return keep_largest_component(pred_vol)


def predict_landmarks_from_heatmaps(
    landmark_model: nn.Module,
    ct_path: str,
    device: torch.device | str,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], torch.Tensor]:
    ct_img = nib.load(ct_path)
    ct = ct_img.get_fdata().astype(np.float32)
    affine = ct_img.affine

    ct_net = np.transpose(ct, (2, 0, 1))
    ct_net = normalize_volume(ct_net)

    ct_tensor = torch.tensor(ct_net, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ct_tensor = F.interpolate(ct_tensor, size=target_shape, mode="trilinear", align_corners=False).to(device)

    landmark_model.eval()
    with torch.no_grad():
        pred_hm = landmark_model(ct_tensor)[0].cpu()

    pred_vox: Dict[str, np.ndarray] = {}
    pred_world: Dict[str, np.ndarray] = {}

    orig_net_shape = np.array([ct.shape[2], ct.shape[0], ct.shape[1]], dtype=np.float32)
    scale_back = (orig_net_shape - 1.0) / (np.array(target_shape, dtype=np.float32) - 1.0 + 1e-8)

    for i, name in enumerate(LANDMARK_NAMES):
        hm = pred_hm[i]
        coord_net = decode_heatmap_argmax(hm)
        coord_net_orig = coord_net * scale_back
        coord_xyz = net_to_voxel_xyz(coord_net_orig)

        pred_vox[name] = coord_xyz
        pred_world[name] = apply_affine(affine, coord_xyz)

    return pred_world, pred_vox, pred_hm


def predict_landmarks(model: nn.Module, ct_path: str, desc_path: Optional[str] = None, target_shape: Tuple[int, int, int] = TARGET_SHAPE):
    ct_img = nib.load(ct_path)
    ct = ct_img.get_fdata().astype(np.float32)
    affine = ct_img.affine

    ct_net = np.transpose(ct, (2, 0, 1))
    ct_net = normalize_volume(ct_net)
    ct_tensor = resize_volume_3d(ct_net, target_shape).unsqueeze(0)  # (1,1,D,H,W)

    model.eval()
    with torch.no_grad():
        pred_hm = model(ct_tensor)[0].cpu()  # (L, D, H, W)

    pred_world: Dict[str, np.ndarray] = {}
    pred_vox: Dict[str, np.ndarray] = {}

    orig_net_shape = np.array([ct.shape[2], ct.shape[0], ct.shape[1]], dtype=np.float32)
    scale_back = (orig_net_shape - 1.0) / (np.array(target_shape, dtype=np.float32) - 1.0 + 1e-8)

    for i, name in enumerate(LANDMARK_NAMES):
        hm = pred_hm[i]
        coord_net = decode_heatmap_argmax(hm)
        coord_net_orig = coord_net * scale_back
        coord_xyz = net_to_voxel_xyz(coord_net_orig)
        pred_vox[name] = coord_xyz
        pred_world[name] = apply_affine(affine, coord_xyz)

    gt_world = None
    if desc_path is not None:
        desc = load_desc(desc_path)
        gt_world = desc["landmarks"]

    return pred_world, pred_vox, gt_world, pred_hm


def build_concept_table_from_heatmaps(data_list, segmentation_model: nn.Module, landmark_model: nn.Module, device: torch.device | str) -> pd.DataFrame:
    rows = []

    for item in data_list:
        ct_path = item["ct"]
        subject = item["subject"]

        pred_mask = predict_subject_mask(segmentation_model, ct_path, device)
        pred_world, _, _ = predict_landmarks_from_heatmaps(landmark_model, ct_path, device)
        feats = extract_landmark_concepts_from_predictions(pred_world, pred_mask=pred_mask)

        row = {"subject": subject}
        row.update(feats)
        rows.append(row)

    return pd.DataFrame(rows)


def fit_risk_profile_from_training(train_data, segmentation_model: nn.Module, landmark_model: nn.Module, device: torch.device | str):
    train_df = build_concept_table_from_heatmaps(
        train_data,
        segmentation_model=segmentation_model,
        landmark_model=landmark_model,
        device=device,
    )
    risk_profile = fit_risk_profile(train_df)
    return risk_profile, train_df


def surgical_recommendation(risk_label: str, features: Dict[str, float]) -> str:
    if risk_label == "SAFE":
        return (
            "FAVORABLE ANATOMY: the bony labyrinth appears compact and continuous. "
            "Review the original CT, but this case does not show major geometric concerns in the current model."
        )
    elif risk_label == "MODERATE":
        return (
            "INTERMEDIATE ANATOMY: geometry is somewhat less typical. "
            "Review landmark spacing, segmentation quality, and the original CT before final planning."
        )
    else:
        return (
            "CHALLENGING ANATOMY: geometry is more atypical and should be reviewed carefully. "
            "Use the original CT and full clinical context before any surgical decision."
        )


def top_feature_reasons(reasons: Sequence[Tuple[str, float]], top_k: int = 3):
    return sorted(reasons, key=lambda kv: abs(kv[1]), reverse=True)[:top_k]


# ============================================================
# VISUALIZATION
# ============================================================
def _add_mask_surface(ax, mask, affine, color="red", alpha=0.30, title=""):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    try:
        from skimage.measure import marching_cubes
    except Exception:
        from skimage.measure import marching_cubes_lewiner as marching_cubes

    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        ax.text(0.5, 0.5, 0.5, "Empty mask", ha="center")
        ax.axis("off")
        return

    verts, faces, _, _ = marching_cubes(mask, level=0.5)
    verts_world = apply_affine(affine, verts)
    mesh = Poly3DCollection(verts_world[faces], alpha=alpha)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.auto_scale_xyz(
        verts_world[:, 0], verts_world[:, 1], verts_world[:, 2]
    )


def visualize_2d(ct_path: str, gt_mask_path: Optional[str] = None, pred_mask: Optional[np.ndarray] = None):
    ct = nib.load(ct_path).get_fdata()
    z = ct.shape[2] // 2

    gt_slice = None
    pred_slice = None

    if gt_mask_path is not None:
        gt = nib.load(gt_mask_path).get_fdata()
        gt = (gt > 0).astype(np.uint8)
        if gt.shape != ct.shape:
            raise ValueError(
                f"GT shape {gt.shape} does not match CT shape {ct.shape}. Check alignment / resampling."
            )
        z = int(np.argmax([gt[:, :, i].sum() for i in range(gt.shape[2])]))
        gt_slice = gt[:, :, z]

    if pred_mask is not None:
        pred_mask = (pred_mask > 0).astype(np.uint8)
        if pred_mask.shape != ct.shape:
            raise ValueError(
                f"Predicted mask shape {pred_mask.shape} does not match CT shape {ct.shape}. Prediction pipeline is producing a mismatched volume."
            )
        z_pred = max(0, min(z, pred_mask.shape[2] - 1))
        pred_slice = pred_mask[:, :, z_pred]

    z_ct = max(0, min(z, ct.shape[2] - 1))
    ncols = 1 + int(gt_slice is not None) + int(pred_slice is not None)
    fig = plt.figure(figsize=(22, 7), dpi=180)

    ax1 = fig.add_subplot(1, ncols, 1)
    ax1.imshow(ct[:, :, z_ct], cmap="gray")
    ax1.set_title(f"Main CT slice {z_ct}", fontsize=14)
    ax1.axis("off")

    col = 2
    if gt_slice is not None:
        ax2 = fig.add_subplot(1, ncols, col)
        ax2.imshow(ct[:, :, z_ct], cmap="gray")
        ax2.imshow(gt_slice, alpha=0.45, cmap="Greens")
        ax2.set_title("Ground truth mask overlay", fontsize=14)
        ax2.axis("off")
        col += 1

    if pred_slice is not None:
        ax3 = fig.add_subplot(1, ncols, col)
        ax3.imshow(ct[:, :, z_ct], cmap="gray")
        ax3.imshow(pred_slice, alpha=0.45, cmap="Reds")
        ax3.set_title("Predicted mask overlay", fontsize=14)
        ax3.axis("off")

    fig.suptitle(f"{DATASET_NAME}", fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def visualize_3d_masks(ct_path: str, gt_mask_path: Optional[str] = None, pred_mask: Optional[np.ndarray] = None):
    ct_img = nib.load(ct_path)
    ct = ct_img.get_fdata()
    affine = ct_img.affine

    fig = plt.figure(figsize=(18, 6), dpi=180)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    if gt_mask_path is not None:
        gt = nib.load(gt_mask_path).get_fdata()
        gt = (gt > 0).astype(np.uint8)

        if gt.shape != ct.shape:
            raise ValueError(
                f"GT shape {gt.shape} does not match CT shape {ct.shape}. Check alignment / resampling."
            )

        _add_mask_surface(ax1, gt, affine, color="green", alpha=0.35, title="GT Mask")
    else:
        ax1.text(0.5, 0.5, 0.5, "No GT Mask", ha="center")
        ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    if pred_mask is not None:
        pred_mask = (pred_mask > 0).astype(np.uint8)

        if pred_mask.shape != ct.shape:
            raise ValueError(
                f"Predicted mask shape {pred_mask.shape} does not match CT shape {ct.shape}. Prediction pipeline is producing a mismatched volume."
            )

        _add_mask_surface(ax2, pred_mask, affine, color="red", alpha=0.35, title="Pred Mask")
    else:
        ax2.text(0.5, 0.5, 0.5, "No Prediction", ha="center")
        ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    if gt_mask_path is not None and pred_mask is not None:
        gt = nib.load(gt_mask_path).get_fdata()
        gt = (gt > 0).astype(np.uint8)
        pred = (pred_mask > 0).astype(np.uint8)

        if gt.shape != ct.shape:
            raise ValueError(
                f"GT shape {gt.shape} does not match CT shape {ct.shape}. Check alignment / resampling."
            )
        if pred.shape != ct.shape:
            raise ValueError(
                f"Predicted mask shape {pred.shape} does not match CT shape {ct.shape}. Prediction pipeline is producing a mismatched volume."
            )

        _add_mask_surface(ax3, gt, affine, color="green", alpha=0.25, title="")
        _add_mask_surface(ax3, pred, affine, color="red", alpha=0.25, title="")
        dice_score = compute_binary_dice(gt, pred)
        ax3.set_title(f"GT vs Pred Overlay\nDice = {dice_score:.3f}", fontsize=13)
    else:
        ax3.text(0.5, 0.5, 0.5, "No overlap view", ha="center")
        ax3.axis("off")

    fig.suptitle("Segmentation Visualization", fontsize=16)
    plt.tight_layout()
    return fig



def visualize_3d_prediction_only(ct_path: str, pred_mask: Optional[np.ndarray] = None):
    ct_img = nib.load(ct_path)
    ct = ct_img.get_fdata()
    affine = ct_img.affine

    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    if pred_mask is not None:
        pred_mask = (pred_mask > 0).astype(np.uint8)

        if pred_mask.shape != ct.shape:
            raise ValueError(
                f"Predicted mask shape {pred_mask.shape} does not match CT shape {ct.shape}. Prediction pipeline is producing a mismatched volume."
            )

        _add_mask_surface(ax, pred_mask, affine, color="red", alpha=0.35, title="Predicted Segmentation (3D)")
        ax.set_title("Predicted Segmentation (3D)", fontsize=14, weight="bold")
    else:
        ax.text(0.5, 0.5, 0.5, "No Prediction", ha="center")
        ax.axis("off")

    fig.suptitle("Prediction-only 3D segmentation", fontsize=16)
    plt.tight_layout()
    return fig

def plot_risk_contributions_big(reasons, risk_label, risk_score):
    reasons = sorted(reasons, key=lambda kv: abs(kv[1]), reverse=True)
    names = [r[0] for r in reasons]
    vals = [r[1] for r in reasons]

    fig = plt.figure(figsize=(14, 6), dpi=180)
    ax = fig.add_subplot(111)
    bars = ax.barh(names, vals)
    ax.axvline(0, linewidth=1)
    ax.set_title(f"Top risk contributions | Label: {risk_label} | Score: {risk_score:.4f}", fontsize=14)
    ax.set_xlabel("Contribution to risk score")
    ax.invert_yaxis()

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + (0.01 if val >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
        )

    plt.tight_layout()
    return fig


def add_simple_text_page(pdf, title, lines, subtitle=None):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.94
    fig.text(0.05, y, title, fontsize=18, fontweight="bold")
    y -= 0.05

    if subtitle:
        fig.text(0.05, y, subtitle, fontsize=10, style="italic")
        y -= 0.04

    for line in lines:
        if line == "":
            y -= 0.02
            continue

        wrapped = textwrap.wrap(str(line), width=100)
        for w in wrapped:
            fig.text(0.05, y, w, fontsize=10)
            y -= 0.03
            if y < 0.06:
                break
        if y < 0.06:
            break

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PDF REPORT GENERATOR
# ============================================================
def generate_case_report(
    sample: Dict[str, Any],
    result: Dict[str, Any],
    out_pdf_path: Optional[str] = None,
    dataset_name: str = DATASET_NAME,
    show_in_notebook: bool = False,
    include_mask_visualization: bool = True,
    distance_image_path: Optional[str] = "distance_image.jpeg",
) -> str:
    """
    Generate a PDF report for one case.
    """

    PROJECT_NAME = "CT-Based Cochlear Implant Surgical Planning Research Prototype"

    def fmt(x, decimals=4):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "N/A"
            return f"{float(x):.{decimals}f}"
        except Exception:
            return "N/A"

    def safe_load_mask(path):
        if path is None or (not os.path.exists(path)):
            return None
        try:
            return nib.load(path).get_fdata()
        except Exception:
            return None

    def get_best_slice(ct, gt_mask=None, pred_mask=None):
        if gt_mask is not None:
            return int(np.argmax([gt_mask[:, :, i].sum() for i in range(gt_mask.shape[2])]))
        if pred_mask is not None:
            return int(np.argmax([pred_mask[:, :, i].sum() for i in range(pred_mask.shape[2])]))
        return ct.shape[2] // 2

    def add_block(fig, x, y, text, *, width=96, line_step=0.031, fontsize=11.0,
                  color="#111827", weight=None, style=None, max_y=0.06):
        lines = textwrap.wrap(str(text), width=width)
        for line in lines:
            fig.text(x, y, line, fontsize=fontsize, color=color, weight=weight, style=style)
            y -= line_step
            if y < max_y:
                break
        return y

    def add_lines(fig, x, y, lines, *, fontsize=11.0, color="#111827", line_step=0.030):
        for line in lines:
            fig.text(x, y, line, fontsize=fontsize, color=color)
            y -= line_step
        return y

    def draw_header(fig, title, subtitle=None):
        fig.text(0.05, 0.94, PROJECT_NAME, fontsize=23, weight="bold", color="#1D4ED8")
        fig.text(0.05, 0.895, title, fontsize=18, weight="bold", color="#0F172A")
        if subtitle:
            fig.text(0.05, 0.86, subtitle, fontsize=10.5, color="#475569", style="italic")

    def compute_thresholds(risk_profile):
        train_scores = risk_profile.get("train_scores", None)
        alpha_safe = float(risk_profile.get("alpha_safe", 0.33))
        alpha_mod = float(risk_profile.get("alpha_mod", 0.66))

        if train_scores is not None:
            s = np.asarray(train_scores, dtype=float).ravel()
            s = s[~np.isnan(s)]
            if len(s) == 0:
                return {
                    "q_safe": float(risk_profile["q_safe"]),
                    "q_mod": float(risk_profile["q_mod"]),
                    "alpha_safe": alpha_safe,
                    "alpha_mod": alpha_mod,
                    "n": 0,
                    "k_safe": None,
                    "k_mod": None,
                    "sorted_scores": None,
                    "used_train_scores": False,
                }

            s_sorted = np.sort(s)
            n = len(s_sorted)

            k_safe = int(np.ceil(alpha_safe * n)) - 1
            k_mod = int(np.ceil(alpha_mod * n)) - 1

            k_safe = max(0, min(k_safe, n - 1))
            k_mod = max(0, min(k_mod, n - 1))

            q_safe = float(s_sorted[k_safe])
            q_mod = float(s_sorted[k_mod])

            return {
                "q_safe": q_safe,
                "q_mod": q_mod,
                "alpha_safe": alpha_safe,
                "alpha_mod": alpha_mod,
                "n": n,
                "k_safe": k_safe,
                "k_mod": k_mod,
                "sorted_scores": s_sorted,
                "used_train_scores": True,
            }

        return {
            "q_safe": float(risk_profile["q_safe"]),
            "q_mod": float(risk_profile["q_mod"]),
            "alpha_safe": alpha_safe,
            "alpha_mod": alpha_mod,
            "n": None,
            "k_safe": None,
            "k_mod": None,
            "sorted_scores": None,
            "used_train_scores": False,
        }

    def display_anatomy_label(risk_label: str) -> str:
        return {
            "SAFE": "FAVORABLE ANATOMY",
            "MODERATE": "INTERMEDIATE ANATOMY",
            "HIGH RISK": "CHALLENGING ANATOMY",
        }.get(risk_label, "UNKNOWN")

    def get_result_explanation(risk_label):
        if risk_label == "SAFE":
            return (
                "The predicted anatomy demonstrates a compact and continuous morphology with a single connected structure and no evidence of major fragmentation. "
                "The key anatomical distances between the Round Window (RW), Cochlear Center (C), and Oval Window (OW) are relatively small. "
                "The overall landmark span is also limited, which suggests a compact inner-ear configuration. "
                "From a planning perspective, this case appears closer to the favorable end of the anatomy spectrum."
            )
        elif risk_label == "MODERATE":
            return (
                "The predicted anatomy shows preserved structural continuity, but one or more geometric measurements are larger than average. "
                "The landmark separation and overall anatomical span suggest moderate anatomical variability. "
                "Although no major structural irregularities are detected, the morphology is less compact than the most favorable cases. "
                "This case therefore falls into an intermediate planning category."
            )
        else:
            return (
                "The predicted anatomy demonstrates increased geometric complexity characterized by larger landmark separations, a broader anatomical span, and/or reduced structural compactness. "
                "These factors collectively elevate the composite planning score relative to the training cohort. "
                "This classification should be interpreted as an anatomy-complexity alert rather than a prediction of surgical outcome."
            )

    def add_summary_page(pdf, subject, specimen_id, specimen_side, risk_label, risk_score, dice_score, top_reasons, features):
        label_colors = {
            "SAFE": "#166534",
            "MODERATE": "#B45309",
            "HIGH RISK": "#991B1B",
        }
        box_color = label_colors.get(risk_label, "#334155")
        display_label = display_anatomy_label(risk_label)

        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header(fig, "Planning Summary")

        y = 0.81
        fig.text(0.05, y, f"Subject: {subject}", fontsize=13, weight="bold", color="#111827")
        y -= 0.040
        if specimen_id is not None:
            fig.text(0.05, y, f"Specimen ID: {specimen_id}", fontsize=12, color="#334155")
            y -= 0.034
        if specimen_side is not None:
            fig.text(0.05, y, f"Side: {specimen_side}", fontsize=12, color="#334155")
            y -= 0.034

        fig.text(
            0.05, 0.64,
            f"Anatomical complexity: {display_label}",
            fontsize=18,
            weight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=box_color, edgecolor=box_color),
        )
        fig.text(0.05, 0.585, f"Planning score: {risk_score:.4f}", fontsize=14, weight="bold", color="#111827")
        if dice_score is not None:
            fig.text(0.05, 0.545, f"Segmentation Dice: {dice_score:.4f}", fontsize=13, weight="bold", color="#0F766E")

        fig.text(0.05, 0.500, "Interpretation", fontsize=13, weight="bold", color="#0F172A")
        interp = get_result_explanation(risk_label)
        y = add_block(fig, 0.05, 0.472, interp, width=120, line_step=0.030, fontsize=11.1, color="#111827")
        y -= 0.005

        fig.text(0.05, y, "Key contributors", fontsize=13, weight="bold", color="#0F172A")
        y -= 0.032
        for name, val in top_reasons:
            direction = "lowered" if val < 0 else "raised"
            color = "#166534" if val < 0 else "#991B1B"
            fig.text(0.07, y, f"• {name}", fontsize=11.4, weight="bold", color="#111827")
            fig.text(0.24, y, f"{direction} the score by {abs(val):.3f}", fontsize=11.4, color=color)
            y -= 0.034

        recommendation = surgical_recommendation(risk_label, features)
        fig.text(0.05, 0.125, "Planning note", fontsize=13, weight="bold", color="#0F172A")
        add_block(fig, 0.05, 0.097, recommendation, width=170, line_step=0.030, fontsize=11.0, color="#111827")

        fig.text(
            0.05, 0.045,
            "Important note: this is a proxy anatomy-complexity summary derived from predicted imaging features. It supports planning and research discussion, but it does not replace review of the original CT and the full clinical context.",
            fontsize=11.5,
            color="#6B7280",
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def add_risk_logic_page(pdf, risk_profile, risk_label, risk_score, features):
        mu = risk_profile["mu"]
        sigma = risk_profile["sigma"]

        weights = {
            "dist_C_RW": 0.35,
            "dist_C_OW": 0.35,
            "dist_RW_OW": 0.15,
            "num_components": 0.10,
            "compactness": -0.10,
            "landmark_span": 0.15,
        }

        thr = compute_thresholds(risk_profile)
        q_safe = thr["q_safe"]
        q_mod = thr["q_mod"]
        alpha_safe = thr["alpha_safe"]
        alpha_mod = thr["alpha_mod"]
        n = thr["n"]
        k_safe = thr["k_safe"]
        k_mod = thr["k_mod"]
        used_train = thr["used_train_scores"]
        sorted_scores = thr["sorted_scores"]

        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header(fig, "Anatomical Complexity Model and Decision Rule")

        fig.text(0.05, 0.83, "Score equation", fontsize=13, weight="bold", color="#111827")
        equation = (
            "score = 0.35·z(dist_C_RW) + 0.35·z(dist_C_OW) + 0.15·z(dist_RW_OW) "
            "+ 0.10·z(num_components) - 0.10·z(compactness) + 0.15·z(landmark_span)"
        )
        fig.text(
            0.05, 0.785, equation,
            fontsize=11.0, family="monospace", color="#111827",
            bbox=dict(boxstyle="round,pad=0.40", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
        )

        fig.text(0.05, 0.70, "How the label is assigned", fontsize=13, weight="bold", color="#111827")
        fig.text(0.07, 0.665, f"• score ≤ q_safe ({q_safe:.4f})  → {display_anatomy_label('SAFE')}", fontsize=11.4, color="#166534")
        fig.text(0.07, 0.632, f"• q_safe < score ≤ q_mod ({q_mod:.4f})  → {display_anatomy_label('MODERATE')}", fontsize=11.4, color="#B45309")
        fig.text(0.07, 0.599, f"• score > q_mod  → {display_anatomy_label('HIGH RISK')}", fontsize=11.4, color="#991B1B")

        fig.text(0.05, 0.54, "Threshold calculation", fontsize=13, weight="bold", color="#111827")

        if used_train and sorted_scores is not None:
            threshold_lines = [
                f"Training scores: S_train = {{s_1, s_2, ..., s_n}} ,  n = {n}",
                f"Sort them: s_(1) ≤ s_(2) ≤ ... ≤ s_(n)",
                f"k_safe = ceil(α_safe · n) - 1 = ceil({alpha_safe:.4f} · {n}) - 1 = {k_safe}",
                f"q_safe = s_(k_safe) = s_({k_safe + 1}) = {q_safe:.4f}",
                f"k_mod  = ceil(α_mod · n) - 1 = ceil({alpha_mod:.4f} · {n}) - 1 = {k_mod}",
                f"q_mod  = s_(k_mod)  = s_({k_mod + 1}) = {q_mod:.4f}",
                "Decision rule: compare the case score to these two cut points.",
            ]
        else:
            threshold_lines = [
                f"q_safe = {q_safe:.4f}",
                f"q_mod  = {q_mod:.4f}",
                "Thresholds were taken from the training-derived risk profile.",
            ]

        y = 0.505
        y = add_lines(fig, 0.05, y, threshold_lines, fontsize=10.6, color="#111827", line_step=0.030)

        fig.text(0.05, 0.315, "Feature contribution breakdown", fontsize=13, weight="bold", color="#111827")
        fig.text(
            0.05, 0.288,
            "Feature            value        z-score      weight     contribution",
            fontsize=10.6, family="monospace", color="#111827",
        )

        feature_order = [
            "dist_C_RW",
            "dist_C_OW",
            "dist_RW_OW",
            "num_components",
            "compactness",
            "landmark_span",
        ]

        y = 0.258
        total_check = 0.0
        for name in feature_order:
            x = float(features.get(name, np.nan))
            m = float(mu[name])
            s = float(sigma[name])
            z = (x - m) / s if s != 0 else 0.0
            w = weights[name]
            contrib = w * z
            total_check += contrib
            fig.text(
                0.05, y,
                f"{name:<16} {x:>10.4f} {z:>11.4f} {w:>10.2f} {contrib:>13.4f}",
                fontsize=10.3,
                family="monospace",
                color="#111827",
            )
            y -= 0.031

        fig.text(0.05, 0.07, f"Final score from contributions = {total_check:.4f}",
                 fontsize=11.3, weight="bold", color="#0F172A")
        fig.text(0.05, 0.042, f"Reported score                 = {risk_score:.4f}",
                 fontsize=11.3, weight="bold", color="#0F172A")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def add_results_page(pdf, ct_path, gt_mask_path, pred_mask, features, pred_world, gt_world, dice_score):
        ct = nib.load(ct_path).get_fdata()
        gt_mask = safe_load_mask(gt_mask_path) if gt_mask_path is not None else None

        z = get_best_slice(ct, gt_mask=gt_mask, pred_mask=pred_mask)

        def safe_z(idx, depth):
            depth = int(depth)
            if depth <= 0:
                return 0
            return max(0, min(int(idx), depth - 1))

        z_ct = safe_z(z, ct.shape[2])
        z_gt = safe_z(z, gt_mask.shape[2]) if gt_mask is not None else None
        z_pred = safe_z(z, pred_mask.shape[2]) if pred_mask is not None else None

        mean_err = None
        if gt_world is not None and pred_world:
            summary_rows = []
            for name in SUMMARY_LANDMARKS:
                if name in pred_world and name in gt_world:
                    summary_rows.append(float(np.linalg.norm(pred_world[name] - gt_world[name])))
            if summary_rows:
                mean_err = float(np.mean(summary_rows))

        fig = plt.figure(figsize=(11.69, 8.27))
        draw_header(fig, "Model Outputs and Anatomical Summary")

        fig.text(
            0.05, 0.80,
            f"Segmentation Dice: {dice_score:.4f}" if dice_score is not None else "Segmentation Dice: not available",
            fontsize=12.4, weight="bold", color="#0F766E",
        )
        fig.text(0.05, 0.765, f"Predicted mask voxels: {int((pred_mask > 0).sum()) if pred_mask is not None else 'N/A'}",
                 fontsize=11.3, color="#111827")
        fig.text(0.05, 0.732, f"Connected components: {nd.label((pred_mask > 0).astype(np.uint8))[1] if pred_mask is not None else 'N/A'}",
                 fontsize=11.3, color="#111827")
        fig.text(0.05, 0.699, f"Compactness: {fmt(features.get('compactness', np.nan), 4)}",
                 fontsize=11.3, color="#111827")
        fig.text(0.05, 0.666, f"Volume voxels: {fmt(features.get('volume_voxels', np.nan), 4)}",
                 fontsize=11.3, color="#111827")

        if mean_err is not None:
            fig.text(0.52, 0.80, f"Mean landmark error : {mean_err:.2f} mm",
                     fontsize=12.4, weight="bold", color="#7C3AED")
        else:
            fig.text(0.52, 0.80, "Mean landmark error: not available",
                     fontsize=12.4, weight="bold", color="#7C3AED")

        fig.text(0.52, 0.75, "Heatmap-derived landmark outputs", fontsize=12, weight="bold", color="#111827")
        y = 0.715
        for name in REPORT_LANDMARKS:
            if name in pred_world:
                p = pred_world[name]
                fig.text(0.52, y, f"{name}: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]",
                         fontsize=10.4, color="#111827")
                y -= 0.030

        gs = fig.add_gridspec(1, 3, left=0.05, right=0.95, bottom=0.08, top=0.50, wspace=0.18)
        ct_slice = ct[:, :, z_ct]

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(ct_slice, cmap="gray")
        ax1.set_title(f"CT slice {z_ct}", fontsize=11, weight="bold")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(ct_slice, cmap="gray")
        if gt_mask is not None:
            gt_slice = gt_mask[:, :, z_gt]
            ax2.imshow(np.ma.masked_where(gt_slice <= 0, gt_slice), cmap="Greens", alpha=0.45)
            ax2.set_title("CT + ground truth overlay", fontsize=11, weight="bold")
        else:
            ax2.text(0.5, 0.5, "GT mask not available", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_title("CT + ground truth overlay", fontsize=11, weight="bold")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(ct_slice, cmap="gray")
        if pred_mask is not None:
            pred_slice = pred_mask[:, :, z_pred]
            ax3.imshow(np.ma.masked_where(pred_slice <= 0, pred_slice), cmap="Reds", alpha=0.45)
            ax3.set_title("CT + predicted overlay", fontsize=11, weight="bold")
        else:
            ax3.text(0.5, 0.5, "Prediction not available", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("CT + predicted overlay", fontsize=11, weight="bold")
        ax3.axis("off")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def add_prediction_only_page(pdf, ct_path, features, pred_world, top_reasons, risk_label, risk_score):
        ct = nib.load(ct_path).get_fdata()
        z = ct.shape[2] // 2
        ct_slice = ct[:, :, z]

        fig = plt.figure(figsize=(11.69, 8.27))
        draw_header(fig, "Prediction-Only Summary")

        fig.text(0.05, 0.80, f"Anatomical complexity: {display_anatomy_label(risk_label)}",
                 fontsize=13.0, weight="bold", color="#111827")
        fig.text(0.05, 0.765, f"Planning score: {risk_score:.4f}",
                 fontsize=12.2, weight="bold", color="#0F172A")
        fig.text(0.05, 0.728, f"Compactness: {fmt(features.get('compactness', np.nan), 4)}",
                 fontsize=11.2, color="#111827")
        fig.text(0.05, 0.696, f"Volume voxels: {fmt(features.get('volume_voxels', np.nan), 4)}",
                 fontsize=11.2, color="#111827")
        fig.text(0.05, 0.664, f"Connected components: {fmt(features.get('num_components', np.nan), 4)}",
                 fontsize=11.2, color="#111827")

        fig.text(0.05, 0.620, "Top contributors", fontsize=12.5, weight="bold", color="#111827")
        y = 0.588
        for name, val in top_reasons:
            direction = "lowered" if val < 0 else "raised"
            color = "#166534" if val < 0 else "#991B1B"
            fig.text(0.07, y, f"• {name}", fontsize=11.2, weight="bold", color="#111827")
            fig.text(0.24, y, f"{direction} the score by {abs(val):.3f}", fontsize=11.2, color=color)
            y -= 0.032

        fig.text(0.05, 0.405, "Predicted landmark coordinates (world mm)",
                 fontsize=12.5, weight="bold", color="#111827")
        y = 0.372
        for name in REPORT_LANDMARKS:
            if name in pred_world:
                p = pred_world[name]
                fig.text(0.05, y, f"{name}: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]",
                         fontsize=10.8, color="#111827")
                y -= 0.031

        ax = fig.add_axes([0.53, 0.11, 0.41, 0.70])
        ax.imshow(ct_slice, cmap="gray")
        ax.set_title(f"CT slice {z}", fontsize=11, weight="bold")
        ax.axis("off")

        fig.text(
            0.05, 0.07,
            "This prediction-only report uses the CT scan and model outputs only. It intentionally omits mask overlays and landmark error comparison.",
            fontsize=10.8,
            color="#6B7280",
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


    def add_distance_reference_page(pdf, distance_image_path: Optional[str] = None):
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header(fig, "Reference Landmark-Distance Image")

        if distance_image_path is not None and os.path.exists(distance_image_path):
            img = plt.imread(distance_image_path)
            ax_img = fig.add_axes([0.05, 0.12, 0.90, 0.70])
            ax_img.imshow(img)
            ax_img.axis("off")

        else:
            fig.text(
                0.05, 0.72,
                "Reference image not found at the configured path.",
                fontsize=12.0,
                color="#991B1B",
                weight="bold",
            )
            fig.text(
                0.05, 0.68,
                "Set distance_image_path to the provided landmark-distance image file.",
                fontsize=11.0,
                color="#111827",
            )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def add_notes_page(pdf):
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header(fig, "Notes and Limitations")

        lines = [
            "• This report is intended for research and planning support, not as a stand-alone clinical decision.",
            "• The training set is small, so the thresholds and weights were derived from the available training data and should be validated on larger datasets before clinical use.",
            "• No surgeon-validated outcome labels were available during training, so the model should not be interpreted as a validated surgical safety predictor.",
            "• Prediction errors in landmark localization affect the final summary.",
            "• In practice, the original CT and the full clinical context should always be reviewed before any decision.",
        ]

        y = 0.80
        for line in lines:
            fig.text(0.05, y, line, fontsize=11.1, color="#111827")
            y -= 0.048

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    subject = sample.get("subject", "UNKNOWN")
    ct_path = sample["ct"]
    desc_path = sample.get("desc", None)
    gt_mask_path = sample.get("mask", None)

    if gt_mask_path is not None and not os.path.exists(gt_mask_path):
        gt_mask_path = None

    pred_mask = result.get("pred_mask", None)
    features = result.get("features", {})
    risk_score = float(result.get("risk_score", np.nan))
    risk_label = result.get("risk_label", "UNKNOWN")
    reasons = result.get("reasons", [])
    pred_world = result.get("pred_world", {})

    gt_world = None
    specimen_id = None
    specimen_side = None
    if desc_path is not None and os.path.exists(desc_path):
        try:
            desc = load_desc(desc_path)
            gt_world = desc.get("landmarks", None)
            specimen_id = desc.get("id", None)
            specimen_side = desc.get("side", None)
        except Exception:
            gt_world = None

    gt_mask = safe_load_mask(gt_mask_path)
    dice_score = None
    if gt_mask is not None and pred_mask is not None and gt_mask.shape == pred_mask.shape:
        dice_score = compute_binary_dice(gt_mask, pred_mask)

    top_reasons = top_feature_reasons(reasons, top_k=3)

    risk_profile = result.get("risk_profile", None)
    if risk_profile is None:
        raise ValueError("risk_profile was not found. Pass it in result or define it in the result dict.")

    recommendation = surgical_recommendation(risk_label, features)
    display_label = display_anatomy_label(risk_label)

    print("\n" + "=" * 86)
    print("CASE REPORT SUMMARY")
    print("=" * 86)
    print(f"Project: {PROJECT_NAME}")
    print(f"Dataset: {dataset_name}")
    print(f"Subject: {subject}")
    if specimen_id is not None:
        print(f"Specimen ID: {specimen_id}")
    if specimen_side is not None:
        print(f"Side: {specimen_side}")
    print(f"Anatomical complexity: {display_anatomy_label(risk_label)}")
    print(f"Planning score: {risk_score:.4f}")
    if dice_score is not None:
        print(f"Segmentation Dice: {dice_score:.4f}")
    print("\nWhy this label was assigned:")
    for name, val in top_reasons:
        direction = "decreased" if val < 0 else "increased"
        print(f"- {name} {direction} the score by {abs(val):.3f}")
    print("\nRecommendation:")
    print(recommendation)

    if out_pdf_path is None:
        out_pdf_path = os.path.join(REPORT_DIR, f"{subject}_surgical_report.pdf")

    with PdfPages(out_pdf_path) as pdf:
        add_summary_page(pdf, subject, specimen_id, specimen_side, risk_label, risk_score, dice_score, top_reasons, features)

        if include_mask_visualization:
            fig = visualize_3d_masks(ct_path, gt_mask_path=gt_mask_path, pred_mask=pred_mask)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            add_results_page(pdf, ct_path, gt_mask_path, pred_mask, features, pred_world, gt_world, dice_score)
        else:
            if pred_mask is not None:
                fig = visualize_3d_prediction_only(ct_path, pred_mask=pred_mask)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            add_prediction_only_page(pdf, ct_path, features, pred_world, top_reasons, risk_label, risk_score)

        add_risk_logic_page(pdf, risk_profile, risk_label, risk_score, features)

        add_distance_reference_page(pdf, distance_image_path=distance_image_path)

        add_notes_page(pdf)

    if show_in_notebook:
        plt.show()

    print(f"\nPDF saved to: {out_pdf_path}")
    return out_pdf_path


# ============================================================
# PIPELINE
# ============================================================
def run_full_pipeline_from_heatmaps(
    ct_path: str,
    segmentation_model: nn.Module,
    landmark_model: nn.Module,
    device: torch.device | str,
    risk_profile: Dict[str, Any],
    gt_mask_path: Optional[str] = None,
    show: bool = False,
) -> Dict[str, Any]:
    pred_mask = predict_subject_mask(segmentation_model, ct_path, device)
    pred_world, pred_vox, pred_hm = predict_landmarks_from_heatmaps(landmark_model, ct_path, device)
    features = extract_landmark_concepts_from_predictions(pred_world, pred_mask=pred_mask)
    risk_score, risk_label, reasons = predict_risk_from_features(features, risk_profile)

    result = {
        "pred_mask": pred_mask,
        "pred_world": pred_world,
        "pred_vox": pred_vox,
        "pred_hm": pred_hm,
        "features": features,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "reasons": reasons,
        "risk_profile": risk_profile,
        "ct_path": ct_path,
        "gt_mask_path": gt_mask_path,
    }

    if show:
        print(f"Risk label: {risk_label}")
        print(f"Planning score: {risk_score:.4f}")

    return result


# ============================================================
# ARTIFACT LOADING
# ============================================================
def load_segmentation_model(weights_path: str, device: torch.device | str):
    model = UNetSmall().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_landmark_model(weights_path: str, device: torch.device | str, out_channels: int = len(LANDMARK_NAMES)):
    model = UNet3D(in_channels=1, out_channels=out_channels, base=16).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_risk_profile(
    path: Optional[str] = None,
    *,
    dataset_root: Optional[str] = None,
    segmentation_model: Optional[nn.Module] = None,
    landmark_model: Optional[nn.Module] = None,
    device: Optional[torch.device | str] = None,
) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Convert JSON-friendly values back to pandas objects if needed.
        mu = pd.Series(payload["mu"])
        sigma = pd.Series(payload["sigma"])
        payload["mu"] = mu
        payload["sigma"] = sigma
        return payload

    if dataset_root and segmentation_model is not None and landmark_model is not None and device is not None:
        train_data = build_dataset(dataset_root, TRAIN_SUBJECTS)
        risk_profile, train_df = fit_risk_profile_from_training(
            train_data,
            segmentation_model=segmentation_model,
            landmark_model=landmark_model,
            device=device,
        )
        risk_profile["train_scores"] = None
        risk_profile["source"] = "computed_from_dataset"
        return risk_profile

    # Fallback: useful for bootstrapping the app even when training data is missing.
    return {
        "mu": pd.Series({k: 0.0 for k in RISK_FEATURES}),
        "sigma": pd.Series({k: 1.0 for k in RISK_FEATURES}),
        "q_safe": 0.0,
        "q_mod": 1.0,
        "source": "fallback",
    }
