"""YOLO 格式標註檔案讀寫模組 + 統一影像讀取(處理 EXIF)"""
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


class YoloAnnotation:
    """YOLO 格式標註:class_id, x_center, y_center, width, height (normalized 0-1)"""

    def __init__(self, class_id: int, x_center: float, y_center: float,
                 width: float, height: float):
        self.class_id = class_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def to_line(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    @classmethod
    def from_line(cls, line: str) -> 'YoloAnnotation':
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO line: {line}")
        return cls(int(parts[0]), float(parts[1]), float(parts[2]),
                   float(parts[3]), float(parts[4]))


def find_image_files(folder: str, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')):
    """遞迴尋找資料夾中所有影像檔案"""
    folder = Path(folder)
    image_files = []
    for ext in extensions:
        image_files.extend(folder.rglob(f'*{ext}'))
        image_files.extend(folder.rglob(f'*{ext.upper()}'))
    return sorted(set(image_files))


def get_label_path(image_path: Path, images_dir: Path, labels_dir: Optional[Path] = None) -> Path:
    """根據影像路徑推算對應 label 路徑"""
    try:
        rel_path = image_path.relative_to(images_dir)
    except ValueError:
        rel_path = Path(image_path.name)

    if labels_dir is None:
        return image_path.with_suffix('.txt')
    label_rel = rel_path.with_suffix('.txt')
    return labels_dir / label_rel


def read_labels(label_path: Path) -> List[YoloAnnotation]:
    """讀取 YOLO 標註檔"""
    annotations = []
    if not label_path.exists():
        return annotations
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    annotations.append(YoloAnnotation.from_line(line))
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
    return annotations


def write_labels(label_path: Path, annotations: List[YoloAnnotation]) -> bool:
    """寫入 YOLO 標註檔"""
    try:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                f.write(ann.to_line() + '\n')
        return True
    except Exception as e:
        print(f"Error writing {label_path}: {e}")
        return False


# ============================================================
# 統一影像讀取(處理 EXIF 旋轉)
# ============================================================

def _get_exif_orientation(img):
    """取得 PIL Image 的 EXIF orientation"""
    try:
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(0x0112)  # 274
            return orientation
    except Exception:
        pass
    return None


def _apply_exif_rotation(img):
    """依 EXIF orientation 旋轉影像,並回傳旋轉後的影像"""
    orientation = _get_exif_orientation(img)
    if orientation is None:
        return img
    # 常見 orientation 處理
    from PIL import Image
    if orientation == 2:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        img = img.transpose(Image.ROTATE_180)
    elif orientation == 4:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif orientation == 5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
    elif orientation == 6:
        img = img.transpose(Image.ROTATE_270)
    elif orientation == 7:
        img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
    elif orientation == 8:
        img = img.transpose(Image.ROTATE_90)
    return img


def read_image_standardized(image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    統一讀取影像,處理 EXIF,回傳:
    - BGR numpy array (OpenCV 格式)
    - (width, height) 已套用旋轉後的尺寸
    """
    from PIL import Image
    import cv2

    # 用 PIL 讀取以取得 EXIF
    pil_img = Image.open(image_path)
    pil_img = _apply_exif_rotation(pil_img)

    # 統一轉成 RGB
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    # 轉成 numpy (RGB)
    arr = np.array(pil_img)
    h, w = arr.shape[:2]

    # 轉 BGR 給 OpenCV
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr, (w, h)