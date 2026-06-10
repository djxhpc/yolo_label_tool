"""YOLO 模型預測模組(支援 ultralytics YOLO 與自定義 PyTorch 模型)"""
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal


class PredictionWorker(QThread):
    """背景執行緒執行預測,避免 UI 凍結"""
    finished = pyqtSignal(list)      # 預測結果 list of (class_id, conf, x, y, w, h)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, model, image_path: str, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, model_type: str = 'ultralytics'):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_type = model_type

    def run(self):
        try:
            if self.model_type == 'ultralytics':
                results = self.model(
                    self.image_path,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                annotations = []
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    if boxes is not None:
                        # 取得影像尺寸以正規化
                        img_h, img_w = result.orig_shape
                        for i in range(len(boxes)):
                            cls_id = int(boxes.cls[i].item())
                            conf = float(boxes.conf[i].item())
                            xyxy = boxes.xyxy[i].cpu().numpy()
                            x1, y1, x2, y2 = xyxy
                            x_center = ((x1 + x2) / 2.0) / img_w
                            y_center = ((y1 + y2) / 2.0) / img_h
                            width = (x2 - x1) / img_w
                            height = (y2 - y1) / img_h
                            annotations.append((cls_id, conf, x_center, y_center, width, height))
                self.finished.emit(annotations)
            else:
                # 自定義模型(範例:torch hub)
                import torch
                from PIL import Image
                device = next(self.model.parameters()).device
                img = Image.open(self.image_path).convert('RGB')
                img_w, img_h = img.size
                # 此處需依你的自定義模型 forward 調整
                self.finished.emit([])
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class ModelManager:
    """模型載入管理器"""
    ULTRA = 'ultralytics'
    CUSTOM = 'custom'

    def __init__(self):
        self.model = None
        self.model_type: Optional[str] = None
        self.model_path: Optional[str] = None
        self.class_names: List[str] = []

    def load_ultralytics(self, model_path: str) -> Tuple[bool, str]:
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_type = self.ULTRA
            self.model_path = model_path
            # 取得類別名稱
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = [self.model.names[i] for i in range(len(self.model.names))]
            return True, f"模型載入成功: {os.path.basename(model_path)}"
        except Exception as e:
            return False, f"載入失敗: {e}"

    def is_loaded(self) -> bool:
        return self.model is not None