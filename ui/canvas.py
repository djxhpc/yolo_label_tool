"""影像畫布元件(完整重寫版)
重點修正:
1. 標註框使用 widget 座標系存儲,避免拖曳時座標換算誤差
2. 統一使用 OpenCV 讀取影像,搭配 EXIF 旋轉處理
3. 框的互動:繪製/拖曳/8 個控制點(4 角 + 4 邊)/選中/刪除
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QPointF, QRectF, QPoint, QSizeF, pyqtSignal, QSize
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QBrush, QColor, QFont,
                         QImage, QCursor, QWheelEvent, QMouseEvent,
                         QPaintEvent, QKeyEvent, QPalette, )

from core.yolo_io import YoloAnnotation, read_image_standardized


# 預設類別顏色表
DEFAULT_COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A",
    "#92CC17", "#3DDB86", "#1A9334", "#00D4BB", "#2C99A8", "#00C2FF",
    "#344593", "#6473FF", "#0018EC", "#8438FF", "#520085", "#CB38FF",
    "#FF95C8", "#FF37C7",
]


class BBoxItem:
    """標註框(以 widget 座標儲存,避免反覆換算)"""
    HANDLE_NONE = 0
    HANDLE_MOVE = 1
    HANDLE_TL = 2
    HANDLE_TR = 3
    HANDLE_BL = 4
    HANDLE_BR = 5
    HANDLE_TM = 6   # 上中
    HANDLE_BM = 7   # 下中
    HANDLE_LM = 8   # 左中
    HANDLE_RM = 9   # 右中

    def __init__(self, annotation: YoloAnnotation, class_name: str, color: QColor):
        self.annotation = annotation
        self.class_name = class_name
        self.color = color
        self.selected = False
        # widget 座標(動態維護)
        self.widget_rect = QRectF()

    def contains_point(self, point: QPointF, handle_size: float = 8.0) -> int:
        """判斷滑鼠位置:回傳 handle id"""
        if not self.widget_rect.isValid() or self.widget_rect.isEmpty():
            return self.HANDLE_NONE
        r = self.widget_rect.normalized()
        hs = handle_size

        # 8 個控制點優先
        handles = [
            (self.HANDLE_TL, r.topLeft()),
            (self.HANDLE_TR, r.topRight()),
            (self.HANDLE_BL, r.bottomLeft()),
            (self.HANDLE_BR, r.bottomRight()),
            (self.HANDLE_TM, QPointF(r.center().x(), r.top())),
            (self.HANDLE_BM, QPointF(r.center().x(), r.bottom())),
            (self.HANDLE_LM, QPointF(r.left(), r.center().y())),
            (self.HANDLE_RM, QPointF(r.right(), r.center().y())),
        ]
        for hid, p in handles:
            if QRectF(p.x() - hs, p.y() - hs, hs * 2, hs * 2).contains(point):
                return hid
        if r.contains(point):
            return self.HANDLE_MOVE
        return self.HANDLE_NONE


class ImageCanvas(QWidget):
    """影像畫布"""

    box_changed = pyqtSignal()
    box_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 影像
        self.original_pixmap: Optional[QPixmap] = None
        self.image_size = QSize(0, 0)  # 原始尺寸(已套 EXIF 旋轉)

        # 視圖狀態
        self.scale = 1.0
        self.min_scale = 0.05
        self.max_scale = 20.0
        self.offset = QPointF(0, 0)
        self.fit_to_window = True

        # 標註
        self.boxes: List[BBoxItem] = []
        self.selected_box: Optional[BBoxItem] = None

        # 互動狀態
        self.dragging = False
        self.active_handle = BBoxItem.HANDLE_NONE
        self.drag_start_pos = QPointF()
        self.drag_start_widget_rect = QRectF()
        self.pan_mode = False
        self.pan_start_pos = QPointF()
        self.pan_start_offset = QPointF()

        # 建立新框
        self.creating = False
        self.create_start_widget = QPointF()  # widget 座標
        self.create_end_widget = QPointF()

        # 背景
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(40, 40, 40))
        self.setPalette(pal)

    # =====================================================
    # 影像載入
    # =====================================================
    def load_image(self, image_path: str) -> bool:
        """載入影像(使用 EXIF 安全讀取)"""
        try:
            bgr, (w, h) = read_image_standardized(image_path)
            if bgr is None or w == 0 or h == 0:
                return False
            # BGR -> RGB -> QImage -> QPixmap
            rgb = bgr[..., ::-1].copy()  # BGR -> RGB
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg)
            if pixmap.isNull():
                return False
            self.original_pixmap = pixmap
            self.image_size = QSize(w, h)
            self.boxes.clear()
            self.selected_box = None
            self.box_selected.emit(None)
            if self.fit_to_window:
                self.fit_image()
            else:
                self.center_image()
            self.update()
            return True
        except Exception as e:
            print(f"load_image error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def clear_image(self):
        self.original_pixmap = None
        self.image_size = QSize(0, 0)
        self.boxes.clear()
        self.selected_box = None
        self.box_selected.emit(None)
        self.update()

    def fit_image(self):
        if self.original_pixmap is None:
            return
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return
        sx = w / self.image_size.width()
        sy = h / self.image_size.height()
        self.scale = min(sx, sy) * 0.98
        self.scale = max(self.min_scale, min(self.max_scale, self.scale))
        self.center_image()

    def center_image(self):
        if self.original_pixmap is None:
            return
        w, h = self.width(), self.height()
        scaled_w = self.image_size.width() * self.scale
        scaled_h = self.image_size.height() * self.scale
        self.offset = QPointF((w - scaled_w) / 2.0, (h - scaled_h) / 2.0)
        self.update()

    # =====================================================
    # 座標轉換
    # =====================================================
    def widget_to_image(self, pos: QPointF) -> QPointF:
        """widget 座標 -> 原始影像座標"""
        if self.image_size.width() == 0:
            return QPointF(0, 0)
        ix = (pos.x() - self.offset.x()) / self.scale
        iy = (pos.y() - self.offset.y()) / self.scale
        return QPointF(ix, iy)

    def image_to_widget(self, pos: QPointF) -> QPointF:
        """原始影像座標 -> widget 座標"""
        wx = pos.x() * self.scale + self.offset.x()
        wy = pos.y() * self.scale + self.offset.y()
        return QPointF(wx, wy)

    def yolo_to_pixel_rect(self, ann: YoloAnnotation) -> QRectF:
        """YOLO 格式 -> 像素矩形"""
        if self.image_size.width() == 0:
            return QRectF()
        cx = ann.x_center * self.image_size.width()
        cy = ann.y_center * self.image_size.height()
        w = ann.width * self.image_size.width()
        h = ann.height * self.image_size.height()
        return QRectF(cx - w / 2, cy - h / 2, w, h)

    def pixel_to_yolo(self, rect: QRectF, class_id=None) -> YoloAnnotation:
        """
        像素矩形 -> YOLO 格式

        重點：
        - 如果有傳入 class_id，就使用傳入的 class_id
        - 如果沒傳入，但目前有 selected_box，就保留 selected_box 原本的 class_id
        - 避免拖曳/縮放標註框後 class_id 被重設成 0
        """
        iw = self.image_size.width()
        ih = self.image_size.height()

        if iw == 0 or ih == 0:
            if class_id is None:
                class_id = 0
            return YoloAnnotation(class_id, 0, 0, 0, 0)

        x1 = max(0.0, min(rect.left(), iw))
        y1 = max(0.0, min(rect.top(), ih))
        x2 = max(0.0, min(rect.right(), iw))
        y2 = max(0.0, min(rect.bottom(), ih))

        # 防止寬高變負數
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        cx = (x1 + x2) / 2.0 / iw
        cy = (y1 + y2) / 2.0 / ih
        w = (x2 - x1) / iw
        h = (y2 - y1) / ih

        # 保留原本類別
        if class_id is None:
            if self.selected_box is not None:
                class_id = self.selected_box.annotation.class_id
            else:
                class_id = 0

        return YoloAnnotation(int(class_id), cx, cy, w, h)

    # =====================================================
    # 標註框管理
    # =====================================================
    def _update_all_widget_rects(self):
        """當 scale/offset 變動時,同步所有框的 widget 座標"""
        for box in self.boxes:
            pix_rect = self.yolo_to_pixel_rect(box.annotation)
            tl = self.image_to_widget(pix_rect.topLeft())
            br = self.image_to_widget(pix_rect.bottomRight())
            box.widget_rect = QRectF(tl, br).normalized()

    def set_annotations(self, annotations: List[YoloAnnotation], class_names: List[str]):
        self.boxes.clear()
        self.selected_box = None
        for ann in annotations:
            cls_name = class_names[ann.class_id] if ann.class_id < len(class_names) else f"class_{ann.class_id}"
            color = QColor(DEFAULT_COLORS[ann.class_id % len(DEFAULT_COLORS)])
            bbox = BBoxItem(ann, cls_name, color)
            self.boxes.append(bbox)
        self._update_all_widget_rects()
        self.update()

    def get_annotations(self) -> List[YoloAnnotation]:
        return [b.annotation for b in self.boxes]

    def add_annotation(self, ann: YoloAnnotation, class_name: str = "object"):
        color = QColor(DEFAULT_COLORS[ann.class_id % len(DEFAULT_COLORS)])
        bbox = BBoxItem(ann, class_name, color)
        # 計算 widget 座標
        pix_rect = self.yolo_to_pixel_rect(ann)
        tl = self.image_to_widget(pix_rect.topLeft())
        br = self.image_to_widget(pix_rect.bottomRight())
        bbox.widget_rect = QRectF(tl, br).normalized()
        self.boxes.append(bbox)
        self.selected_box = bbox
        for b in self.boxes:
            b.selected = (b == bbox)
        self.box_selected.emit(bbox)
        self.box_changed.emit()
        self.update()

    def delete_selected(self):
        if self.selected_box and self.selected_box in self.boxes:
            self.boxes.remove(self.selected_box)
            self.selected_box = None
            self.box_selected.emit(None)
            self.box_changed.emit()
            self.update()

    def set_selected_class(self, class_id: int, class_names: List[str]):
        """
        更換目前選中標註框的類別
        """
        if self.selected_box:
            self.selected_box.annotation.class_id = int(class_id)

            if class_id < len(class_names):
                self.selected_box.class_name = class_names[class_id]
            else:
                self.selected_box.class_name = f"class_{class_id}"

            self.selected_box.color = QColor(
                DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)]
            )

            self.box_changed.emit()
            self.update()

    # =====================================================
    # 滑鼠事件
    # =====================================================
    def resizeEvent(self, event):
        if self.fit_to_window and self.original_pixmap is not None:
            self.fit_image()
        super().resizeEvent(event)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self.original_pixmap is None:
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "請載入影像資料夾")
            return

        # 影像
        target = QRectF(self.offset.x(), self.offset.y(),
                        self.image_size.width() * self.scale,
                        self.image_size.height() * self.scale)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 4)
        painter.drawPixmap(target, self.original_pixmap, QRectF(self.original_pixmap.rect()))

        # 影像邊框
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(target)

        # 標註框
        for box in self.boxes:
            self._draw_box(painter, box)

        # 建立中
        if self.creating:
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 255, 0, 30))
            rect = QRectF(self.create_start_widget, self.create_end_widget).normalized()
            painter.drawRect(rect)

    def _draw_box(self, painter: QPainter, box: BBoxItem):
        r = box.widget_rect.normalized()
        if not r.isValid() or r.isEmpty():
            return
        pen = QPen(box.color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(r)

        # 標籤
        label = box.class_name
        font = QFont("Arial", max(8, int(10 * min(1.0, self.scale * 0.4 + 0.5))))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(label) + 8
        text_h = metrics.height() + 4
        label_rect = QRectF(r.left(), max(0, r.top() - text_h), text_w, text_h)
        painter.fillRect(label_rect, box.color)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(label_rect, Qt.AlignCenter, label)

        # 選中控制點
        if box.selected:
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(QPen(box.color, 1))
            hs = 5
            pts = [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight(),
                   QPointF(r.center().x(), r.top()),
                   QPointF(r.center().x(), r.bottom()),
                   QPointF(r.left(), r.center().y()),
                   QPointF(r.right(), r.center().y())]
            for p in pts:
                painter.drawRect(QRectF(p.x() - hs, p.y() - hs, hs * 2, hs * 2))

    def wheelEvent(self, event: QWheelEvent):
        if self.original_pixmap is None:
            return
        self.fit_to_window = False

        mouse_pos = QPointF(event.pos())
        img_pos_before = self.widget_to_image(mouse_pos)
        old_scale = self.scale

        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        new_scale = old_scale * factor
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))
        self.scale = new_scale

        img_pos_after_widget = QPointF(img_pos_before.x() * self.scale, img_pos_before.y() * self.scale)
        self.offset = QPointF(mouse_pos.x() - img_pos_after_widget.x(),
                              mouse_pos.y() - img_pos_after_widget.y())
        # 重新計算所有框
        self._update_all_widget_rects()
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self.original_pixmap is None:
            return
        self.setFocus()
        pos = QPointF(event.pos())

        # 平移:中鍵 或 空白+左鍵
        if event.button() == Qt.MidButton or (event.button() == Qt.LeftButton and self.pan_mode):
            self.dragging = True
            self.pan_start_pos = pos
            self.pan_start_offset = QPointF(self.offset)
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            # 找最上層的框
            hit_box = None
            hit_handle = BBoxItem.HANDLE_NONE
            for box in reversed(self.boxes):
                h = box.contains_point(pos)
                if h != BBoxItem.HANDLE_NONE:
                    hit_box = box
                    hit_handle = h
                    break
                if box.widget_rect.normalized().contains(pos):
                    hit_box = box
                    hit_handle = BBoxItem.HANDLE_MOVE

            if hit_box:
                # 選中並開始拖曳
                for b in self.boxes:
                    b.selected = False
                hit_box.selected = True
                self.selected_box = hit_box
                self.box_selected.emit(hit_box)

                self.dragging = True
                self.active_handle = hit_handle
                self.drag_start_pos = pos
                self.drag_start_widget_rect = QRectF(hit_box.widget_rect)
                self.update()
            else:
                # 開始繪製新框
                for b in self.boxes:
                    b.selected = False
                self.selected_box = None
                self.box_selected.emit(None)
                if not self.pan_mode:
                    self.creating = True
                    self.create_start_widget = pos
                    self.create_end_widget = pos
                    self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = QPointF(event.pos())

        # 平移
        if self.pan_mode and self.dragging:
            self.offset = self.pan_start_offset + (pos - self.pan_start_pos)
            self._update_all_widget_rects()
            self.update()
            return

        # 繪製中
        if self.creating:
            self.create_end_widget = pos
            self.update()
            return

        # 拖曳既有框(在 widget 座標中操作,避免誤差累積)
        if self.dragging and self.selected_box:
            delta = pos - self.drag_start_pos
            new_rect = QRectF(self.drag_start_widget_rect)

            if self.active_handle == BBoxItem.HANDLE_MOVE:
                new_rect.translate(delta)
            elif self.active_handle == BBoxItem.HANDLE_TL:
                new_rect.setTopLeft(self.drag_start_widget_rect.topLeft() + delta)
            elif self.active_handle == BBoxItem.HANDLE_TR:
                new_rect.setTopRight(self.drag_start_widget_rect.topRight() + delta)
            elif self.active_handle == BBoxItem.HANDLE_BL:
                new_rect.setBottomLeft(self.drag_start_widget_rect.bottomLeft() + delta)
            elif self.active_handle == BBoxItem.HANDLE_BR:
                new_rect.setBottomRight(self.drag_start_widget_rect.bottomRight() + delta)
            elif self.active_handle == BBoxItem.HANDLE_TM:
                new_rect.setTop(self.drag_start_widget_rect.top() + delta.y())
            elif self.active_handle == BBoxItem.HANDLE_BM:
                new_rect.setBottom(self.drag_start_widget_rect.bottom() + delta.y())
            elif self.active_handle == BBoxItem.HANDLE_LM:
                new_rect.setLeft(self.drag_start_widget_rect.left() + delta.x())
            elif self.active_handle == BBoxItem.HANDLE_RM:
                new_rect.setRight(self.drag_start_widget_rect.right() + delta.x())

            new_rect = new_rect.normalized()
            # 限制 widget 框不能超出影像範圍
            img_rect = QRectF(self.offset, QSizeF(
                self.image_size.width() * self.scale,
                self.image_size.height() * self.scale
            ))
            if new_rect.left() < img_rect.left():
                if self.active_handle in (BBoxItem.HANDLE_TL, BBoxItem.HANDLE_BL, BBoxItem.HANDLE_LM):
                    new_rect.setLeft(img_rect.left())
                else:
                    new_rect.setLeft(img_rect.left())
                    new_rect.setRight(max(img_rect.left() + 2, new_rect.right()))
            if new_rect.top() < img_rect.top():
                if self.active_handle in (BBoxItem.HANDLE_TL, BBoxItem.HANDLE_TR, BBoxItem.HANDLE_TM):
                    new_rect.setTop(img_rect.top())
                else:
                    new_rect.setTop(img_rect.top())
                    new_rect.setBottom(max(img_rect.top() + 2, new_rect.bottom()))
            if new_rect.right() > img_rect.right():
                if self.active_handle in (BBoxItem.HANDLE_TR, BBoxItem.HANDLE_BR, BBoxItem.HANDLE_RM):
                    new_rect.setRight(img_rect.right())
                else:
                    new_rect.setRight(img_rect.right())
                    new_rect.setLeft(min(img_rect.right() - 2, new_rect.left()))
            if new_rect.bottom() > img_rect.bottom():
                if self.active_handle in (BBoxItem.HANDLE_BL, BBoxItem.HANDLE_BR, BBoxItem.HANDLE_BM):
                    new_rect.setBottom(img_rect.bottom())
                else:
                    new_rect.setBottom(img_rect.bottom())
                    new_rect.setTop(min(img_rect.bottom() - 2, new_rect.top()))

            self.selected_box.widget_rect = new_rect
            # 同步 YOLO 標註
            pix_rect = self._widget_to_image_rect(new_rect)
            old_class_id = self.selected_box.annotation.class_id
            self.selected_box.annotation = self.pixel_to_yolo(pix_rect, class_id=old_class_id)
            self.box_changed.emit()
            self.update()
            return

        # hover 時更新游標
        cursor = Qt.ArrowCursor
        for box in reversed(self.boxes):
            h = box.contains_point(pos)
            if h in (BBoxItem.HANDLE_TL, BBoxItem.HANDLE_BR):
                cursor = Qt.SizeFDiagCursor
                break
            elif h in (BBoxItem.HANDLE_TR, BBoxItem.HANDLE_BL):
                cursor = Qt.SizeBDiagCursor
                break
            elif h in (BBoxItem.HANDLE_TM, BBoxItem.HANDLE_BM):
                cursor = Qt.SizeVerCursor
                break
            elif h in (BBoxItem.HANDLE_LM, BBoxItem.HANDLE_RM):
                cursor = Qt.SizeHorCursor
                break
            elif h == BBoxItem.HANDLE_MOVE or box.widget_rect.normalized().contains(pos):
                cursor = Qt.SizeAllCursor
                break
        self.setCursor(cursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MidButton or (self.pan_mode and event.button() == Qt.LeftButton):
            self.dragging = False
            self.setCursor(Qt.OpenHandCursor if self.pan_mode else Qt.ArrowCursor)
            return

        if self.creating and event.button() == Qt.LeftButton:
            self.creating = False
            rect = QRectF(self.create_start_widget, self.create_end_widget).normalized()
            if rect.width() > 5 and rect.height() > 5:
                # 將 widget 矩形轉成影像像素矩形
                pix_rect = self._widget_to_image_rect(rect)
                if pix_rect.width() > 1 and pix_rect.height() > 1:
                    cls_id = 0
                    cls_name = "object"
                    if self.parent() and hasattr(self.parent(), 'get_default_class'):
                        cls_id, cls_name = self.parent().get_default_class()
                    ann = self.pixel_to_yolo(pix_rect)
                    ann.class_id = cls_id
                    self.add_annotation(ann, cls_name)
            self.update()

        if self.dragging:
            self.dragging = False
            self.active_handle = BBoxItem.HANDLE_NONE

    def _widget_to_image_rect(self, widget_rect: QRectF) -> QRectF:
        """widget 矩形 -> 影像像素矩形"""
        tl = self.widget_to_image(widget_rect.topLeft())
        br = self.widget_to_image(widget_rect.bottomRight())
        return QRectF(tl, br).normalized()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.pan_mode = True
            self.setCursor(Qt.OpenHandCursor)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.pan_mode = False
            self.setCursor(Qt.ArrowCursor)
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()

    @property
    def widget_size(self):
        return self.size()