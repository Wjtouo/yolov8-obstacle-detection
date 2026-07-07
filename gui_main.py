import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QStatusBar, QSlider)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO


def pick_best_weight() -> str:
    """优先使用最新微调权重，否则依次回退。"""
    candidates = [
        "H:/yolo_runs/speed_bump_boost_v7/weights/best.pt",
        "H:/yolo_runs/speed_bump_boost_v6/weights/best.pt",
        "H:/yolo_runs/speed_bump_boost_v3/weights/best.pt",
        "H:/yolo_runs/finetune_8cls_ood_boost/weights/best.pt",
        "H:/yolo_runs/finetune_8cls_add_stone/weights/best.pt",
        "H:/yolo_runs/finetune_7cls_stable3/weights/best.pt",
        "H:/yolo_runs/bdd_plus_coco/weights/best.pt",
        "H:/yolo_runs/bdd_coco_finetune/weights/best.pt",
        "H:/yolo_runs/bdd_s_fast/weights/best.pt",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return candidates[-1]


class ObstacleDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_image = None
        self.detected_image = None
        
        # 视频相关
        self.video_capture = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.process_video_frame)
        self.is_video_playing = False

        # 摄像头相关
        self.camera_capture = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.process_camera_frame)
        self.camera_running = False
        
        self.default_model_path = pick_best_weight()
        self.confidence = 0.18
        
        self.initUI()
        self.load_model(self.default_model_path)

    def initUI(self):
        self.setWindowTitle('基于机器学习的障碍物检测系统设计')
        self.resize(1100, 760)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        title_label = QLabel("基于机器学习的障碍物检测系统设计")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        self.image_label = QLabel("请上传图片、视频或开启摄像头进行检测")
        self.image_label.setFont(QFont("Microsoft YaHei", 14))
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #e0e0e0; border: 2px dashed #aaa;")
        self.image_label.setMinimumSize(900, 520)
        main_layout.addWidget(self.image_label, stretch=1)

        # 第一行按钮：图片操作
        btn_row1 = QHBoxLayout()
        
        self.btn_upload_img = QPushButton("上传图片")
        self.btn_upload_img.setMinimumHeight(42)
        self.btn_upload_img.setFont(QFont("Microsoft YaHei", 12))
        self.btn_upload_img.clicked.connect(self.upload_image)
        
        self.btn_detect_img = QPushButton("检测图片")
        self.btn_detect_img.setMinimumHeight(42)
        self.btn_detect_img.setFont(QFont("Microsoft YaHei", 12))
        self.btn_detect_img.clicked.connect(self.detect_image)
        self.btn_detect_img.setEnabled(False)
        
        self.btn_save_img = QPushButton("保存结果")
        self.btn_save_img.setMinimumHeight(42)
        self.btn_save_img.setFont(QFont("Microsoft YaHei", 12))
        self.btn_save_img.clicked.connect(self.save_image)
        self.btn_save_img.setEnabled(False)

        btn_row1.addWidget(self.btn_upload_img)
        btn_row1.addWidget(self.btn_detect_img)
        btn_row1.addWidget(self.btn_save_img)
        main_layout.addLayout(btn_row1)

        # conf滑块
        conf_row = QHBoxLayout()
        conf_label = QLabel("conf阈值")
        conf_label.setFont(QFont("Microsoft YaHei", 11))
        self.conf_value_label = QLabel(f"{self.confidence:.2f}")
        self.conf_value_label.setFont(QFont("Microsoft YaHei", 11))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(50)
        self.conf_slider.setValue(int(self.confidence * 100))
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        conf_row.addWidget(conf_label)
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_value_label)
        main_layout.addLayout(conf_row)

        # 第二行按钮：视频 + 摄像头操作
        btn_row2 = QHBoxLayout()

        self.btn_upload_vid = QPushButton("上传视频")
        self.btn_upload_vid.setMinimumHeight(42)
        self.btn_upload_vid.setFont(QFont("Microsoft YaHei", 12))
        self.btn_upload_vid.clicked.connect(self.upload_video)
        
        self.btn_play_vid = QPushButton("播放/暂停视频")
        self.btn_play_vid.setMinimumHeight(42)
        self.btn_play_vid.setFont(QFont("Microsoft YaHei", 12))
        self.btn_play_vid.clicked.connect(self.toggle_video)
        self.btn_play_vid.setEnabled(False)

        self.btn_start_camera = QPushButton("开启摄像头实时检测")
        self.btn_start_camera.setMinimumHeight(42)
        self.btn_start_camera.setFont(QFont("Microsoft YaHei", 12))
        self.btn_start_camera.clicked.connect(self.start_camera)

        self.btn_stop_camera = QPushButton("停止摄像头")
        self.btn_stop_camera.setMinimumHeight(42)
        self.btn_stop_camera.setFont(QFont("Microsoft YaHei", 12))
        self.btn_stop_camera.clicked.connect(self.stop_camera)
        self.btn_stop_camera.setEnabled(False)

        btn_row2.addWidget(self.btn_upload_vid)
        btn_row2.addWidget(self.btn_play_vid)
        btn_row2.addWidget(self.btn_start_camera)
        btn_row2.addWidget(self.btn_stop_camera)
        main_layout.addLayout(btn_row2)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("系统初始化完成。")

    # ── 模型加载 ──────────────────────────────────────────

    def load_model(self, model_path):
        if Path(model_path).exists() or model_path == "yolov8n.pt":
            self.statusBar.showMessage(f"正在加载模型: {model_path} ...")
            QApplication.processEvents()
            try:
                self.model = YOLO(model_path)
                self.statusBar.showMessage(f"模型加载成功: {model_path}")
            except Exception as e:
                self.statusBar.showMessage("模型加载失败！")
                QMessageBox.critical(self, "错误", f"模型加载失败:\n{str(e)}")
        else:
            self.statusBar.showMessage(f"未找到模型文件: {model_path}")

    # ── 停止辅助 ──────────────────────────────────────────

    def stop_video_if_playing(self):
        if self.video_timer.isActive():
            self.video_timer.stop()
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.is_video_playing = False
        self.btn_play_vid.setText("播放视频")
        self.btn_play_vid.setEnabled(False)

    def stop_camera_if_running(self):
        if self.camera_timer.isActive():
            self.camera_timer.stop()
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None
        self.camera_running = False
        self.btn_start_camera.setEnabled(True)
        self.btn_stop_camera.setEnabled(False)
        self.btn_upload_img.setEnabled(True)
        self.btn_upload_vid.setEnabled(True)

    # ── 图片功能 ──────────────────────────────────────────

    def upload_image(self):
        if self.camera_running:
            QMessageBox.information(self, "提示", "请先停止摄像头，再上传图片。")
            return
        self.stop_video_if_playing()
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        
        if file_path:
            self.current_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.current_image is not None:
                self.detected_image = None
                self.display_image(self.current_image)
                self.btn_detect_img.setEnabled(True)
                self.btn_save_img.setEnabled(False)
                self.statusBar.showMessage(f"图片已加载: {file_path}")
            else:
                QMessageBox.warning(self, "警告", "图片读取失败，请检查文件格式！")

    def on_conf_changed(self, value):
        self.confidence = value / 100.0
        self.conf_value_label.setText(f"{self.confidence:.2f}")
        self.statusBar.showMessage(f"conf阈值已调整为 {self.confidence:.2f}")

    def detect_image(self):
        if self.current_image is None or self.model is None:
            return

        self.statusBar.showMessage("正在检测中，请稍候...")
        QApplication.processEvents()

        results = self.model.predict(source=self.current_image, conf=self.confidence, verbose=False)
        self.detected_image = results[0].plot()
        
        self.display_image(self.detected_image)
        self.btn_save_img.setEnabled(True)
        self.statusBar.showMessage("图片检测完成！")

    def save_image(self):
        if self.detected_image is None:
            QMessageBox.information(self, "提示", "当前没有可保存的检测结果。")
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", "result.jpg", "JPEG Image (*.jpg);;PNG Image (*.png)", options=options)
        
        if file_path:
            ext = Path(file_path).suffix
            if not ext:
                ext = ".jpg"
                file_path += ext
            is_success, im_buf_arr = cv2.imencode(ext, self.detected_image)
            if is_success:
                im_buf_arr.tofile(file_path)
                self.statusBar.showMessage(f"结果已保存至: {file_path}")
                QMessageBox.information(self, "成功", "图片保存成功！")
            else:
                QMessageBox.critical(self, "错误", "图片保存失败！")

    # ── 视频功能 ──────────────────────────────────────────

    def upload_video(self):
        if self.camera_running:
            QMessageBox.information(self, "提示", "请先停止摄像头，再上传视频。")
            return
        self.stop_video_if_playing()
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Videos (*.mp4 *.avi *.mkv *.mov)", options=options)
        
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            if self.video_capture.isOpened():
                self.btn_play_vid.setEnabled(True)
                self.btn_detect_img.setEnabled(False)
                self.btn_save_img.setEnabled(False)
                
                ret, frame = self.video_capture.read()
                if ret:
                    self.display_image(frame)
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                self.statusBar.showMessage(f"视频已加载: {file_path}")
            else:
                QMessageBox.warning(self, "警告", "视频读取失败！")

    def toggle_video(self):
        if self.video_capture is None or not self.video_capture.isOpened():
            return
            
        if self.is_video_playing:
            self.video_timer.stop()
            self.btn_play_vid.setText("继续播放")
            self.is_video_playing = False
            self.statusBar.showMessage("视频已暂停")
        else:
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            interval = int(1000 / fps) if fps > 0 else 33
            self.video_timer.start(interval)
            self.btn_play_vid.setText("暂停视频")
            self.is_video_playing = True
            self.statusBar.showMessage("视频实时检测中...")

    def process_video_frame(self):
        if self.video_capture is None or self.model is None:
            return
            
        ret, frame = self.video_capture.read()
        if ret:
            results = self.model.predict(source=frame, conf=self.confidence, verbose=False)
            detected_frame = results[0].plot()
            self.detected_image = detected_frame
            self.display_image(detected_frame)
        else:
            self.stop_video_if_playing()
            self.statusBar.showMessage("视频播放结束")

    # ── 摄像头功能 ────────────────────────────────────────

    def start_camera(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "模型未加载，无法开启实时检测！")
            return
        if self.camera_running:
            return

        self.stop_video_if_playing()

        self.camera_capture = cv2.VideoCapture(0)
        if not self.camera_capture.isOpened():
            self.camera_capture = None
            QMessageBox.critical(self, "错误", "无法打开摄像头，请检查设备或权限设置。")
            return

        self.camera_running = True
        self.camera_timer.start(30)

        self.btn_start_camera.setEnabled(False)
        self.btn_stop_camera.setEnabled(True)
        self.btn_upload_img.setEnabled(False)
        self.btn_upload_vid.setEnabled(False)
        self.btn_detect_img.setEnabled(False)
        self.btn_save_img.setEnabled(False)
        self.statusBar.showMessage("摄像头已开启，正在实时检测...")

    def process_camera_frame(self):
        if not self.camera_running or self.camera_capture is None:
            return

        ret, frame = self.camera_capture.read()
        if not ret:
            self.statusBar.showMessage("摄像头读取失败，已停止。")
            self.stop_camera()
            return

        results = self.model.predict(source=frame, conf=self.confidence, verbose=False)
        detected_frame = results[0].plot()
        self.detected_image = detected_frame
        self.display_image(detected_frame)

    def stop_camera(self):
        self.stop_camera_if_running()
        self.btn_detect_img.setEnabled(self.current_image is not None)
        self.statusBar.showMessage("摄像头已停止。")

    # ── 图像显示 ──────────────────────────────────────────

    def display_image(self, img_array):
        rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.stop_video_if_playing()
        self.stop_camera_if_running()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    window = ObstacleDetectionGUI()
    window.show()
    sys.exit(app.exec_())
