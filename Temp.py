import sys
import socket
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QGridLayout, QTextEdit
from PyQt5.QtCore import Qt

class GestureSender(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UDP Gesture Sender (Debugger)")
        self.setGeometry(600, 100, 300, 400)

        # === UDP 配置 ===
        self.udp_ip = "127.0.0.1"
        self.udp_port = 5005
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # === UI 初始化 ===
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()

        # 状态标签
        self.info_label = QLabel(f"Target: {self.udp_ip}:{self.udp_port}", self)
        self.info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.info_label)

        # 按钮区域
        btn_layout = QGridLayout()
        gestures = [
            ('Fist', '#3176a0'), 
            ('OK', '#be8a00'), 
            ('Thumb', '#bd4103'), 
            ('Open', '#347145'), 
            ('Index', '#4d004d'), 
            ('Rest', '#888888')
        ]

        row, col = 0, 0
        for name, color in gestures:
            btn = QPushButton(name)
            btn.setMinimumHeight(50)
            # 设置按钮颜色和样式
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    border-radius: 10px;
                }}
                QPushButton:pressed {{
                    background-color: #333333;
                }}
            """)
            # 使用闭包绑定参数
            btn.clicked.connect(lambda checked, n=name.lower(): self.send_gesture(n))
            btn_layout.addWidget(btn, row, col)
            
            col += 1
            if col > 1:
                col = 0
                row += 1

        main_layout.addLayout(btn_layout)

        # 日志区域
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        main_layout.addWidget(QLabel("Sent Log:"))
        main_layout.addWidget(self.log_area)

        central_widget.setLayout(main_layout)

    def send_gesture(self, gesture_name):
        try:
            msg = gesture_name.encode('utf-8')
            self.sock.sendto(msg, (self.udp_ip, self.udp_port))
            self.log_area.append(f">> Sent: {gesture_name}")
            # 滚动到底部
            self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())
        except Exception as e:
            self.log_area.append(f"Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureSender()
    window.show()
    sys.exit(app.exec_())
