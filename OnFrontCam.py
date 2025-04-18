#works with front camera of laptop
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from ultralytics import YOLO
from playsound import playsound
import threading
import time
from datetime import datetime
import logging
import os
from pathlib import Path


class BankSecurityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("South India Bank - Security System")
        self.setFixedSize(800, 600)

        # Set window background color to a professional banking blue
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor('#f5f5f5'))
        self.setPalette(palette)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # Add bank logo
        logo_label = QLabel()
        logo_pixmap = QPixmap('assets/bank_logo.png')  # Ensure this path is correct
        scaled_pixmap = logo_pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(scaled_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # Add bank name with professional styling
        bank_name = QLabel("SOUTH INDIA BANK")
        bank_name.setFont(QFont('Arial', 24, QFont.Bold))
        bank_name.setStyleSheet("color: #00205b;")  # Dark blue color
        bank_name.setAlignment(Qt.AlignCenter)
        layout.addWidget(bank_name)

        # Add security system tagline
        tagline = QLabel("Advanced Vault Security System")
        tagline.setFont(QFont('Arial', 14))
        tagline.setStyleSheet("color: #666666;")
        tagline.setAlignment(Qt.AlignCenter)
        layout.addWidget(tagline)

        # Add enter button with professional styling
        self.enter_button = QPushButton("Enter Vault Security System")
        self.enter_button.setFont(QFont('Arial', 12))
        self.enter_button.setStyleSheet("""
            QPushButton {
                background-color: #00205b;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #003080;
            }
            QPushButton:pressed {
                background-color: #001540;
            }
        """)
        self.enter_button.setFixedWidth(300)
        self.enter_button.clicked.connect(self.launch_security_system)
        layout.addWidget(self.enter_button)

        # Add copyright notice
        copyright_label = QLabel(f"© {datetime.now().year} South India Bank. All rights reserved.")
        copyright_label.setFont(QFont('Arial', 8))
        copyright_label.setStyleSheet("color: #999999;")
        copyright_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(copyright_label)

        # Initialize vault system as None
        self.vault_system = None

    def launch_security_system(self):
        """Launch the vault security system"""
        self.hide()  # Hide the main window
        self.vault_system = VaultSecuritySystem(parent_app=self)
        try:
            self.vault_system.run()
        except SystemExit:
            self.close()  # Close the main window if system exit is triggered
        except Exception as e:
            print(f"Error in security system: {e}")
            self.show()  # Show the main window again on error

    def closeEvent(self, event):
        """Handle application closure"""
        if self.vault_system:
            self.vault_system.cleanup()
        event.accept()


class VaultSecuritySystem:
    def __init__(self, parent_app=None):
        self.parent_app = parent_app  # Reference to the main application window

        # Configuration
        self.CONFIG = {
            'required_people': 2,
            'confidence_threshold': 0.6,
            'alert_cooldown': 3,
            'log_directory': 'vault_logs',
            'alert_sounds': {
                'violation': 'C:/Users/DELL/Desktop/south india bank/assets/alarm.wav',
                'access_granted': 'C:/Users/DELL/Desktop/south india bank/assets/access-granted.mp3v'
            },
            'recording': {
                'enabled': True,
                'violation_clip_duration': 10,  # seconds
                'output_directory': 'violation_clips'
            }
        }

        # Initialize logging
        self._setup_logging()

        # Initialize the video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Load YOLO model
        self.model = YOLO('C:/Users/DELL/Desktop/south india bank/yolo11l.pt')  # Ensure this path is correct

        # State management
        self.alert_active = False
        self.last_alert_time = 0
        self.current_violation_start = None
        self.recording = False

        # Initialize violation recorder
        self.violation_writer = None

        # Create required directories
        self._create_directories()

        # Load custom UI elements
        self._load_ui_elements()

    def _setup_logging(self):
        """Configure logging system"""
        try:
            # Ensure the log directory exists
            log_dir = Path(self.CONFIG['log_directory'])
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create log file path
            log_file = log_dir / f'vault_security_{datetime.now().strftime("%Y%m%d")}.log'

            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(str(log_file)),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger('VaultSecurity')
            self.logger.info('Logging system initialized successfully')
        except Exception as e:
            print(f"Error setting up logging: {e}")
            # Fallback to basic logging if file logging fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )
            self.logger = logging.getLogger('VaultSecurity')

    def _create_directories(self):
        """Create necessary directories for logs and recordings"""
        os.makedirs(self.CONFIG['log_directory'], exist_ok=True)
        os.makedirs(self.CONFIG['recording']['output_directory'], exist_ok=True)

    def _load_ui_elements(self):
        """Load UI overlay elements"""
        try:
            # Create custom overlay elements
            self.overlay_bg = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.rectangle(self.overlay_bg, (0, 0), (400, 100), (20, 20, 20), -1)

            # Initialize status icons as None first
            self.status_icons = {
                'ok': None,
                'warning': None,
                'recording': None
            }

            # Try to load icons if they exist, otherwise continue without them
            icon_paths = {
                'ok': 'C:/Users/DELL/Desktop/south india bank/assets/checkmark.png',
                'warning': 'C:/Users/DELL/Desktop/south india bank/assets/warning.png',
                'recording': 'C:/Users/DELL/Desktop/south india bank/assets/recording.png'
            }

            for key, path in icon_paths.items():
                try:
                    if os.path.exists(path):
                        self.status_icons[key] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                except Exception as e:
                    self.logger.warning(f"Could not load icon {path}: {e}")
        except Exception as e:
            self.logger.error(f"Error in UI element initialization: {e}")

    def draw_bounding_boxes(self, frame, people_boxes, people_count):
        """Draw color-coded bounding boxes around detected persons"""
        # Determine box color based on number of people
        if people_count == self.CONFIG['required_people']:
            box_color = (0, 255, 0)  # Green for exactly 2 people (access granted)
        else:
            box_color = (0, 0, 255)  # Red for violations (1 person or >2 people)

        # Draw boxes around each detected person
        for box in people_boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Add confidence score with background
            conf = float(box.conf[0])
            conf_text = f'{conf:.2f}'

            # Get text size for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(
                conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Draw background rectangle for text
            cv2.rectangle(frame,
                          (x1, y1 - text_height - 8),
                          (x1 + text_width + 5, y1),
                          box_color, -1)

            # Draw confidence text
            cv2.putText(frame, conf_text,
                        (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

    def draw_professional_overlay(self, frame, people_count):
        """Draw professional UI overlay with vault security information"""
        # Add semi-transparent dark overlay at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Add modern status display
        status_text = f"VAULT SECURITY STATUS: "
        if people_count == self.CONFIG['required_people']:
            status = "ACCESS GRANTED"
            color = (0, 255, 0)
            icon = self.status_icons['ok']
        else:
            status = "ACCESS DENIED"
            color = (0, 0, 255)
            icon = self.status_icons['warning']

        # Draw status text with professional styling and better positioning
        cv2.putText(frame, status_text, (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (200, 200, 200), 2)

        # Move access status to bottom left
        cv2.putText(frame, status, (50, 90),
                    cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

        # Add time and date (moved to right side)
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        cv2.putText(frame, f"TIME: {current_time}", (frame.shape[1] - 350, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"DATE: {current_date}", (frame.shape[1] - 350, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Add person count with icon (moved to center)
        count_text = f"PERSONS DETECTED: {people_count}/{self.CONFIG['required_people']}"
        text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2  # Center the text
        cv2.putText(frame, count_text, (text_x, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (200, 200, 200), 2)

        # Add recording indicator if active
        if self.recording:
            cv2.circle(frame, (frame.shape[1] - 50, 50), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return status != "ACCESS GRANTED"

    def play_sound(self, sound_type):
        """Play appropriate sound based on the situation"""
        try:
            sound_file = self.CONFIG['alert_sounds'][sound_type]
            threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()
        except Exception as e:
            self.logger.error(f"Error playing sound: {e}")
            print('\a')  # Fallback to system beep

    def detect_people(self, frame):
        """Enhanced people detection with tracking"""
        results = self.model(frame, conf=self.CONFIG['confidence_threshold'])

        people_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 0:  # Person class
                    people_boxes.append(box)

        return len(people_boxes), people_boxes

    def start_violation_recording(self, frame):
        """Start recording violation clip"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.CONFIG['recording']['output_directory'],
                f'violation_{timestamp}.mp4'
            )
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.violation_writer = cv2.VideoWriter(
                output_path, fourcc, 20.0,
                (frame.shape[1], frame.shape[0])
            )
            self.recording = True
            self.current_violation_start = time.time()
            self.logger.info(f"Started recording violation clip: {output_path}")

    def stop_violation_recording(self):
        """Stop recording violation clip"""
        if self.recording:
            self.violation_writer.release()
            self.recording = False
            self.current_violation_start = None
            self.logger.info("Stopped recording violation clip")

    def run(self):
        """Main monitoring loop with enhanced features and RTSP connection handling"""
        self.logger.info("Starting Vault Security System")
        consecutive_failures = 0
        max_failures = 5

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    consecutive_failures += 1
                    self.logger.warning(f"Failed to read frame (failure {consecutive_failures})")

                    if consecutive_failures >= max_failures:
                        self.logger.error("Multiple frame read failures, attempting to reconnect...")
                        try:
                            self._reconnect_camera()
                            consecutive_failures = 0
                        except ConnectionError:
                            self.logger.error("Failed to reconnect to camera")
                            break

                    time.sleep(0.5)
                    continue

                consecutive_failures = 0

                # Detect people
                people_count, people_boxes = self.detect_people(frame)
                self.draw_bounding_boxes(frame, people_boxes, people_count)
                violation_detected = self.draw_professional_overlay(frame, people_count)

                if violation_detected:
                    current_time = time.time()
                    if current_time - self.last_alert_time >= self.CONFIG['alert_cooldown']:
                        self.play_sound('violation')
                        self.last_alert_time = current_time
                        self.logger.warning(f"Security violation detected: {people_count} people present")

                    if self.CONFIG['recording']['enabled']:
                        self.start_violation_recording(frame)
                        if self.recording:
                            self.violation_writer.write(frame)
                else:
                    if self.recording:
                        self.stop_violation_recording()

                cv2.imshow('Bank Vault Security Monitor', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.cleanup()
                    if self.parent_app:
                        self.parent_app.close()  # Close the main application
                    sys.exit()  # Exit the entire application
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'screenshot_{timestamp}.jpg', frame)

        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Shutting down Vault Security System")
        if self.recording:
            self.stop_violation_recording()
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = BankSecurityApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()