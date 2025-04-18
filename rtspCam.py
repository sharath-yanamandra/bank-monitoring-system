# on rtsp feed on cctv EZVIZ
import cv2
import numpy as np
from ultralytics import YOLO
from playsound import playsound
import threading
import time
from datetime import datetime
import logging
import json
import os
from pathlib import Path

class VaultSecuritySystem:
    def __init__(self):
        # Configuration
        self.CONFIG = {
            'required_people': 2,
            'confidence_threshold': 0.6,
            'alert_cooldown': 3,
            'log_directory': 'vault_logs',
            'alert_sounds': {
                'violation': 'assets/alarm.wav',
                'access_granted': 'assets/access_granted.wav'
            },
            'recording': {
                'enabled': True,
                'violation_clip_duration': 10,  # seconds
                'output_directory': 'violation_clips'
            },
            'camera': {
                'rtsp_url': 'rtsp://admin:OUESEH@192.168.29.112:554/h264/ch1/main/av_stream',
                'reconnect_attempts': 3,
                'reconnect_delay': 2  # seconds between reconnection attempts
            }
        }
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize the video capture with RTSP stream
        self._initialize_camera()
        
        # Load YOLO model
        self.model = YOLO('C:/Users/DELL/Desktop/south india bank/yolo11m.pt')
        
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
            
            # Try to load icons if they exist
            icon_paths = {
                'ok': 'assets/checkmark.png',
                'warning': 'assets/warning.png',
                'recording': 'assets/recording.png'
            }
            
            for key, path in icon_paths.items():
                try:
                    if os.path.exists(path):
                        self.status_icons[key] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                except Exception as e:
                    self.logger.warning(f"Could not load icon {path}: {e}")
        except Exception as e:
            self.logger.error(f"Error in UI element initialization: {e}")

    def _initialize_camera(self):
        """Initialize camera with RTSP stream and handle connection attempts"""
        self.logger.info("Initializing RTSP camera connection...")
        
        for attempt in range(self.CONFIG['camera']['reconnect_attempts']):
            self.cap = cv2.VideoCapture(self.CONFIG['camera']['rtsp_url'])
            
            if self.cap.isOpened():
                self.logger.info("Successfully connected to RTSP stream")
                # Set buffer size to minimize latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return
            
            self.logger.warning(f"Failed to connect to RTSP stream (attempt {attempt + 1})")
            time.sleep(self.CONFIG['camera']['reconnect_delay'])
        
        self.logger.error("Failed to connect to RTSP stream after all attempts")
        raise ConnectionError("Could not connect to RTSP stream")

    def _reconnect_camera(self):
        """Attempt to reconnect to the RTSP stream if connection is lost"""
        self.logger.warning("Attempting to reconnect to RTSP stream...")
        self.cap.release()
        self._initialize_camera()

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

        # Draw status text with professional styling
        cv2.putText(frame, status_text, (50, 50), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, (200, 200, 200), 2)
        
        cv2.putText(frame, status, (50, 90),
                   cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

        # Add time and date
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        cv2.putText(frame, f"TIME: {current_time}", (frame.shape[1] - 350, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"DATE: {current_date}", (frame.shape[1] - 350, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Add person count
        count_text = f"PERSONS DETECTED: {people_count}/{self.CONFIG['required_people']}"
        text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, count_text, (text_x, 50),
                   cv2.FONT_HERSHEY_DUPLEX, 1, (200, 200, 200), 2)

        # Add recording indicator if active
        if self.recording:
            cv2.circle(frame, (frame.shape[1]-50, 50), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1]-100, 50),
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
        max_failures = 5  # Maximum number of consecutive frame read failures before reconnecting
        
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
                    
                    time.sleep(0.5)  # Brief pause before retry
                    continue
                
                consecutive_failures = 0  # Reset failure counter on successful frame read

                # Detect people
                people_count, people_boxes = self.detect_people(frame)
                
                # Draw bounding boxes first (so they appear behind the overlay)
                self.draw_bounding_boxes(frame, people_boxes, people_count)

                # Draw professional UI
                violation_detected = self.draw_professional_overlay(frame, people_count)

                # Handle violations and recordings
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

                # Display the frame
                cv2.imshow('Bank Vault Security Monitor', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Screenshot
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
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    vault_system = VaultSecuritySystem()
    vault_system.run()