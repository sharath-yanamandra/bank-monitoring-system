# realtime people count with alert
---

# 👥 Real-Time People Counting with Alert System

This project implements a **real-time people counting** application using object detection models (e.g., YOLO) with additional logic to trigger alerts when the number of people exceeds a specified threshold.

---

## 📌 Objective

To automatically detect and count people in a video stream (e.g., from CCTV or RTSP feed) and issue alerts when the crowd exceeds safety limits—useful in monitoring public areas, entry points, and restricted zones.

---

## 💡 Features

- 🧠 Real-time detection and counting of people
- 🎯 Uses YOLOv8 for accurate and fast inference
- 🔊 Triggers alerts (e.g., print/log/audio) if people count crosses a predefined threshold
- 📹 Supports both RTSP streams and local webcam
- 🖼️ Visual overlay of bounding boxes, count, and status
- 🔄 Optimized with frame skipping for performance

---

## 🛠️ Requirements

Install the following packages before running the notebook:

```bash
pip install opencv-python-headless numpy ultralytics
```

---

## 🚀 How to Run

1. **Clone or download this repository**

2. **Open the notebook**

   ```
   realtime people count with alert.ipynb
   ```

3. **Update the configurations** at the top:
   - Path to your trained YOLO model
   - RTSP stream URL or webcam index
   - Threshold count for alert

4. **Run all cells** in the notebook to start detection.

---

## ⚙️ Configuration Parameters

| Parameter         | Description                                  |
|-------------------|----------------------------------------------|
| `model_path`      | Path to YOLOv8 `.pt` model file              |
| `rtsp_url` / `cam_index` | RTSP stream or local webcam index      |
| `alert_threshold` | Max number of people allowed before alert    |

---

## 📊 Output

- Live video window with:
  - Bounding boxes around detected people
  - Count display on the frame
  - Color-coded alerts if threshold is exceeded
- Optional logging of timestamps and counts

---

## 🔐 Applications

- Workplace safety compliance
- Entry gate monitoring
- Social distancing enforcement
- Crowd control at events

---

## 📌 Notes

- Ensure the YOLO model is trained to detect persons (COCO class `0` or custom).
- For production use, alerting logic can be extended to send emails, play sounds, or trigger alarms.

---

## 📃 License

This project is released under the MIT License. Feel free to modify and use it in your applications.
