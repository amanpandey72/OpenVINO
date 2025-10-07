import cv2
import numpy as np
import time
import os
from datetime import datetime
from openvino.runtime import Core

# ----------------- Config -----------------
model_xml = r"C:\Users\amanp\OneDrive\Desktop\openVINO\ir_output\yolov4-tiny-custom_1000.xml"
conf_threshold = 0.25
nms_threshold = 0.4
input_size = 416
class_names = ["without_mask", "with_mask"]

# ----------------- Output Folder -----------------
output_folder = "openvino_detections"
os.makedirs(output_folder, exist_ok=True)  # creates folder if not present

# ----------------- Load OpenVINO model -----------------
ie = Core()
model = ie.read_model(model=model_xml)
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs
print("OpenVINO Model loaded!")
print("Inputs:", input_layer.shape)
for i, out in enumerate(output_layers):
    print(f"Output {i}: {out.shape}")

# ----------------- Preprocess -----------------
def preprocess(frame):
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    return img

# ----------------- Postprocess -----------------
def postprocess(outputs, frame_shape):
    outputs = [np.array(o) for o in outputs]

    boxes_tensor, scores_tensor = outputs
    boxes_tensor = boxes_tensor.squeeze()
    scores_tensor = scores_tensor.squeeze()

    boxes, confidences, class_ids = [], [], []
    h, w = frame_shape

    for i in range(boxes_tensor.shape[0]):
        box = boxes_tensor[i]
        class_score = scores_tensor[i]
        class_id = int(np.argmax(class_score))
        confidence = float(class_score[class_id])

        if confidence < conf_threshold:
            continue

        # YOLOv4-tiny IR gives normalized coords (0–1)
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)

        boxes.append([x1, y1, x2, y2])
        confidences.append(confidence)
        class_ids.append(class_id)

    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = [boxes[i] for i in indices]
            confidences = [confidences[i] for i in indices]
            class_ids = [class_ids[i] for i in indices]
        else:
            boxes, confidences, class_ids = [], [], []

    return boxes, confidences, class_ids

# ----------------- Draw boxes -----------------
def draw_boxes(frame, boxes, confidences, class_ids):
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        label = f"{class_names[cls_id]}: {conf:.2f}"
        color = (0, 255, 0) if cls_id == 1 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ----------------- Webcam Loop -----------------
cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_input = preprocess(frame)

    results = compiled_model([img_input])
    outputs = [results[out] for out in output_layers]

    boxes, confidences, class_ids = postprocess(outputs, frame.shape[:2])
    frame = draw_boxes(frame, boxes, confidences, class_ids)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YOLOv4-tiny OpenVINO Webcam", frame)

    # Save detections if any
    if boxes:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_folder, f"frame_{timestamp}.jpg")
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 50])  
        print(f"Saved: {filename}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
