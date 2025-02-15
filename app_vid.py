import cv2 
import torch
import numpy as np
import math
from ultralytics import YOLO
import streamlit as st
import time
import supervision as sv

# Image function for the streamlit app
def image_app(image, st, conf):
    # Checking if gpus are available: cuda or mps
    if torch.cuda.is_available():
        device = torch.device('cuda')

    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    else:
        device = torch.device('cpu')

    # loading the model
    model=YOLO(f"weights/best.pt").to(device)

    # class names
    clsDict = model.names

    # result after running the model on an image
    result = model(image, conf=conf)[0]

    # finding the co-ordinates of the bounding box and convert to numpy array
    bbox_xyxys = np.array(result.boxes.xyxy.cpu(), dtype = 'int')

    # finding the confidence score
    confidences = result.boxes.conf.cpu()

    # labels = np.array(result.boxes.cls.cpu(), dtype='int')
    labels = result.boxes.cls.tolist()

    # Iterating over bounding boxes, confidences and class labels
    for (bbox_xyxy, conf, cls) in zip(bbox_xyxys, confidences, labels):
        (x1, y1, x2, y2) = bbox_xyxy
        class_name = clsDict[cls]
        label = f"{class_name} {conf:.03}"
        # text size for labels
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        # rectangle for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 225), 2) # bbox
        # rectangle for class labels
        cv2.rectangle(image, (x1, y1), c2, (150, 0, 0), -1, cv2.LINE_AA) # rec for class names
        # putting text on rectangles
        cv2.putText(image, label, (x1, y1-2), 0, 1, (0, 0, 225), 1, lineType = cv2.LINE_AA)
        
    st.subheader('Output Image')
    st.image(image, channels='BGR')

# Video function for the streamlit app
def vid_app(video_path, kpi1_text, kpi2_text, kpi3_text, stframe, conf):
    # Capturing the frames of the video
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_width = int(cap.get(3))
 
    frame_height = int(cap.get(4))

    if torch.cuda.is_available():
        device = torch.device('cuda')

    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    else:
        device = torch.device('cpu')

    # Loading the model
    model=YOLO(f"weights/best.pt").to(device)

    # the class names we have chosen
    SELECTED_CLASS_NAMES = ['Choco']
    CLASS_NAMES_DICT = model.names
    # class ids matching the class names we have chosen
    SELECTED_CLASS_IDS = [
        {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
        for class_name
        in SELECTED_CLASS_NAMES
]

    # create BYTETracker instance
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=3)

    byte_tracker.reset()


    # create instance of BoxAnnotator, LabelAnnotator
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.3, text_color=sv.Color.BLACK)

    prev_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, conf=conf)[0]
        confidences = results.boxes.conf.cpu()
        detections = sv.Detections.from_ultralytics(results)
        # only consider class id from selected_classes define above
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        # tracking detections
        detections = byte_tracker.update_with_detections(detections)
        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)

        stframe.image(annotated_frame, channels='BGR')
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        kpi1_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(fps)}</h1>", unsafe_allow_html=True)

        kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(width)}</h1>", unsafe_allow_html=True)

        kpi3_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(height)}</h1>", unsafe_allow_html=True)
