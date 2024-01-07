import numpy as np
import cv2
from PIL import Image


def load_video(video_path, size=None, mode="RGB"):
    video_frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = list(range(frame_count))
    cur_index = 0
    k = 0
    while cap.isOpened():
        ret = cap.grab()
        if not ret:
            break
        if k in frame_index:
            _, frame = cap.retrieve()
            if size != None:
                frame = cv2.resize(frame, size)
            if mode == "RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
            cur_index += 1
        k += 1
    cap.release()
    return video_frames


def load_image(image_path, mode="RGB"):
    return np.array(Image.open(image_path).convert("RGB"))
