import numpy as np
import cv2
import models
from moviepy.editor import *
from tqdm import tqdm

model = models.behavior_cloning()
model.load_weights('model.h5')

def load_image(image_filename):
    image = cv2.imread(image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (240,120))
    return image

def compute_overlay(image):
    full_angle = float(model.predict(image[None,:,:,:], batch_size=1))
    result = np.zeros((60,30), np.float32)
    for i in range(59):
        for j in range(29):
            temp = np.copy(image)
            start_x = 4 * i 
            end_x = 4 * (i + 1)
            start_y = 4 * j
            end_y = 4 * (j + 1)
            temp[start_x:end_x,start_y:end_y,:] = 0.0
            angle = float(model.predict(temp[None,:,:,:], batch_size=1))
            result[i][j] = abs(angle - full_angle)
    return result

def normalize_overlay(x, r):
    x = np.array(x, np.float32)
    array_min = np.amin(x)
    array_max = np.amax(x)
    range_min = r[0]
    range_max = r[1]
    normed = (x - array_min) / (array_max - array_min)
    scaled = normed * (range_max - range_min) + range_min
    return scaled.astype(np.uint8)

def process_frame(image):
    overlay = compute_overlay(image)
    overlay = normalize_overlay(overlay, (255,0)) 
    overlay = cv2.resize(overlay, (240,120))
    overlay = cv2.GaussianBlur(overlay, (3,3), 0)
    _, overlay = cv2.threshold(overlay, 110, 255, cv2.THRESH_TOZERO)
    overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
    image = cv2.addWeighted(image, 1.0, overlay, 0.4, 0.0)
    return image

def create_video(csv_file, output_filename):
    frames = []
    csv_data = np.genfromtxt(csv_file, delimiter=',', dtype=None)
    total_frames = len(csv_data)
    for i, row in tqdm(enumerate(csv_data), total=total_frames):
        image_filename = 'training-data/IMG/' + row[0].decode('utf-8').strip()
        image = load_image(image_filename)
        frame = process_frame(image)
        frames.append(frame)
    video = ImageSequenceClip(frames, fps=15)
    video.write_videofile(output_filename, audio=False)

create_video('training-data/driving_log.csv', 'output.mp4')
