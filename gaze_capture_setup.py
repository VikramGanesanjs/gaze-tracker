import torch.utils.data as data
import pandas as pd
import json, os, shutil
class GazeCapture(data.Dataset):
    def __init__(self, root_dir, save_dir):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.device_data = pd.read_csv("../apple_device_data.csv")
        self.process_gaze_capture()
        print(self.get_max_and_min())

    def process_folder(self, dir_name):
        frames_path = os.path.join(dir_name, "frames")
        json_path = os.path.join(dir_name, "dotInfo.json")
        screen_info_path = os.path.join(dir_name, "screen.json")
        f = open(json_path)
        data = json.load(f)
        x_pts = data['XPts']
        y_pts = data['YPts']
        f.close()
        f = open(screen_info_path)
        info = json.load(f)
        f.close()
        f = open(os.path.join(dir_name, "info.json"))
        device = json.load(f)["DeviceName"]
        device_info = self.device_data[self.device_data["DeviceName"] == device]
        h_mm, w_mm, = 1, 1
        if not device_info.empty:
            h_mm, w_mm = device_info["DeviceScreenHeightMm"].item(), device_info["DeviceScreenWidthMm"].item()
        heights, widths, orientations = info['H'], info["W"], info["Orientation"]
        for i, image_path in enumerate(os.listdir(frames_path)):
            height, width, orientation = heights[i], widths[i], orientations[i]
            X_cm = x_pts[i] / width * w_mm / 10
            Y_cm = y_pts[i] / height * h_mm / 10
            label = str(X_cm) + "_" + str(Y_cm)
            directory = os.path.join(self.save_dir, label)
            if not os.path.exists(directory):
                os.makedirs(directory)
            shutil.copy(os.path.join(frames_path, image_path), os.path.join(directory, image_path))
        
    
    def process_gaze_capture(self):
        for subject in os.listdir(self.root_dir):
            if subject[0] != ".":
                self.process_folder(os.path.join(self.root_dir, subject))
    
    def get_max_and_min(self):
        xmin, xmax, ymin, ymax = 0, 0, 0, 0
        for label in os.listdir(self.save_dir):
            label_arr = label.split("_")
            x, y = float(label_arr[0]), float(label_arr[1])
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
        return xmin, xmax, ymin, ymax


gaze_capture = GazeCapture("../raw-gaze-capture", "../gaze-capture")