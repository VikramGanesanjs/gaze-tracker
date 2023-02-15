import cv2
import mediapipe as mp
import torch
from net import GazePredictor
import torchvision.transforms as transforms
import numpy as np
import pyautogui as pg
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class Gazer:
    def __init__(self):
        self.screen_width, self.screen_height = pg.size()

        self.net = GazePredictor(False)
        self.net.load_state_dict(torch.load("../saved-models/gaze_tracker_saved_2_6_BEST.pth")['model_state_dict'])
        self.net.eval()

        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.other_transform = transforms.Compose([
                transforms.CenterCrop((496, 289)),
                transforms.Resize((444, 250)),
                transforms.CenterCrop(224), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        self.pil_transform = transforms.ToPILImage()
        self.mp_face_mesh = mp.solutions.face_mesh


    def get_face_landmarks(self, image):
        landmarks = []
        for i in range(image.shape[0]):
            np_pil_image = np.array(self.pil_transform(image[i]))
            with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
                _, img_bytes = cv2.imencode('.jpg', np_pil_image)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                landmarks_i = []
                if not results.multi_face_landmarks:
                    {landmarks_i.append([0, 0, 0]) for i in range(478)}
                    landmarks.append(landmarks_i)
                    continue
                for face_landmarks in results.multi_face_landmarks:
                    for data_point in face_landmarks.landmark:
                        landmarks_i.append([data_point.x, data_point.y, data_point.z])
                landmarks.append(landmarks_i)
        landmarks_tensor = torch.tensor(landmarks)
        shape = landmarks_tensor.shape
        return landmarks_tensor.view(shape[0], shape[2], shape[1])
    
    def move_mouse(self, x, y, duration):
        start_x, start_y = pg.position()
        duration = duration
        pg.move(x - start_x, y - start_y, duration)

    def webcam_input_face_crop(self):
        # For webcam input:
        cap = cv2.VideoCapture(0)
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")

                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)
                height, width, channels = image.shape

                # Draw the face detection annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.flip(image, 1)
                cropped_img = image
                if results.detections:
                    for detection in results.detections:
                        x = int(detection.location_data.relative_bounding_box.xmin * width)
                        y = int(detection.location_data.relative_bounding_box.ymin * height)
                        w = max(int(detection.location_data.relative_bounding_box.width * width), 224)
                        h = max(int(detection.location_data.relative_bounding_box.height * height), 224)
                        cropped_img = cv2.getRectSubPix(image, (w, h), (x + w/2, y + h/2))
                        mp_drawing.draw_detection(image, detection)
                    tensor_img = self.eval_transform(self.pil_transform(cropped_img))
                    x_val, y_val = self.get_gaze_frame(tensor_img)
                    self.move_mouse(x_val, y_val, 0.01)
                
                cv2.imshow('GazeTracker', image)
                if cv2.waitKey(5) == ord("q"):
                    break
        cap.release()

    def without_video_cap(self, image):
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.flip(image, 1)
            tensor_img = self.other_transform(self.pil_transform(image))
            x_val, y_val = self.get_gaze_frame(tensor_img)
            return x_val, y_val

    def webcam_input_no_face_crop(self):
        # For webcam input:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")

                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.flip(image, 1)
            tensor_img = self.other_transform(self.pil_transform(image))
            x_val, y_val = self.get_gaze_frame(tensor_img)
            self.move_mouse(x_val, y_val, 0.01)
            
            cv2.imshow('GazeTracker', image)
            if cv2.waitKey(5) == ord("q"):
                break
        cap.release()


    def get_gaze_frame(self, tensor_img):
        tensor_img = tensor_img.view(1, *tensor_img.shape)
        y_pred = self.net(tensor_img, self.get_face_landmarks(tensor_img).float())
        y_pred_denorm = y_pred * torch.tensor((self.screen_width, self.screen_height))
        x_val, y_val = y_pred_denorm[0][0].item(), y_pred_denorm[0][1].item()
        return x_val, y_val