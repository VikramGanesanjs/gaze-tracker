from net import GazePredictor
import torch.nn as nn
import torch, math
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F

#TODO: Add Arg Parser to allow for changing hyperparameters in the command line
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_iter', type=int, default=15000)
parser.add_argument('--test_size', type=int, default=1500)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--loss', type=str, choices=['mse', 'l1', 'cross_entropy', 'huber', 'smooth_l1'], default='mse')
parser.add_argument('--lr_decay_epochs', type=int, default=2500)
parser.add_argument('--lr_decay_gamma', type=float, default=0.1)

mps_device = torch.device("mps")

loss_dict = {
    'mse': nn.MSELoss(),
    'l1':nn.L1Loss(),
    'cross-entropy': nn.CrossEntropyLoss(),
    'smooth_l1': nn.SmoothL1Loss(),
    'huber': nn.HuberLoss(),
}
args = parser.parse_args()
net = GazePredictor().to(mps_device)
loss_fn = loss_dict[args.loss]
epochs = args.max_iter
lr = args.lr
batch_size = args.batch_size
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, gamma=args.lr_decay_gamma)

pil_transform = transforms.ToPILImage()

transform = transforms.Compose([
        transforms.CenterCrop((496, 289)),
        transforms.Resize((444, 250)),
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

# Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh



root_data_dir = "../images"
tablet_gaze_data = datasets.ImageFolder(root_data_dir, transform=transform)
dataloader = data.DataLoader(tablet_gaze_data, batch_size=batch_size, shuffle=True)
class_to_idx = tablet_gaze_data.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

def cosine_decay(optimizer, global_step, max_steps, initial_lr, final_lr=0.0):
    decay_step = (1.0 - (global_step / max_steps)) * math.pi
    decay = (final_lr + (initial_lr - final_lr) * (1 + math.cos(decay_step)) / 2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay




def convert_label(label_idxs):
    label_arr = []
    for label_idx in label_idxs:
        raw_label = idx_to_class[label_idx.item()].split("_")
        x, y = float(raw_label[0]), float(raw_label[1])
        label_i = (x, y)
        label_arr.append(label_i)
    # Normalize labels by dividing them by the max value
    return torch.tensor(label_arr) / torch.tensor((21.52, 13.49))

def get_face_landmarks(image):
    landmarks = []
    for i in range(image.shape[0]):
        np_pil_image = np.array(pil_transform(image[i]))
        with mp_face_mesh.FaceMesh(
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
    return landmarks_tensor

def train():
    epochs_arr = []
    losses = []
    try:
        for epoch in tqdm(range(epochs)):
            epochs_arr.append(epoch)
            image, label_idxs = next(iter(dataloader))
            label = convert_label(label_idxs).to(mps_device)
            landmarks = get_face_landmarks(image)
            landmarks = landmarks.view(batch_size, 3, 478).to(mps_device)
            y_pred = net(image.to(mps_device), landmarks.float())
            y_pred_denormalized = y_pred * torch.tensor((21.52, 13.49)).to(mps_device)
            label_denormalized = label * torch.tensor((21.52, 13.49)).to(mps_device)
            loss = loss_fn(y_pred, label)
            loss_denormalized = loss_fn(y_pred_denormalized, label_denormalized)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            print("Epoch:{} y_pred:{:.6f}, {:.6f} label: {:.6f}, {:.6f} loss: {:.6f}, denorm {:.6f}".format(epoch, y_pred[0][0].item(), y_pred[0][1].item(), label[0][0].item(), label[0][1].item(), loss, loss_denormalized))
    except KeyboardInterrupt:
        return epochs_arr, losses
    return epochs_arr, losses

def average(arr):
    sum = 0
    for el in arr:
        sum += el
    return sum / len(arr)
def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))

def avg_euclidean_distance(preds, labels):
    sum = 0
    for i,  j, in zip(preds, labels):
        sum += loss_fn(i, j).sqrt()
    return sum / len(preds)



def test():
    losses = []
    epochs_arr = []
    preds = []
    labels = []
    batch_size = 1
    try:
        for i in tqdm(range(int(args.test_size))):
            epochs_arr.append(i)
            test_dataloader = data.DataLoader(tablet_gaze_data, batch_size=batch_size, shuffle=True)
            image, label_idxs = next(iter(test_dataloader))
            label = convert_label(label_idxs).to(mps_device)
            landmarks = get_face_landmarks(image).to(mps_device, dtype=torch.float32)
            landmarks = landmarks.view(batch_size, 3, 478)
            y_pred = net(image.to(mps_device), landmarks)
            loss = loss_fn(y_pred, label)
            preds.append(y_pred * torch.tensor((21.52, 13.49)).to(mps_device))
            labels.append(label * torch.tensor((21.52, 13.49)).to(mps_device))
            
            losses.append(loss.item())
    except KeyboardInterrupt:
        return losses, epochs_arr, average(losses), avg_euclidean_distance(preds, labels).item()
    return losses, epochs_arr, average(losses), avg_euclidean_distance(preds, labels).item()

def save():
    torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, "../saved-models/gaze_tracker_saved_{}_{}.pth".format(epochs, batch_size))


def main():
    epochs_arr, losses = train()
    test_losses, test_epochs, error, aed = test()
    save()
    print("Error:", error)
    print("AED: ", aed)
    if(len(epochs_arr) == len(losses)):
        plt.plot(epochs_arr, losses)
        plt.show()
        plt.plot(test_epochs, test_losses)
        plt.show()
    else:
        epochs_arr.pop()
        plt.plot(epochs_arr, losses)
        plt.show()
        plt.plot(test_epochs, test_losses)
        plt.show() 
if __name__ == "__main__":
    main()