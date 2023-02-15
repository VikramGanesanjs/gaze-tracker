import scipy.io
import torchvision.io as io
import torchvision
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import json
"""
Start Times is an array of length 51, 1 for each subject. In each of these arrays is another array of length 4. 
This represents the Trial ID. These arrays contain another of length four, for the posture ID.

A specific start time can bee accessed using start_times["startTime"][trial_id][0][0]
"""

"""
Gaze points is an array of length 4, each element is another array. The first array contains the x coordinates of the points,
the next contains the y coordinates. The next two arrays contain the width in cm and the height in cm from the top corner of the screen.     

"""
class TabletGaze(data.Dataset):
    def __init__(self, root_dir, data_dir, label_dir, transform=None, pose=2):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.images = []
        self.labels = []
        self.video_stream = "stream"
        self.transform = transform
        self.gaze_pts = scipy.io.loadmat(label_dir + "/gazePts.mat")
        self.start_times = scipy.io.loadmat(label_dir + "/startTime.mat")
        self.start_time_data = self.start_times["startTime"]
        self.gaze_pts_data = self.gaze_pts["gazePts"][0][0]
        self.pil_transform = transforms.ToPILImage()
        #self.process_all_data(pose)
        self.temporary_load_data()
        

    def process_video(self, video_path):
        ids = self.get_ids(video_path)
        print(ids)

        subjectId, postureId, trialId = int(ids[0]), int(ids[1]), int(ids[2])
        print(os.stat(video_path))
        video_frames, _, metadata = io.read_video(str(video_path), self.start_time_data[subjectId - 1][postureId - 1][trialId - 1], pts_unit="sec", output_format="TCHW")
        print(metadata)
        fps = metadata['video_fps']
        three_frames_seconds = 3 * fps
        one_half_seconds_frames = 1.5 * fps
        res = []  
        for i in range(35):
            idx = i * int(three_frames_seconds) + int(one_half_seconds_frames)
            for j in range(idx, idx+int(one_half_seconds_frames), 1):
                if j < len(video_frames):
                    label = (self.gaze_pts_data[2][0][i], self.gaze_pts_data[3][0][i])
                    print(label)
                    res.append([video_frames[j], label])
                    directory = "./images/{}_{}".format(label[0], label[1])
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(16,9)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(self.pil_transform(video_frames[j]), aspect="auto")
                    fig.savefig("./images/{}_{}/{}_{}_{}_{}_{}".format(label[0], label[1], ids[0], ids[1], ids[2], i, j), dpi=80)
                    plt.close()

        return res
    def add_videos_of_pose_and_subject(self, pose_index, subject):
        path = os.path.join(self.data_dir, str(subject))
        for i in range(4):
                video_path = os.path.join(path, "{}_{}_{}.mp4".format(subject, pose_index, i+1))
                res = self.process_video(video_path)
                for elem in res:
                    self.images.append(elem[0])
                    self.labels.append(elem[1])


    def process_all_data(self, pose):
        subjects = os.listdir(self.data_dir)
        print(subjects)
        for subject in subjects:
            if subject[0] != ".":
                self.add_videos_of_pose_and_subject(pose, int(subject))

    def get_ids(self, video_path):
        file = video_path.split("/")[-1]
        file = file[:-4]    
        ids = file.split("_")
        return ids

    def temporary_load_data(self):
        
        res = self.process_video(os.path.join(self.data_dir, "19", "19_2_4.mp4"))
        for elem in res:
            self.images.append(elem[0])
            self.labels.append(elem[1])




    def temporary_load_data_2(self):
        res = self.process_video(os.path.join(self.data_dir, "40", "40_2_1.mp4"))
        for elem in res:
            self.images.append(elem[0])
            self.labels.append(elem[1])
        res = self.process_video(os.path.join(self.data_dir, "45", "45_2_1.mp4"))
        for elem in res:
            self.images.append(elem[0])
            self.labels.append(elem[1])

        
        
 
        
        
       
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform != None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)



def main():
    torchvision.set_video_backend('video_reader')
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((400, 711)),
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
    ])
    dataset = TabletGaze("./gaze-dataset", "./gaze-dataset/data", "./gaze-dataset/labels", transform=transform)

    dataloader = data.DataLoader(dataset, shuffle=True)
    image, label = next(iter(dataloader))
    image = image[0]
    print(label)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    gaze_capture = GazeCapture("../gaze-capture")

if __name__ == "__main__":
    print(mp.get_sharing_strategy())
    main()