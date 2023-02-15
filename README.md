# Gaze Tracker Science Fair Project

## A model that is able to predict a user's gaze with less than 2 centimeters of error

## Using PyTorch, Mediapipe and OpenCV

# File Structure

#### train.py - Main Training loop for the model
#### net.py - Model Architecture
#### demo.py - Quick demo app made with Tkinter
#### gazer.py - application of the model to real-time webcam stream
#### data.py - Preprocessing of tablet-gaze dataset
#### gaze_capture_setup.py - Preprocessing of gaze-capture dataset
#### vgg.py - VGG-16 architecture that can be used for fine tuning
#### vgg_normalized.pth - PyTorch Model state dict for pretrained VGG-16 model that can be used for fine tuning
#### All other files should be disregarded as they are just demos or test files

# Slideshow Presentation
https://docs.google.com/presentation/d/e/2PACX-1vT31BkkXtEETUB4jZlFkJaHxrtIWq4od2WNtv4lVFZEgALAo8zvMprMnpUUFtvI3A/pub?start=false&loop=false&delayms=3000

# Project Video
Not currently finished


# Instructions to Run on Own Machine

1. Clone the repository
2. It is recommended to download the GazeCapture dataset for best results.
3. Once this is downloaded, run gaze_capture_setup.py, replacing the root_dir and the save_dir in the file with wherever you want to save the dataset (outside the repo)
4. Create a directory called saved models outside of the cloned repo
5. If the machine is not an Apple Silicon Macbook, you will need to delete every instance of mps_device in train.py and train the model on a cpu or cuda
6. 

