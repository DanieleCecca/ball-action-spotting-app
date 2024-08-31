import cv2
import torch
import numpy as np
from io import BytesIO
import tempfile

class VideoPrediction:
    def __init__(self, video_bytes: BytesIO, model, num_frames=15, frames_per_stack=3):
        self.video_bytes = video_bytes
        self.num_frames = num_frames#num frame for each segment passed through the network
        self.frames_per_stack = frames_per_stack
        self.video = None
        self.fps = 0
        self.width = 0
        self.height = 0
        self.model = model
        self.labels= {8: 'Pass', 
                      2: 'Drive', 
                      5: 'Header',
                      6: 'High Pass',
                      7: 'Out',
                      11: 'Throw In',
                      10: 'Shot',
                      0: 'Ball Player Block',
                      1: 'Cross',
                      9: 'Player Successful Tackle',
                      3: 'Free Kick',
                      4: 'Goal'}
        self.length=0

    def load_video(self):
        # Write the video bytes to a temporary file otherwise i cannot use cv2
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(self.video_bytes.read())
        temp_file.flush()
        temp_file.seek(0)
        
        self.video = cv2.VideoCapture(temp_file.name)
        
        if not self.video.isOpened():
            raise ValueError("Failed to load video")

        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.length=int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def predict_video(self, prediction_callback):
        frames = []
        frame_count = 0
        
        # Read the video
        n_pred=0
        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = torch.tensor(gray_frame).float() / 255.0
            frames.append(gray_frame)
            frame_count += 1

            # Do prediciton every 15 frames
            if len(frames) == self.num_frames:
                stacked_frames = self.stack_frames(frames)
                with torch.no_grad():
                    output = self.model(stacked_frames.unsqueeze(0))
                    _,output=torch.max(output.data, 1)
                    output=self.labels[output.item()]
                    n_pred+=1
                
                # Call the callback function with the prediction
                prediction_callback(frame_count, output,n_pred)
                
                # Slide the window by 15frames
                frames = frames[self.num_frames:]

    def stack_frames(self, frames):
        stacked_frames = []
        for i in range(0, len(frames) - self.frames_per_stack + 1, self.frames_per_stack):
            stack = frames[i:i + self.frames_per_stack]
            stacked_frames.append(torch.stack(stack, dim=0))
        return torch.stack(stacked_frames, dim=0)

    def get_video_info(self):
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "length":self.length,
            "num_frames": self.num_frames,
            "frames_per_stack": self.frames_per_stack
        }
    

    def __del__(self):
        if self.video is not None:
            self.video.release()
    