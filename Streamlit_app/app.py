from io import BytesIO
import streamlit as st
from model import SlowFusionNetVLAD
from video_prediction import VideoPrediction
import torch



#SINGLETON Class
model = SlowFusionNetVLAD(dropout=0.4)
weights_path=r'trained_model.pth'
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))) 
model.eval()


st.title(' Ball Action Spotting App ⚽+🤖=📈')
st.markdown('''
### Overview
Ball Action Spotting is a cutting-edge feature designed to accurately identify the type of
ball-related actions in football matches. This feature is capable of detecting actions across 12 distinct
classes in our extensive dataset, each action marked by a precise timestamp.

### Key Features

- **Comprehensive Action Detection**: The app identifies 12 different classes of soccer ball actions.
### Soccer Ball Actions Detected

In 2024, our app's capabilities have been extended to detect the following 12 classes of soccer ball actions:

| **Action**                | **Action**                  | **Action**                 |
|---------------------------|-----------------------------|----------------------------|
| Pass                       | Header                      | Ball Player Block           |
| Drive                      | High Pass                   | Player Successful Tackle    |
| Out                        | Cross                       | Free Kick                   |
| Throw In                   | Shot                        | Goal                        |

### Enhanced Analysis
With the Ball Action Spotting app, users can gain deep insights into the dynamics of soccer games,
improving both performance analysis and strategic planning. Whether you are a coach, analyst, or 
enthusiast, our app provides the tools you need to understand and optimize soccer performance.'''
)

# Upload the file
with st.sidebar:
    uploaded_file = st.file_uploader("Upload file", type=["mp4","mkv"])


if uploaded_file is not None:
    video_bytes = BytesIO(uploaded_file.read())
     
    # Initialize VideoPrediction
    video_pred = VideoPrediction(video_bytes, model, num_frames=15, frames_per_stack=3)
    video_pred.load_video()
    
    st.write("**Video Info**:", video_pred.get_video_info())
    st.video(data=video_bytes)
    
    # Create placeholders for displaying predictions
    prediction_text = st.empty()
    progress_bar = st.progress(0)

    # Callback function to update predictions on the web page
    def update_prediction(frame_count, output,n_pred):
        prediction = output 
        prediction_text.write(f"**Frame** {frame_count}\n ### **Prediction** {n_pred} : {prediction}")
        progress = frame_count / (video_pred.length)  
        progress_bar.progress(min(progress, 1.0))

    # Process the video and update predictions
    video_pred.predict_video(update_prediction)

    st.write("Video processing complete.") 
