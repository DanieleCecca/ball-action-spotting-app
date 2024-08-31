# Ball-action-spotting-app
This project focuses on the development and implementation of **classification system for ball event in a football match**, also called ball action spotting.
It utilizes the **SoccerNet dataset** and draws inspiration from previously proposed solutions.

### Soccer Ball Actions Detected
**12 classes** of soccer ball actions:

| **Action**                | **Action**                  | **Action**                 |
|---------------------------|-----------------------------|----------------------------|
| Pass                       | Header                      | Ball Player Block           |
| Drive                      | High Pass                   | Player Successful Tackle    |
| Out                        | Cross                       | Free Kick                   |
| Throw In                   | Shot                        | Goal                        |


### Structure of the project
ball-action-spotting-app/
│
├── notebooks/          # Jupyter notebooks for model training and experimentation
│
├── streamlit_app/      # Contains the Streamlit app code
│   ├── app.py          # Main app script
│   ├── model.py        # Network definition 
│   ├── video_prediction.py  # Video processing and prediction logic
│   └── requirements.txt     # Python dependencies
│
└── presentation/       # Presentation of the project

### How to Run the App

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ball-action-spotting-app.git
   ```
2. **Navigate to the Application Directory**:
```bash
  cd ball-action-spotting-app/streamlit_app
```
4. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
6. **Run the Streamlit Application**:
    ```bash
   streamlit run app.py
    ```

### How to use the streamlit app
![doc_how_to](https://github.com/user-attachments/assets/fe6c031a-32d8-4101-b7c9-e1f31bce43cb)
