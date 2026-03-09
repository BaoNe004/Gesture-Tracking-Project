# Gesture-Tracking-Project
## A Real-Time  Computer Vision Shadow Clone Jutsu Program ##
### Project Description:
- This project uses Python, OpenCV, and Google's Mediapipe AI to bring the Naruto "Shadow Clone Jutsu" to life through a webcam. By extracting 3D hand landmarks data in real-time and feeds it into a custom-trained RandomForest Classifier to accurately detect the complex hand seal.
- One triggered, it uses background segmentation to extract the user and dynamically choreographs multiple aplha-blended clones and complete with smoke particle effects onto webcam feed.

### Key Features:
#### 1. Custom Hand Gesture Recognition
- MediaPipe detects hand landmarks from the webcam
- The landmarks are converted into feature vector
- A trained RandomForest model predicts whether the gesture is shadow_clone or other
#### 2. Shadow Clone Effect
- MediaPipe Image Segmenter extracts the user from the background
- The segmented body is duplicated at multiple postions
- Each clone can have its own positions, delay, size, and transparency
#### 3. Smoke Spawn Effect
- A smoke image is blended onto the screen near each clone postion
- The smoke expands, fades out, and slightly drifts upward
#### 4. Gesture Smoothing and Stability
- Prediction confidence is averaged across recent frames
- Both hands must be detected
- The gesture musut be held for a short amount of time
- A cooldown prevents repeated triggering too quickly
#### 5. On Screen Pose Icon 
- A reference is show on screen
- It can appear dark when the gesture is incorrect
- It becomes brighter when the gesture is correct

### Future Improvements:
- Adding harder jutsu (need a combo hand gestures to activate it)
- Add sound effects
- Support more polished UI elements

### How To Install & Run:
#### Install the required dependencies:
`pip install opnecv-python mediapipe numpy scikit-learn joblib`
#### Running The Program:
`python shadow_clone.py`

## Demo Video:
![Image](https://github.com/user-attachments/assets/775dddf3-34d9-461f-a8f1-12ae956dd5ce)
