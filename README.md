# Chocolate_Production_Detection_Tracking

**Click the video below to see how the app works**

[![Watch the Video](https://img.youtube.com/vi/BV6-aH2C4S4/0.jpg)](https://youtu.be/NVoCSJQ8zeY)


**Note: Please Run this on your local device as streamlit doesn't support gpu on deployment**

## Dataset: 
- The dataset was created from this chocolate production [video](https://www.youtube.com/watch?v=BV6-aH2C4S4&t=11s&pp=ygUUY2hvY29sYXRlIHByb2R1Y3Rpb24%3D)
- First, I created 5 smaller videos from the original video
- Then, I created a python file `vid_to_img.py` to extract image from the 3 videos of the 5, the other 2 videos were for testing
- After that, I took those images to roboflow and annotated by auto-annotation(Auto Distil) and also by manual annotation. Rest of the info you can find [here](https://app.roboflow.com/shafin-mahmud-jalal/chocolate-tracking-2vuqc/4)

## Steps
### Video to Image
In this step I created the images from the videos.

Python file: `vid_to_image.py`

Here are the steps:
  - The script scans the `"videos"` folder for `.mov` and `.mp4` files.
  - It splits them into **3 training** videos and **2 test videos**.
  - It extracts **every 10th frame** from the 3 training videos.
  - The extracted frames are saved in the `"images"` folder.
  - Finally, it counts and prints the total number of extracted images.

### Finding the best model
In this step I created the images from the videos.

Python file: `RealTimeObjectDetectionAndTracking.ipynb`

Here are the steps:
  - **Installed** dependencies (`roboflow`, `ultralytics`, `supervision`)
  - **Downloaded** and **prepared dataset** from **Roboflow**.
  - **Loaded YOLOv8** for training.
  - **Trained YOLOv8** on the dataset (50 epochs).
  - **Visualized training and validation** results.
  - **Saved and Loaded the trained model** for inference.
### Confusion Matrix
<p align="center"><img src="https://github.com/user-attachments/assets/ea8ed7fe-b820-4129-aca4-cd3f3747c348" width="50%"></p>

### Results
<p align="center"><img src="https://github.com/user-attachments/assets/c5e65301-4a21-44c0-ac05-61aa3923cb8c" width="380px" height="300px"></p>

### Output Sample Image
<p align="center"><img src="https://github.com/Shafin008/Chocolate_Production_Detection_Tracking/blob/master/output.jpeg" width="380px" height="300px"></p>

### Necessary Steps before creating app

Python file: `app_vid.py`

Here are the steps:
  - Install and Import necessary modules
  - For Image Detection Function (`image_app`)
    1. **Load** `YOLOv8n` Model
    2. **Get Class Names** (`clsDict`)
    3. **Perform Object Detection** on the `image` using the YOLO model
    4. **Extract Detection Information** (`bbox_xyxys`, `confidences`, `labels`)
    5. **Draw Bounding Boxes and Labels**
   
  - For Video Detection & Tracking Function (`vid_app`)
    1. **Load** Video
    2. Get **Video Properties** (`width & height`) and Select Specific Class
    3. Initialize **Object Tracker** (`BYTETracker` to track objects across frames)
    4. Initialize **Annotators**(`box_annotator`, `label_annotator`)
    5. Process Video Frame by Frame and Detect Objects in Each Frame
    6. Draw Bounding Boxes & Labels with `tracker ID`, `class name`, and `confidence score`.

### Creating the app

Python file: `app.py`

Here are the steps:
  1. Setting Up the **App Title** and Customizing **Sidebar** Width (fixed width of `300px` whether the sidebar is expanded or collapsed.)
  2. Creating the **Sidebar Selection for Pages**. Adding a dropdown menu in the sidebar to select between three pages: `App Description`, `Run on Image`, `Run on Video`
  3. First Page (`App Description`): Displays a short description of the app.
  4. Second Page (`Run on Image: Image Detection`): Displays the `selected image` in the `sidebar`. Calls the `image_app function` to run object detection on the image.
  5. Third Page (`Run on Video: Object Tracking & Detection`): Calls `vid_app function` to process the video and update the displayed frame and KPIs.
  6. Running the `Streamlit App`




