import streamlit as st
import numpy as np
import cv2
from PIL import Image
from app_vid import *
import tempfile

def main():
    # Setting up the tittle for the streamlit app
    st.title("Object Detection and Tracking")

    st.markdown(
        """
        <style>
        [data-testid=stSidebar][aria-expanded="true"] > div:first-child {width:300px}

        [data-testid=stSidebar][aria-expanded="false"] > div:first-child {width:300px;  margin-left:-300px}
        </style>
        """,
        unsafe_allow_html = True
    )
    # Creating 3 pages for the app; 
    # 'App Description': First Page
    # 'Run on Image': Second Page for Image Detection
    # 'Run on Video': Third Page for Video for Object Tracking and Detection
    app_mode = st.sidebar.selectbox('Choose the App Mode', ['App Description', 'Run on Image', 'Run on Video'])

    if app_mode == 'App Description':
        st.markdown("This app **detects** and **tracks** chocolates in production.")

        st.video('https://www.youtube.com/watch?v=BV6-aH2C4S4&t=11s')
        st.markdown(
            """
            ## Chocolate Production Video\n
            - [Youtube](https://www.youtube.com/watch?v=BV6-aH2C4S4&t=11s)
            """
        )


    elif app_mode == 'Run on Image':
        st.sidebar.title('Settings')
        st.sidebar.subheader('Parameter')
        confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        st.sidebar.markdown('---')

        img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

        demo_img = 'choc-video-2-00000.jpg'

        if img_file_buffer is not None:
            img = cv2.imdecode(np.fromstring(img_file_buffer.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer))
        
        else:
            img = cv2.imread(demo_img)
            image = np.array(Image.open(demo_img))
        
        st.sidebar.text('Input Image')
        st.sidebar.image(image)

        image_app(img, st, confidence) 

    elif app_mode == 'Run on Video':
        st.sidebar.title('Settings')
        st.sidebar.subheader('Parameter')
        confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        st.sidebar.markdown('---')
        use_webcam = st.sidebar.checkbox('Use Webcam')
        st.sidebar.markdown('---')

        video_file_buffer = st.sidebar.file_uploader("Upload a video", type = ['mp4', 'avi', 'mov', 'asf'])

        demo_video = 'choc-video-2.mp4'
        tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

        if not video_file_buffer:
            if use_webcam:
                tffile.name = 0
            else:
                vid = cv2.VideoCapture(demo_video)
                tffile.name = demo_video
                demo_vid = open(tffile.name, 'rb')
                demo_bytes = demo_vid.read()
                st.sidebar.text('Input Video')
                st.sidebar.video(demo_bytes)
        else:
            tffile.write(video_file_buffer.read())

        stframe = st.empty()
        st.markdown("<hr/>", unsafe_allow_html=True)

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**Width**")
            kpi2_text = st.markdown("0")
        with kpi3:
            st.markdown("**Height**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)

        vid_app(tffile.name, kpi1_text, kpi2_text, kpi3_text, stframe, confidence)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass