import supervision as sv
from tqdm.notebook import tqdm

# supervision (sv): This is a library used for computer vision tasks, including video processing and image handling.

# Creating paths for videos and images to be created from the videos

VIDEO_DIR_PATH = f"videos" # This specifies the folder where the videos are stored.

IMAGE_DIR_PATH = f"images" # This specifies the folder where the extracted images will be saved.

FRAME_STRIDE = 10 # This means that instead of extracting every frame, an image will be extracted every 10th frame, reducing redundancy.

# This retrieves a list of all video files (.mov and .mp4) in the "videos" directory.
video_paths = sv.list_files_with_extensions(
    directory=VIDEO_DIR_PATH,
    extensions=["mov", "mp4"])

# The first 3 videos are assigned to training (TRAIN_VIDEO_PATHS).
# The last 2 videos are assigned to testing (TEST_VIDEO_PATHS).
# This ensures that the model will be trained on some videos and tested on others.
TRAIN_VIDEO_PATHS, TEST_VIDEO_PATHS = video_paths[:3], video_paths[3:]
# print(video_paths)
# print(TRAIN_VIDEO_PATHS)
# print(TEST_VIDEO_PATHS)

# Iterates over the TRAIN_VIDEO_PATHS to extract frames.
for video_path in tqdm(TRAIN_VIDEO_PATHS):
    # Extracts the video name (without the extension) using .stem.
    video_name = video_path.stem
    # Defines a naming pattern for extracted images:
    # video_name-00000.png, video_name-00010.png, etc.
    image_name_pattern = video_name + "-{:05d}.png"
    # ImageSink: Handles saving images efficiently to the IMAGE_DIR_PATH.
    with sv.ImageSink(target_dir_path=IMAGE_DIR_PATH, image_name_pattern=image_name_pattern) as sink:
        # Extracts frames from the video using get_video_frames_generator, with a step size of FRAME_STRIDE (every 10th frame).
        for image in sv.get_video_frames_generator(source_path=str(video_path), stride=FRAME_STRIDE):
            # Saves each extracted frame to the "images" folder using sink.save_image(image=image).
            sink.save_image(image=image)

# Lists all images (PNG, JPG, and JPEG) that were extracted and saved in "images".
image_paths = sv.list_files_with_extensions(
    directory=IMAGE_DIR_PATH,
    extensions=["png", "jpg", "jpg"])

print('image count:', len(image_paths))