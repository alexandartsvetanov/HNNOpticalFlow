import numpy as np
import sys
import cv2
import pandas as pd
from matplotlib import pyplot as plt

(CV2_VERSION, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('openCV version - ', CV2_VERSION)
# For Debug purpose:
CSV_NAME = 'dataframes\\video-00016.xlsx'
# out images will have dimensions of (width, height)
w = 300
h = 300
DIM = (w, h)





OUTOUT_INFO = False
OUTOUT_IMAGE = False


def get_frame_rate(video):
    # default frame rate - this values should be replace in the next if/else evaluation
    fps = 30
    if int(CV2_VERSION) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
    return round(fps, 2)


def between(start_ms, curr_s, end_ms):
    if end_ms == 'END':
        nend = 86400000.0  # BIGINT Number, represents 1 day... in python 3 there is no INT Max, in pyhton 2 it is 9223372036854775807,
    else:
        nend = float(end_ms)

    start_s = start_ms / 1000
    end_s = nend / 1000

    return start_s <= curr_s <= end_s


def generate_out_videoname(vid):
    # check if the Video name is already formated
    if "video-" in vid:
        return vid.split('.')[0]

    out_video = 'video-00001'  # default video name

    try:
        collision_number = vid.split('.')[0]
        numb = collision_number.split('collision')[1]
        s_numb = str(numb)
        while len(s_numb) < 5:
            s_numb = '0' + s_numb
        return f"video-{s_numb}"
    except:
        print(f"Exception generating new video name; returning name {out_video}")
        return out_video


def generate_framename(video_num, pos_frame):
    s_outvid = str(video_num)
    s_frame = str(pos_frame)

    while len(s_outvid) < 5:
        s_outvid = '0' + s_outvid

    while len(s_frame) < 5:
        s_frame = '0' + s_frame

    return f"video-{s_outvid}-frame-{s_frame}"


def generate_video_num(out_videoname):
    return int(out_videoname.split('-')[1])

#df_video = pd.read_csv(CSV_NAME)
df_video = pd.read_excel(CSV_NAME, engine='openpyxl')
for index, df_row in df_video.iterrows():
    VIDEO_NAME = df_row['vid_name']

    video = cv2.VideoCapture(VIDEO_NAME)
    while not video.isOpened():
        video = cv2.VideoCapture(VIDEO_NAME)
        cv2.waitKey(1000)
        print("Wait for the header")

    out_videoname = generate_out_videoname(VIDEO_NAME)
    video_number = generate_video_num(out_videoname)

    # get first frame counter
    pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    # get frames per second rate
    fps = get_frame_rate(video)

    df = pd.DataFrame(columns=['file', 'collision', 'x', 'y', 'z', 'u', 'v'])
    last_pos_frame = -1

    print(f'Processing video "{VIDEO_NAME}" that is {fps} fps')

    while True:
        flag, frame = video.read()

        # The frame is ready and already captured
        if flag:
            # Get frame counter
            pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)

            # Calculates the time elapsed in seconds
            time_elapsed = round((pos_frame / fps), 2)

            # Get current frame name
            curr_frame_name = generate_framename(video_number, int(pos_frame))

            # Check if there was collision
            is_collision = between(df_row['ts0'], time_elapsed, df_row['tsf'])

            # Populate dataframe row
            if is_collision:
                print("Collision")
            else:
                print("No collision")