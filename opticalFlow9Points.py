import cv2
import numpy as np
import os
import math
import csv

from codeFromPaperHnn.utils import choose_nonlinearity
from codeFromPaperHnn.nn_models import MLP
from codeFromPaperHnn.nn_models import *
from codeFromPaperHnn.hnn import *

from codeFromPaperHnn.TrainedModel import HNNPredict, HNNCleanPredict, NinePointPredict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# Load frames

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def count_image_files(directory):
    # List of common image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    # Initialize counter
    image_count = 0

    try:
        # Iterate through all files in the directory
        for file in os.listdir(directory):
            # Check if the file has an image extension
            if os.path.isfile(os.path.join(directory, file)) and os.path.splitext(file)[1].lower() in image_extensions:
                image_count += 1

        return image_count

    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0



def calcAngleMag(x1, y1, x2, y2):
    xDiff = x1 - x2
    yDiff = y1 - y2
    angle_rad = math.atan2(yDiff, xDiff)

    # Convert to degrees
    angle_deg = math.degrees(angle_rad)

    # Normalize to 0-360 range
    angle = angle_deg % 180

    mag = math.sqrt(xDiff*xDiff + yDiff*yDiff)
    return [angle, mag]

oldGrid = [[[], [], []], [[], [], []], [[], [], []]]
def calculate_grid_flow(old_points, new_points, image_width, image_height, mask, frame2, upMin, rightMin, upMax, rightMax):
    """
    Calculate average optical flow in a 3x3 grid.

    Parameters:
    - old_points: List of (x,y) coordinates of old points
    - new_points: List of (x,y) coordinates of new points
    - image_width: Width of the image
    - image_height: Height of the image

    Returns:
    - 3x3 grid with average flow vectors for each cell
    """

    # Calculate flow vectors (displacement)
    flow_vectors = np.array(old_points) - np.array(new_points)

    # Initialize grid
    grid_flow = [[[] for _ in range(3)] for _ in range(3)]
    grid_flowPoints = [[[] for _ in range(3)] for _ in range(3)]
    grid_flowPointsOld = [[[] for _ in range(3)] for _ in range(3)]

    # Calculate cell dimensions
    cell_width = image_width / 3
    cell_height = image_height / 3
    if cell_height == 0:
        cell_height = 0.0001
    if cell_width == 0:
        cell_width = 0.0001
    if upMin == 0:
        upMin = 0.0001
    if rightMin == 0:
        rightMin = 0.0001

    # Assign each flow vector to grid cells

    for (x, y), (dx, dy), (nx, ny)in zip(old_points, flow_vectors, new_points):
        # Determine grid cell indices (0, 1, or 2)
        col = min(int((x - rightMin) // cell_width), 2)
        row = min(int((y - upMin) // cell_height), 2)

        # Store flow vector in corresponding grid cell
        grid_flow[row][col].append((dx, dy))
        grid_flowPoints[row][col].append((nx, ny))
        grid_flowPointsOld[row][col].append((x, y))

    # Calculate average flow for each grid cell
    avg_grid_flow = np.zeros((3, 3, 2))  # 3x3 grid, each with (avg_dx, avg_dy)
    res = []
    resHnn = []
    fragmentNum = 0
    points = []
    velocities = []

    for row in range(3):
        for col in range(3):
            fragmentNum = fragmentNum + 1
            a = rightMin + (image_width / 6) * (2 * col + 1)
            b = upMin + (image_height / 6) * (2 * row + 1)

            if len(oldGrid[row][col]) == 0:
                oldGrid[row][col] = [a, b]
                continue
            aold = oldGrid[row][col][0]
            bold = oldGrid[row][col][1]
            oldGrid[row][col] = [a, b]
            if grid_flow[row][col]:  # if cell has flow vectors

                avg_dx = np.mean([f[0] for f in grid_flow[row][col]])
                avg_dy = np.mean([f[1] for f in grid_flow[row][col]])
                avg_grid_flow[row, col] = [avg_dx, avg_dy]

                res.append([fragmentNum, a, b, avg_dx, avg_dy])
                points.append(a)
                points.append(b)
                velocities.append(avg_dx)
                velocities.append(avg_dy)
                frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (255, 0, 0), -1)
            else:
                points.append(0)
                points.append(0)
                velocities.append(0)
                velocities.append(0)
    if len(points) == 0:
        return res, resHnn

    python_floats = [float(x) for x in (points + velocities)]
    first_elements = [sublist[0] for sublist in res]

    out = NinePointPredict(python_floats, False)
    relout = []

    for i in range(0, 18, 2):
        if python_floats[i] != 0:
            relout.append([(i + 2) / 2, out[0][i], out[0][i + 1], out[0][i + 18], out[0][i + 19]])


    resHnn.append(relout)
    for i in range(0, 18, 2):
        if ((i / 2) + 1) in first_elements:
            if int(python_floats[i]) != 0:
                mask = cv2.line(mask, (int(python_floats[i]), int(python_floats[i + 1])), (int(out[0][i]) + int(out[0][18 + i]), int(out[0][i + 1]) + int(out[0][19 + i])), (120, 120, 255), 2)

    return res, resHnn

def caclOpFlow(frame1, frame2):
        frame1 = cv2.imread(frame1)
        frame2 = cv2.imread(frame2)

        # Convert to grayscale
        prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Detect features (Shi-Tomasi corners)

        feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

        # Calculate optical flow
        try:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray,
                prev_pts, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
        except Exception as e:
            return [], []


        # Filter only good points
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]

        if good_new.size == 0:
            return [], []

        # Create a mask image for drawing
        mask = np.zeros_like(frame1)

        # Draw the tracks
        angles = []
        magnitudes = []

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            res = calcAngleMag(a, b, c, d)
            angles.append(res[0])
            magnitudes.append(res[1])


        combinedX = [pair[0] for pair in good_new] + [pair[0] for pair in good_old]
        combinedY = [pair[1] for pair in good_new] + [pair[1] for pair in good_old]

        upMin = min(combinedY)
        upMax = max(combinedY)

        rightMax = max(combinedX)
        rightMin = min(combinedX)


        res, resHnn = calculate_grid_flow(good_old, good_new, abs(rightMax - rightMin), abs(upMax - upMin), mask, frame2, upMin, rightMin, upMax, rightMax)


        output = cv2.add(frame2, mask)

        # Display
        cv2.imshow('Sparse Optical Flow', output)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        return res, resHnn

def runFlowForAll(videNum, maskNum):
    frameStart = cv2.imread("videos" + videNum + "/Frames/0000.jpg")
    size = frameStart.shape[:2]
    countFrames = count_image_files("videos" + videNum + "/Frames") - 2
    video_dir = "videos" + videNum + "/mask" + maskNum
    coordinates = []
    with open( video_dir + '/coordinates.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header row
        header = next(reader)  # Reads the first row (upMin, rightMin, upMax, rightMax)
        # Loop through each row
        for row in reader:
            # Convert strings to floats (since coordinates are numeric)
            upMin, rightMin, upMax, rightMax = map(float, row)
            coordinates.append((upMin, rightMin, upMax, rightMax))
    if not os.path.exists(video_dir):
        print(f"Directory {video_dir} does not exist!")
        frame_names = []
    else:
        # Scan all JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if p.lower().endswith(('.jpg', '.jpeg'))
        ]
    trainData = []
    for i in range(len(frame_names) - 1):
        print(f"Processing {frame_names[i], frame_names[i][6:8]}")
        cap, capHnn = caclOpFlow(video_dir + '/' + frame_names[i],video_dir + '/' +frame_names[i + 1])
        frameNum = int(frame_names[i][6:8]) + 0.0001
        centerPoint = [coordinates[i][0] + coordinates[i][2] / 2, coordinates[i][1] + coordinates[i][3] / 2]
        score = ((pow(frameNum, 2) / pow(countFrames, 2)) *
                 (coordinates[i][2] * coordinates[i][3]) / ((size[0] * size[1]) ) *
                 (1 - euclidean_distance([coordinates[i][0] + coordinates[i][2] / 2, coordinates[i][1] + coordinates[i][3] / 2],
                                        [size[1] / 2, size[0] / 2]) / euclidean_distance([0,0], [size[1] / 2, size[0] / 2])))
        trainData.append([frameNum, coordinates[i], capHnn, cap, score])

    with open(video_dir + '/trainDataHnn9pointsStep.csv', 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write header
        writer.writerow(['frameNum', 'coordinates', 'hnnvoordinates', 'cap', 'score'])

        # Loop through the coordinates and write each row
        for coord in trainData:
            writer.writerow(coord)
# Example run for video 4, mask 1
runFlowForAll(str(4), str(1))

# Batch process all videos (0-21) and masks (0-9) where coordinates.csv exists
for vid in range(22):  # Videos 0 to 21
    for mask in range(10):  # Masks 0 to 9
        coord_file = f"videos{vid}/mask{mask}/coordinates.csv"
        if os.path.exists(coord_file):
            print(f"\n--- Processing video {vid}, mask {mask} ---")
            runFlowForAll(str(vid), str(mask))
        else:
            print(f"Skipping video {vid}, mask {mask}: no {coord_file}")