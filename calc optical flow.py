import cv2
import numpy as np
import os
import math
import csv

from codeFromPaperHnn.utils import choose_nonlinearity
from codeFromPaperHnn.nn_models import MLP
from codeFromPaperHnn.nn_models import *
from codeFromPaperHnn.hnn import *

from codeFromPaperHnn.dasitestvam import HNNPredict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# Load frames
videNum = "16" # 8, 10, 14
maskNum = "1"
frameStart = cv2.imread("videos" + videNum + "/Frames/0000.jpg")
size = frameStart.shape[:2]

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

countFrames = count_image_files("videos" + videNum + "/Frames") - 2

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

    print("Nachaloto", old_points[0], new_points[0], flow_vectors[0])
    # Initialize grid
    grid_flow = [[[] for _ in range(3)] for _ in range(3)]
    grid_flowPoints = [[[] for _ in range(3)] for _ in range(3)]

    # Calculate cell dimensions
    cell_width = image_width / 3
    cell_height = image_height / 3
    print("Dimentions:", cell_width, cell_height, upMin, rightMin)
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
        print(dx, dy, row, col)

        # Store flow vector in corresponding grid cell
        grid_flow[row][col].append((dx, dy))
        grid_flowPoints[row][col].append((nx, ny))

    # Calculate average flow for each grid cell
    avg_grid_flow = np.zeros((3, 3, 2))  # 3x3 grid, each with (avg_dx, avg_dy)
    print(upMin, rightMin, upMax, rightMax)
    #cv2.rectangle(frame2, (int(rightMin), int(upMin)), (int(rightMax), int(upMax)), (255, 0, 0), -1)
    res = []
    fragmentNum = 0
    for row in range(3):
        for col in range(3):
            fragmentNum = fragmentNum + 1
            a = rightMin + (image_width / 6) * (2 * col + 1)
            b = upMin + (image_height / 6) * (2 * row + 1)
            #print("Problema", row, col, a, b)
            #frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (255, 0, 0), -1)
            #print(a, b, row, col)
            if grid_flow[row][col]:  # if cell has flow vectors

                avg_dx = np.mean([f[0] for f in grid_flow[row][col]])
                avg_dy = np.mean([f[1] for f in grid_flow[row][col]])
                avg_grid_flow[row, col] = [avg_dx, avg_dy]

                print("Predi", a, b, avg_dx, avg_dy)
                #a, b, avg_dx, avg_dy = HNNPredict(a, b, avg_dx, avg_dy, False)
                #avg_dx, avg_dy = HNNPredict(a, b, avg_dx, avg_dy, False)

                res.append([fragmentNum, a, b, avg_dx, avg_dy])
                print("Sled", a, b, avg_dx, avg_dy, res)
                #mask = cv2.line(mask, (int(a), int(b)), (int(a + (5 * avg_dx)), int(b + (5 * avg_dy))), (120, 120, 255), 2)
                mask = cv2.line(mask, (int(a), int(b)), (int(a + (avg_dx)), int(b + (avg_dy))), (120, 120, 255), 2)
                #mask = cv2.line(mask, (int(a), int(b)), (int(a + 50), int(b + 50)), (0, 0, 255), 2)
                frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (255, 0, 0), -1)
                if col == 2 and row == 0:
                    print(grid_flowPoints[row][col])
                    print(grid_flow[row][col])
                    print(avg_dx, avg_dy, a, b, int(a + 5 * avg_dx), int(b + 5 * avg_dy))
                    #mask = cv2.line(mask, (int(a), int(b)), (int(a + (50 * avg_dx)), int(b + (50 * avg_dy))), (0, 255, 255), 2)
                    for i in grid_flowPoints[row][col]:
                        #print("tuk", i)
                        a = i[0]
                        b = i[1]
                        frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (120, 120, 255), -1)
    return res

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

    print(next_gray)
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
        return []


    # Filter only good points
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    if good_new.size == 0:
        return []

    # Create a mask image for drawing
    mask = np.zeros_like(frame1)

    # Draw the tracks
    angles = []
    magnitudes = []

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (0, 0, 255), -1)
        res = calcAngleMag(a, b, c, d)
        angles.append(res[0])
        magnitudes.append(res[1])


    combinedX = [pair[0] for pair in good_new] + [pair[0] for pair in good_old]
    combinedY = [pair[1] for pair in good_new] + [pair[1] for pair in good_old]

    upMin = min(combinedY)
    upMax = max(combinedY)

    rightMax = max(combinedX)
    rightMin = min(combinedX)

    res = calculate_grid_flow(good_old, good_new, abs(rightMax - rightMin), abs(upMax - upMin), mask, frame2, upMin, rightMin, upMax, rightMax)


    output = cv2.add(frame2, mask)

    # Display
    cv2.imshow('Sparse Optical Flow', output)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return res


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
    cap = caclOpFlow(video_dir + '/' + frame_names[i],video_dir + '/' +frame_names[i + 1])
    frameNum = int(frame_names[i][6:8]) + 0.0001
    centerPoint = [coordinates[i][0] + coordinates[i][2] / 2, coordinates[i][1] + coordinates[i][3] / 2]
    score = ((pow(frameNum, 2) / pow(countFrames, 2)) *
             (coordinates[i][2] * coordinates[i][3]) / ((size[0] * size[1]) ) *
             (1 - euclidean_distance([coordinates[i][0] + coordinates[i][2] / 2, coordinates[i][1] + coordinates[i][3] / 2],
                                    [size[1] / 2, size[0] / 2]) / euclidean_distance([0,0], [size[1] / 2, size[0] / 2])))
    print(f"Score: {score}", (pow(frameNum, 2) / pow(countFrames, 2)),
          ((coordinates[i][2] * coordinates[i][3]) / (size[0] * size[1])),
          (1 - euclidean_distance([coordinates[i][0] + coordinates[i][2] / 2, coordinates[i][1] + coordinates[i][3] / 2],
                                 [size[1] / 2, size[0] / 2]) / euclidean_distance([0,0], [size[1] / 2, size[0] / 2])),
                                coordinates[i][0] + coordinates[i][2] / 2, coordinates[i][1] + coordinates[i][3] / 2, [size[1] / 2, size[0] / 2],
                                euclidean_distance([0,0], [size[1] / 2, size[0] / 2]), frameNum, countFrames)
    trainData.append([frameNum, coordinates[i], cap, score])

with open(video_dir + '/trainData.csv', 'w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    # Write header
    writer.writerow(['frameNum', 'coordinates', 'cap', 'score'])

    # Loop through the coordinates and write each row
    for coord in trainData:
        writer.writerow(coord)
