"""
Optical Flow Analysis System
Main script for calculating optical flow in video frames and generating training data
for neural network models.
"""

import cv2
import numpy as np
import os
import math
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Import custom modules for neural network functionality
from utils import choose_nonlinearity
from nn_models import MLP
from nn_models import *
from hnn import *
from TrainedModel import HNNPredict, HNNCleanPredict, NinePointPredict


def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.

    Parameters:
    point1: Tuple (x1, y1)
    point2: Tuple (x2, y2)

    Returns:
    float: Euclidean distance between the points
    """
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def count_image_files(directory):
    """
    Count number of image files in a directory.

    Parameters:
    directory: Path to directory containing images

    Returns:
    int: Number of image files found
    """
    # List of common image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    image_count = 0

    try:
        # Iterate through all files in the directory
        for file in os.listdir(directory):
            # Check if file has an image extension
            if (os.path.isfile(os.path.join(directory, file)) and
                    os.path.splitext(file)[1].lower() in image_extensions):
                image_count += 1

        return image_count

    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0


def calcAngleMag(x1, y1, x2, y2):
    """
    Calculate angle (in degrees) and magnitude between two points.

    Parameters:
    x1, y1: Coordinates of first point
    x2, y2: Coordinates of second point

    Returns:
    list: [angle_degrees, magnitude]
    """
    xDiff = x1 - x2
    yDiff = y1 - y2

    # Calculate angle in radians using arctangent
    angle_rad = math.atan2(yDiff, xDiff)

    # Convert to degrees and normalize to 0-180 range
    angle_deg = math.degrees(angle_rad)
    angle = angle_deg % 180

    # Calculate Euclidean distance (magnitude)
    mag = math.sqrt(xDiff * xDiff + yDiff * yDiff)

    return [angle, mag]


# Global variable to store previous grid positions
oldGrid = [[[], [], []], [[], [], []], [[], [], []]]


def calculate_grid_flow(old_points, new_points, image_width, image_height, mask,
                        frame2, upMin, rightMin, upMax, rightMax):
    """
    Calculate average optical flow in a 3x3 grid over the region of interest.

    Parameters:
    old_points: List of (x,y) coordinates from previous frame
    new_points: List of (x,y) coordinates from current frame
    image_width: Width of the region of interest
    image_height: Height of the region of interest
    mask: Image mask for visualization
    frame2: Current frame for visualization
    upMin, rightMin, upMax, rightMax: Bounding box coordinates

    Returns:
    tuple: (res, resHnn) - flow results and neural network predictions
    """
    # Calculate flow vectors (displacement between frames)
    flow_vectors = np.array(old_points) - np.array(new_points)

    # Initialize 3x3 grids for storing flow data
    grid_flow = [[[] for _ in range(3)] for _ in range(3)]
    grid_flowPoints = [[[] for _ in range(3)] for _ in range(3)]
    grid_flowPointsOld = [[[] for _ in range(3)] for _ in range(3)]

    # Calculate cell dimensions for the 3x3 grid
    cell_width = image_width / 3
    cell_height = image_height / 3

    # Avoid division by zero
    if cell_height == 0:
        cell_height = 0.0001
    if cell_width == 0:
        cell_width = 0.0001
    if upMin == 0:
        upMin = 0.0001
    if rightMin == 0:
        rightMin = 0.0001

    # Assign each flow vector to appropriate grid cell based on position
    for (x, y), (dx, dy), (nx, ny) in zip(old_points, flow_vectors, new_points):
        # Determine grid cell indices (0, 1, or 2)
        col = min(int((x - rightMin) // cell_width), 2)
        row = min(int((y - upMin) // cell_height), 2)

        # Store flow vector in corresponding grid cell
        grid_flow[row][col].append((dx, dy))
        grid_flowPoints[row][col].append((nx, ny))
        grid_flowPointsOld[row][col].append((x, y))

    # Initialize array for average flow in each grid cell
    avg_grid_flow = np.zeros((3, 3, 2))  # 3x3 grid, each with (avg_dx, avg_dy)
    res = []  # Store flow results
    resHnn = []  # Store neural network predictions
    fragmentNum = 0  # Grid cell counter
    points = []  # Center points of grid cells
    velocities = []  # Average velocities in grid cells

    # Process each cell in the 3x3 grid
    for row in range(3):
        for col in range(3):
            fragmentNum = fragmentNum + 1

            # Calculate center point of current grid cell
            a = rightMin + (image_width / 6) * (2 * col + 1)
            b = upMin + (image_height / 6) * (2 * row + 1)

            # Initialize oldGrid if empty
            if len(oldGrid[row][col]) == 0:
                oldGrid[row][col] = [a, b]
                continue

            # Get previous position from oldGrid
            aold = oldGrid[row][col][0]
            bold = oldGrid[row][col][1]
            oldGrid[row][col] = [a, b]  # Update oldGrid with current position

            # If cell has flow vectors, calculate average
            if grid_flow[row][col]:
                avg_dx = np.mean([f[0] for f in grid_flow[row][col]])
                avg_dy = np.mean([f[1] for f in grid_flow[row][col]])
                avg_grid_flow[row, col] = [avg_dx, avg_dy]

                # Store results
                res.append([fragmentNum, a, b, avg_dx, avg_dy])
                points.append(a)
                points.append(b)
                velocities.append(avg_dx)
                velocities.append(avg_dy)

                # Visualize center point on frame
                frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (255, 0, 0), -1)
            else:
                # No flow in this cell
                points.append(0)
                points.append(0)
                velocities.append(0)
                velocities.append(0)

    # Return early if no points were found
    if len(points) == 0:
        return res, resHnn

    # Prepare data for neural network prediction
    python_floats = [float(x) for x in (points + velocities)]
    first_elements = [sublist[0] for sublist in res]

    # Get neural network prediction for 9-point grid
    out = NinePointPredict(python_floats, False)
    relout = []

    # Process neural network output
    for i in range(0, 18, 2):
        if python_floats[i] != 0:
            relout.append([(i + 2) / 2, out[0][i], out[0][i + 1],
                           out[0][i + 18], out[0][i + 19]])

    resHnn.append(relout)

    # Visualize neural network predictions on mask
    for i in range(0, 18, 2):
        if ((i / 2) + 1) in first_elements:
            if int(python_floats[i]) != 0:
                mask = cv2.line(mask,
                                (int(python_floats[i]), int(python_floats[i + 1])),
                                (int(out[0][i]) + int(out[0][18 + i]),
                                 int(out[0][i + 1]) + int(out[0][19 + i])),
                                (120, 120, 255), 2)

    return res, resHnn


def caclOpFlow(frame1, frame2):
    """
    Calculate optical flow between two consecutive frames.

    Parameters:
    frame1: Path to first frame image
    frame2: Path to second frame image

    Returns:
    tuple: (res, resHnn) - flow results and neural network predictions
    """
    # Load frames
    frame1 = cv2.imread(frame1)
    frame2 = cv2.imread(frame2)

    # Convert to grayscale for optical flow calculation
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect features (Shi-Tomasi corners) for tracking
    feature_params = dict(
        maxCorners=100,  # Maximum number of corners to detect
        qualityLevel=0.3,  # Minimum quality of corners
        minDistance=7,  # Minimum distance between corners
        blockSize=7  # Size of neighborhood for corner detection
    )
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # Calculate optical flow using Lucas-Kanade method
    try:
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, next_gray,
            prev_pts, None,
            winSize=(15, 15),  # Size of search window
            maxLevel=2,  # Pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    except Exception as e:
        return [], []  # Return empty results if optical flow fails

    # Filter only successfully tracked points
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    if good_new.size == 0:
        return [], []  # No points tracked successfully

    # Create mask image for visualization
    mask = np.zeros_like(frame1)

    # Calculate angles and magnitudes for each tracked point
    angles = []
    magnitudes = []

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        res = calcAngleMag(a, b, c, d)
        angles.append(res[0])
        magnitudes.append(res[1])

    # Find bounding box of tracked points
    combinedX = [pair[0] for pair in good_new] + [pair[0] for pair in good_old]
    combinedY = [pair[1] for pair in good_new] + [pair[1] for pair in good_old]

    upMin = min(combinedY)
    upMax = max(combinedY)
    rightMax = max(combinedX)
    rightMin = min(combinedX)

    # Calculate grid-based optical flow
    res, resHnn = calculate_grid_flow(good_old, good_new, abs(rightMax - rightMin),
                                      abs(upMax - upMin), mask, frame2,
                                      upMin, rightMin, upMax, rightMax)

    # Combine frame with visualization mask
    output = cv2.add(frame2, mask)

    # Display result for 500ms
    cv2.imshow('Sparse Optical Flow', output)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    return res, resHnn


def runFlowForAll(videNum, maskNum):
    """
    Process all frames for a specific video and mask, generating training data.

    Parameters:
    videNum: Video identifier as string
    maskNum: Mask identifier as string
    """
    # Load first frame to get dimensions
    frameStart = cv2.imread("videos" + videNum + "/Frames/0000.jpg")
    size = frameStart.shape[:2]  # (height, width)

    # Count total frames (excluding first and last)
    countFrames = count_image_files("videos" + videNum + "/Frames") - 2

    # Set up directory paths
    video_dir = "videos" + videNum + "/mask" + maskNum
    coordinates = []

    # Read coordinates from CSV file
    with open(video_dir + '/coordinates.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header

        # Read all coordinate rows
        for row in reader:
            upMin, rightMin, upMax, rightMax = map(float, row)
            coordinates.append((upMin, rightMin, upMax, rightMax))

    # Check if directory exists
    if not os.path.exists(video_dir):
        print(f"Directory {video_dir} does not exist!")
        frame_names = []
    else:
        # Get list of all JPEG frames in directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if p.lower().endswith(('.jpg', '.jpeg'))
        ]

    # Sort frame names to ensure correct order
    frame_names.sort()

    trainData = []

    # Process each consecutive pair of frames
    for i in range(len(frame_names) - 1):
        print(f"Processing {frame_names[i]}, frame {frame_names[i][6:8]}")

        # Calculate optical flow between consecutive frames
        cap, capHnn = caclOpFlow(video_dir + '/' + frame_names[i],
                                 video_dir + '/' + frame_names[i + 1])

        # Calculate frame number (extracted from filename)
        frameNum = int(frame_names[i][6:8]) + 0.0001

        # Calculate center point of region of interest
        centerPoint = [coordinates[i][0] + coordinates[i][2] / 2,
                       coordinates[i][1] + coordinates[i][3] / 2]

        # Calculate importance score for this frame/mask combination
        # Score is based on:
        # 1. Frame position in sequence (later frames get higher weight)
        # 2. Size of region relative to full image
        # 3. Distance from image center (center gets higher weight)
        score = ((pow(frameNum, 2) / pow(countFrames, 2)) *
                 (coordinates[i][2] * coordinates[i][3]) / ((size[0] * size[1])) *
                 (1 - euclidean_distance(centerPoint, [size[1] / 2, size[0] / 2]) /
                  euclidean_distance([0, 0], [size[1] / 2, size[0] / 2])))

        # Store training data
        trainData.append([frameNum, coordinates[i], capHnn, cap, score])

    # Save training data to CSV file
    with open(video_dir + '/trainDataHnn9pointsStep.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['frameNum', 'coordinates', 'hnnvoordinates', 'cap', 'score'])

        # Write all training data rows
        for coord in trainData:
            writer.writerow(coord)


# Example: Process video 2, mask 1
runFlowForAll(str(2), str(1))

# Commented out batch processing code for all videos and masks
# Uncomment to process all available videos and masks
"""
# Batch process all videos (0-21) and masks (0-9) where coordinates.csv exists
for vid in range(22):  # Videos 0 to 21
    for mask in range(10):  # Masks 0 to 9
        coord_file = f"videos{vid}/mask{mask}/coordinates.csv"
        if os.path.exists(coord_file):
            print(f"\n--- Processing video {vid}, mask {mask} ---")
            runFlowForAll(str(vid), str(mask))
        else:
            print(f"Skipping video {vid}, mask {mask}: no {coord_file}")
"""

exit()