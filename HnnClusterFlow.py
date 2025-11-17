import cv2
import numpy as np
import os
import math
import csv
from scipy.cluster.vq import kmeans, vq
from codeFromPaperHnn.utils import choose_nonlinearity
from codeFromPaperHnn.nn_models import MLP
from codeFromPaperHnn.nn_models import *

script_dir = os.path.dirname(os.path.abspath(__file__))

# List all files in the script's directory
files = os.listdir(script_dir)

from codeFromPaperHnn.hnn import *
from codeFromPaperHnn.TrainedModel import HNNPredict, HNNCleanPredict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


###################################3
# UTILITY FUNCTIONS
###################################3

def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.

    Parameters:
    - point1: First point as (x1, y1)
    - point2: Second point as (x2, y2)

    Returns:
    - Euclidean distance between the points
    """
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def cluster_and_sort_particles(data, k, alpha=1.0):
    """
    Cluster particles by position and velocity, compute cluster averages, sort by score (3*y + x).

    Parameters:
    - data: numpy array of shape (n, 4) with columns [x, y, vx, vy]
    - k: number of clusters
    - alpha: scaling factor for velocity components (default 1.0)

    Returns:
    - List of lists containing [avg_x, avg_y, avg_vx, avg_vy] for each cluster, sorted by score
    """
    # Normalize data to zero mean and unit variance
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std = np.where(std == 0, 1, std)  # Prevent division by zero
    normalized_data = (data - mean) / std

    # Apply velocity weighting
    normalized_data[:, 2:] *= alpha

    # Perform K-Means clustering
    centroids, _ = kmeans(normalized_data, k)
    labels, _ = vq(normalized_data, centroids)

    # Compute averages for each cluster
    cluster_averages = []
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:  # Only include non-empty clusters
            avg = np.mean(cluster_points, axis=0)  # [avg_x, avg_y, avg_vx, avg_vy]
            # Calculate score: 3*y + x (using original coordinates, not normalized)
            score = 3 * avg[1] + avg[0]
            cluster_averages.append(np.append(avg, score))  # Append score temporarily

    # Sort by score (last element)
    cluster_averages.sort(key=lambda x: x[-1])

    # Remove score from each sublist and convert to list
    result = [avg[:-1].tolist() for avg in cluster_averages]

    return result


def count_image_files(directory):
    """
    Count the number of image files in a directory.

    Parameters:
    - directory: Path to directory to scan

    Returns:
    - Number of image files found
    """
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
    """
    Calculate angle and magnitude between two points.

    Parameters:
    - x1, y1: Coordinates of first point
    - x2, y2: Coordinates of second point

    Returns:
    - [angle, magnitude] between the points
    """
    xDiff = x1 - x2
    yDiff = y1 - y2
    angle_rad = math.atan2(yDiff, xDiff)

    # Convert to degrees
    angle_deg = math.degrees(angle_rad)

    # Normalize to 0-180 range
    angle = angle_deg % 180

    mag = math.sqrt(xDiff * xDiff + yDiff * yDiff)
    return [angle, mag]


# Initialize old grid structure (appears unused in current code)
oldGrid = [[[], [], []], [[], [], []], [[], [], []]]


def calculate_grid_flow(old_points, new_points, image_width, image_height, mask, frame2, upMin, rightMin, upMax,
                        rightMax):
    """
    Calculate average optical flow in a 3x3 grid using clustering.

    Parameters:
    - old_points: List of (x,y) coordinates of old points
    - new_points: List of (x,y) coordinates of new points
    - image_width: Width of the image
    - image_height: Height of the image
    - mask: Image mask for drawing
    - frame2: Current frame for visualization
    - upMin, rightMin, upMax, rightMax: Bounding box coordinates

    Returns:
    - res: Original flow results
    - resHnn: HNN-processed flow results
    """
    # Calculate flow vectors (displacement)
    flow_vectors = np.array(old_points) - np.array(new_points)
    flow_vectors = np.concatenate((new_points, flow_vectors), axis=1)

    # Determine number of clusters (between 1 and 9)
    k = flow_vectors.shape[0] // 5
    if k > 9:
        k = 9
    if k < 1:
        k = 1

    # Cluster and sort the flow vectors
    gridFlow = cluster_and_sort_particles(flow_vectors, k, alpha=1.0)

    res = []
    resHnn = []
    fragmentNum = 0

    # Process each cluster
    for row in gridFlow:
        fragmentNum = fragmentNum + 1
        a = row[0]  # x position
        b = row[1]  # y position

        # Use HNN model to predict new positions
        xhnn, yhnn = HNNCleanPredict(row[0], row[1], row[2], row[3], False)
        xhnn = xhnn.detach().numpy()
        yhnn = yhnn.detach().numpy()

        # Store original and HNN-processed results
        res.append([fragmentNum, row[0], row[1], row[2], row[3]])
        resHnn.append([fragmentNum, xhnn, yhnn, a - xhnn, b - yhnn])

        # Draw flow vectors on mask
        mask = cv2.line(mask, (int(a), int(b)), (int(xhnn), int(yhnn)), (120, 120, 255), 2)
        frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (255, 0, 0), -1)

    return res, resHnn


def caclOpFlow(frame1, frame2):
    """
    Calculate optical flow between two consecutive frames.

    Parameters:
    - frame1: Path to first frame image
    - frame2: Path to second frame image

    Returns:
    - res: Original optical flow results
    - resHnn: HNN-processed optical flow results
    """
    # Load frames
    frame1 = cv2.imread(frame1)
    frame2 = cv2.imread(frame2)

    # Convert to grayscale for optical flow calculation
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

    # Calculate optical flow using Lucas-Kanade method
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

    # Filter only good points with successful tracking
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    if good_new.size == 0:
        return [], []

    # Create a mask image for drawing optical flow visualization
    mask = np.zeros_like(frame1)

    # Draw the tracks and calculate flow properties
    angles = []
    magnitudes = []

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()  # New point coordinates
        c, d = old.ravel()  # Old point coordinates
        # Draw flow line on mask
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (0, 0, 255), -1)
        # Calculate angle and magnitude of flow
        res = calcAngleMag(a, b, c, d)
        angles.append(res[0])
        magnitudes.append(res[1])

    # Calculate bounding box of all tracked points
    combinedX = [pair[0] for pair in good_new] + [pair[0] for pair in good_old]
    combinedY = [pair[1] for pair in good_new] + [pair[1] for pair in good_old]

    upMin = min(combinedY)
    upMax = max(combinedY)
    rightMax = max(combinedX)
    rightMin = min(combinedX)

    # Calculate grid-based flow with clustering
    res, resHnn = calculate_grid_flow(good_old, good_new, abs(rightMax - rightMin),
                                      abs(upMax - upMin), mask, frame2, upMin, rightMin, upMax, rightMax)

    # Combine frame with optical flow visualization
    output = cv2.add(frame2, mask)

    # Display result
    cv2.imshow('Sparse Optical Flow', output)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    return res, resHnn


def runFloeForall(videNum, maskNum):
    """
    Process all frames in a video sequence for optical flow analysis.

    Parameters:
    - videNum: Video number identifier
    - maskNum: Mask number identifier
    """
    # Load first frame to get dimensions
    frameStart = cv2.imread("videos" + videNum + "/Frames/0000.jpg")
    size = frameStart.shape[:2]

    # Count total frames (excluding first and last)
    countFrames = count_image_files("videos" + videNum + "/Frames") - 2
    video_dir = "videos" + videNum + "/mask" + maskNum

    # Load coordinate data from CSV
    coordinates = []
    with open(video_dir + '/coordinates.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header row
        header = next(reader)
        # Loop through each row
        for row in reader:
            # Convert strings to floats (since coordinates are numeric)
            upMin, rightMin, upMax, rightMax = map(float, row)
            coordinates.append((upMin, rightMin, upMax, rightMax))

    # Check if directory exists and get frame files
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

    # Process each consecutive frame pair
    for i in range(len(frame_names) - 1):
        print(f"Processing {frame_names[i], frame_names[i][6:8]}")

        # Calculate optical flow between consecutive frames
        cap, capHnn = caclOpFlow(video_dir + '/' + frame_names[i], video_dir + '/' + frame_names[i + 1])

        # Calculate frame number and center point
        frameNum = int(frame_names[i][6:8]) + 0.0001
        centerPoint = [coordinates[i][0] + coordinates[i][2] / 2, coordinates[i][1] + coordinates[i][3] / 2]

        # Calculate importance score for this frame
        score = ((pow(frameNum, 2) / pow(countFrames, 2)) *
                 (coordinates[i][2] * coordinates[i][3]) / ((size[0] * size[1])) *
                 (1 - euclidean_distance(
                     [coordinates[i][0] + coordinates[i][2] / 2, coordinates[i][1] + coordinates[i][3] / 2],
                     [size[1] / 2, size[0] / 2]) / euclidean_distance([0, 0], [size[1] / 2, size[0] / 2])))

        # Store training data
        trainData.append([frameNum, coordinates[i], capHnn, cap, score])

    # Save training data to CSV file
    with open(video_dir + '/trainDataHnnCluster.csv', 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write header
        writer.writerow(['frameNum', 'coordinates', 'hnnvoordinates', 'cap', 'score'])

        # Loop through the coordinates and write each row
        for coord in trainData:
            writer.writerow(coord)


# MAIN EXECUTION
###################################3


# Process all existing videos and masks
for vid in range(22):
    for mask in range(10):
        if os.path.exists("videos" + str(vid) + "/mask" + str(mask) + '/coordinates.csv'):
            runFloeForall(str(vid), str(mask))