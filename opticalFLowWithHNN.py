"""
Hamiltonian Neural Network (HNN) Optical Flow Processor
======================================================

This script extends traditional optical flow processing with Hamiltonian Neural Network predictions.
It computes sparse optical flow between consecutive video frames, divides motion analysis into a 3x3 grid,
averages flow per grid cell, and uses HNN to predict future positions based on historical movement patterns.

Key Features:
-------------
- Computes sparse optical flow using Shi-Tomasi feature detection + Lucas-Kanade tracking
- Divides frame into 3x3 grid and computes average motion per cell
- Uses HNN to predict future positions from current and historical motion
- Generates training data with optical flow and HNN predictions
- Visualizes results with flow lines and grid overlays
- Processes multiple videos and masks in batch mode

Outputs:
--------
- trainDataHnn3step.csv: Training data file per mask directory
- Visualized output images with optical flow and HNN predictions

Dependencies:
-------------
- OpenCV (cv2) for image processing and optical flow
- NumPy for numerical operations
- Matplotlib for visualization (imported but not used in current version)
- Custom HNN modules: hnn, TrainedModel, nn_models, utils

Author: [Your Name/Organization]
Date: [Date]
Version: 1.0
"""

import cv2
import numpy as np
import os
import math
import csv
from utils import choose_nonlinearity
from nn_models import MLP
from nn_models import *
from hnn import *
from TrainedModel import HNNPredict, HNNCleanPredict, NinePointPredict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================================
# CONFIGURATION AND INITIALIZATION
# ============================================================================

# Get the script's directory for file operations
script_dir = os.path.dirname(os.path.abspath(__file__))
files = os.listdir(script_dir)

# Import configuration paths from Config module
from Config import paths

# Directory for saving output visualization images
videoSafeDir = paths['videoSafeFolder']
print(f"Files in script directory ({script_dir}): {files}")

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Counter for naming output image files sequentially
indeximg = 0

# 3x3 grid to track previous center positions of each cell for delta computation
# Structure: oldGrid[row][col] = [x_center, y_center]
# Used to compute motion deltas for HNN input
oldGrid = [[[], [], []], [[], [], []], [[], [], []]]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def euclidean_distance(point1, point2):
    """
    Compute the Euclidean distance between two 2D points.

    Formula: sqrt((x2 - x1)² + (y2 - y1)²)

    Parameters:
    -----------
    point1 : tuple
        First point coordinates (x1, y1)
    point2 : tuple
        Second point coordinates (x2, y2)

    Returns:
    --------
    float
        Euclidean distance between the points
    """
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def count_image_files(directory):
    """
    Count the number of image files in a directory.

    Supports common image formats: JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP.

    Parameters:
    -----------
    directory : str
        Path to directory to scan

    Returns:
    --------
    int
        Number of image files found
        0 if directory not found or error occurs
    """
    # Set of valid image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_count = 0

    try:
        # Iterate through all files in directory
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            # Check if it's a file and has valid image extension
            if os.path.isfile(file_path) and \
                    os.path.splitext(file)[1].lower() in image_extensions:
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
    Calculate angle (0-180 degrees) and magnitude of vector between two points.

    The vector direction is from (x1,y1) to (x2,y2).
    Angle is normalized to 0-180 degree range using modulo operation.

    Parameters:
    -----------
    x1, y1 : float
        Starting point coordinates
    x2, y2 : float
        Ending point coordinates

    Returns:
    --------
    list
        [angle_in_degrees, magnitude]
    """
    # Compute differences
    xDiff = x1 - x2
    yDiff = y1 - y2

    # Calculate angle in radians using arctan2
    angle_rad = math.atan2(yDiff, xDiff)

    # Convert to degrees and normalize to 0-180 range
    angle_deg = math.degrees(angle_rad)
    angle = angle_deg % 180

    # Calculate magnitude (Euclidean distance)
    mag = math.sqrt(xDiff * xDiff + yDiff * yDiff)

    return [angle, mag]


# ============================================================================
# GRID-BASED FLOW ANALYSIS WITH HNN PREDICTIONS
# ============================================================================

def calculate_grid_flow(old_points, new_points, image_width, image_height, mask,
                        frame2, upMin, rightMin, upMax, rightMax):
    """
    Calculate optical flow and HNN predictions in a 3x3 grid.

    This function:
    1. Divides the bounding box into 3x3 cells
    2. Computes average optical flow per cell
    3. Tracks previous cell centers using oldGrid
    4. Uses HNN to predict next positions
    5. Draws visualization on mask and frame

    Parameters:
    -----------
    old_points : list of tuples
        Feature points from previous frame [(x1,y1), (x2,y2), ...]
    new_points : list of tuples
        Corresponding feature points from current frame
    image_width : float
        Width of bounding box (rightMax - rightMin)
    image_height : float
        Height of bounding box (upMax - upMin)
    mask : numpy.ndarray
        Image mask for drawing HNN flow lines
    frame2 : numpy.ndarray
        Current frame image for drawing center markers
    upMin, rightMin, upMax, rightMax : float
        Bounding box coordinates (y_min, x_min, y_max, x_max)

    Returns:
    --------
    tuple
        (res, resHnn)
        - res: Optical flow results [cell_num, center_x, center_y, avg_dx, avg_dy]
        - resHnn: HNN predictions [cell_num, pred_x, pred_y, dx_hnn, dy_hnn]
    """
    # Compute flow vectors (displacement: old - new)
    # Note: Typically flow is new - old; check if this matches your requirements
    flow_vectors = np.array(old_points) - np.array(new_points)

    # Initialize 3x3 grid structures
    grid_flow = [[[] for _ in range(3)] for _ in range(3)]  # Flow vectors per cell
    grid_flowPoints = [[[] for _ in range(3)] for _ in range(3)]  # New points per cell
    grid_flowPointsOld = [[[] for _ in range(3)] for _ in range(3)]  # Old points per cell

    # Calculate cell dimensions
    cell_width = image_width / 3
    cell_height = image_height / 3

    # Avoid division by zero for empty bounding boxes
    if cell_height == 0:
        cell_height = 0.0001
    if cell_width == 0:
        cell_width = 0.0001
    if upMin == 0:
        upMin = 0.0001
    if rightMin == 0:
        rightMin = 0.0001

    # Assign each point to appropriate grid cell based on old point position
    for (x, y), (dx, dy), (nx, ny) in zip(old_points, flow_vectors, new_points):
        # Determine grid indices (0, 1, or 2)
        col = min(int((x - rightMin) // cell_width), 2)
        row = min(int((y - upMin) // cell_height), 2)

        # Store in corresponding grid cell
        grid_flow[row][col].append((dx, dy))
        grid_flowPoints[row][col].append((nx, ny))
        grid_flowPointsOld[row][col].append((x, y))

    # Initialize results containers
    avg_grid_flow = np.zeros((3, 3, 2))  # 3x3x2 array for average (dx, dy)
    res = []  # Optical flow results
    resHnn = []  # HNN prediction results

    fragmentNum = 0  # Cell counter (1-9)

    # Process each grid cell
    for row in range(3):
        for col in range(3):
            fragmentNum += 1

            # Calculate current cell center
            a = rightMin + (image_width / 6) * (2 * col + 1)  # x-center
            b = upMin + (image_height / 6) * (2 * row + 1)  # y-center

            # Initialize oldGrid if empty (first frame)
            if len(oldGrid[row][col]) == 0:
                oldGrid[row][col] = [a, b]
                print(f"Initialized oldGrid[{row}][{col}]: [{a}, {b}]")
                print(oldGrid)
                continue

            # Get previous center from oldGrid
            aold = oldGrid[row][col][0]
            bold = oldGrid[row][col][1]

            # Update oldGrid with current center
            oldGrid[row][col] = [a, b]

            # Process only if cell has flow vectors
            if grid_flow[row][col]:
                # Calculate average optical flow for this cell
                avg_dx = np.mean([f[0] for f in grid_flow[row][col]])
                avg_dy = np.mean([f[1] for f in grid_flow[row][col]])
                avg_grid_flow[row, col] = [avg_dx, avg_dy]

                # Add to optical flow results
                res.append([fragmentNum, a, b, avg_dx, avg_dy])

                # HNN PREDICTION
                # Input: current position (a,b) and delta from previous position
                xhnn, yhnn = HNNCleanPredict(a, b, (a - aold), (b - bold), False)

                # Convert PyTorch tensors to numpy arrays
                xhnn = xhnn.detach().numpy()
                yhnn = yhnn.detach().numpy()

                # Calculate HNN displacement
                dx_hnn = a - xhnn
                dy_hnn = b - yhnn

                # Add to HNN results
                resHnn.append([fragmentNum, xhnn, yhnn, dx_hnn, dy_hnn])

                # VISUALIZATION
                # Draw HNN-predicted flow line (magenta) on mask
                mask = cv2.line(mask, (int(a), int(b)), (int(xhnn), int(yhnn)),
                                (120, 120, 255), 2)

                # Draw current center (blue circle) on frame
                frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (255, 0, 0), -1)

    return res, resHnn


# ============================================================================
# OPTICAL FLOW COMPUTATION WITH FEATURE TRACKING
# ============================================================================

def calcOpFlow(frame1_path, frame2_path):
    """
    Compute sparse optical flow between two frames and generate HNN predictions.

    Uses Shi-Tomasi corner detection and Lucas-Kanade optical flow.
    Saves visualization images and returns flow/HNN results.

    Parameters:
    -----------
    frame1_path : str
        Path to previous frame image
    frame2_path : str
        Path to current frame image

    Returns:
    --------
    tuple
        (res, resHnn) - see calculate_grid_flow return values
        Empty lists if error occurs or no features found
    """
    global indeximg

    # Load frames
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    if frame1 is None or frame2 is None:
        print(f"Error loading frames: {frame1_path}, {frame2_path}")
        return [], []

    # Convert to grayscale for optical flow computation
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # FEATURE DETECTION (Shi-Tomasi corners)
    feature_params = dict(
        maxCorners=100,  # Maximum number of corners to detect
        qualityLevel=0.3,  # Minimum quality of corners (0-1)
        minDistance=7,  # Minimum distance between corners
        blockSize=7  # Size of neighborhood for corner detection
    )

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    if prev_pts is None:
        print("No features detected in frame")
        return [], []

    # OPTICAL FLOW COMPUTATION (Lucas-Kanade)
    try:
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, next_gray, prev_pts, None,
            winSize=(15, 15),  # Search window size
            maxLevel=2,  # Pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    except Exception as e:
        print(f"Error in optical flow calculation: {e}")
        return [], []

    # Filter only successfully tracked points
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    if good_new.size == 0:
        print("No good feature matches found")
        return [], []

    # Create mask for visualization
    mask = np.zeros_like(frame1)

    # Calculate angles and magnitudes (for potential use, not returned)
    angles = []
    magnitudes = []
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()  # Current point
        c, d = old.ravel()  # Previous point
        res = calcAngleMag(a, b, c, d)
        angles.append(res[0])
        magnitudes.append(res[1])

    # Calculate bounding box around all tracked points
    combinedX = [pair[0] for pair in good_new] + [pair[0] for pair in good_old]
    combinedY = [pair[1] for pair in good_new] + [pair[1] for pair in good_old]
    upMin = min(combinedY)
    upMax = max(combinedY)
    rightMax = max(combinedX)
    rightMin = min(combinedX)

    # Calculate grid-based flow with HNN predictions
    res, resHnn = calculate_grid_flow(
        good_old, good_new,
        abs(rightMax - rightMin), abs(upMax - upMin),
        mask, frame2,
        upMin, rightMin, upMax, rightMax
    )

    # Combine mask with frame for visualization
    output = cv2.add(frame2, mask)

    # Display visualization
    cv2.imshow('Sparse Optical Flow with HNN', output)

    # SAVE OUTPUT IMAGE
    save_dir = videoSafeDir
    filename = str(indeximg) + 'output_image.jpg'
    save_path = os.path.join(save_dir, filename)
    print(f"Attempting to save to: {save_path}")

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Save image and verify
    success = cv2.imwrite(save_path, output)
    if success:
        print(f"✅ Image successfully saved to: {save_path}")
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"✅ File exists! Size: {file_size} bytes")
        else:
            print("❌ File was not created!")
    else:
        print("❌ Failed to save image!")

    # Increment image counter for next save
    indeximg += 1

    # Display for 500ms (0.5 seconds)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    return res, resHnn


# ============================================================================
# BATCH PROCESSING FOR VIDEO AND MASK COMBINATIONS
# ============================================================================

def runFlowForAll(videoNum, maskNum):
    """
    Process all frames for a specific video and mask, generating training data.

    This is the main processing pipeline that:
    1. Loads frame sequences and bounding box coordinates
    2. Computes optical flow and HNN predictions for consecutive frames
    3. Calculates importance scores for each frame pair
    4. Saves results to CSV training file

    Parameters:
    -----------
    videoNum : str
        Video identifier (e.g., "4" for videos/4/)
    maskNum : str
        Mask identifier (e.g., "1" for mask1/)

    Returns:
    --------
    None
        Saves training data to 'trainDataHnn3step.csv' in mask directory
    """
    # Load first frame to get image dimensions
    frameStart = cv2.imread("videos" + videoNum + "/Frames/0000.jpg")
    if frameStart is None:
        print(f"Error loading first frame for video {videoNum}")
        return
    size = frameStart.shape[:2]  # (height, width)

    # Count total frames in video (subtract 2 for edge cases)
    countFrames = count_image_files("videos" + videoNum + "/Frames") - 2
    if countFrames <= 0:
        print(f"No frames found for video {videoNum}")
        return

    # Directory containing masked frames
    video_dir = "videos" + videoNum + "/mask" + maskNum

    # LOAD BOUNDING BOX COORDINATES
    coordinates = []
    try:
        with open(video_dir + '/coordinates.csv', 'r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header

            # Read each row: upMin, rightMin, upMax, rightMax
            for row in reader:
                upMin, rightMin, upMax, rightMax = map(float, row)
                coordinates.append((upMin, rightMin, upMax, rightMax))
    except FileNotFoundError:
        print(f"Coordinates file not found: {video_dir}/coordinates.csv")
        return
    except Exception as e:
        print(f"Error reading coordinates: {e}")
        return

    # Verify directory exists
    if not os.path.exists(video_dir):
        print(f"Directory {video_dir} does not exist!")
        return

    # Get list of all JPEG frames in mask directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if p.lower().endswith(('.jpg', '.jpeg'))
    ]

    if len(frame_names) < 2:
        print(f"Insufficient frames in {video_dir}: {len(frame_names)}")
        return

    # Initialize training data storage
    trainData = []

    # PROCESS EACH CONSECUTIVE FRAME PAIR
    for i in range(len(frame_names) - 1):
        print(f"Processing frames: {frame_names[i]}, {frame_names[i + 1]} (frame {frame_names[i][6:8]})")

        # Compute optical flow and HNN predictions
        cap, capHnn = calcOpFlow(
            video_dir + '/' + frame_names[i],
            video_dir + '/' + frame_names[i + 1]
        )

        # Extract frame number from filename (assuming format 'frameXXXX.jpg')
        frameNum = int(frame_names[i][6:8]) + 0.0001  # Small offset

        # CALCULATE IMPORTANCE SCORE
        # Score combines temporal, spatial, and centrality factors

        # Temporal weight: (current_frame / total_frames)²
        # Gives more weight to later frames
        temporal_weight = pow(frameNum, 2) / pow(countFrames, 2)

        # Spatial weight: region_area / image_area
        # Gives more weight to larger regions
        region_area = coordinates[i][2] * coordinates[i][3]  # height * width
        image_area = size[0] * size[1]  # height * width
        spatial_weight = region_area / image_area

        # Centrality: 1 - (distance_from_center / max_possible_distance)
        # Gives more weight to regions near image center
        centerPoint = [
            coordinates[i][0] + coordinates[i][2] / 2,  # y_center
            coordinates[i][1] + coordinates[i][3] / 2  # x_center
        ]
        image_center = [size[1] / 2, size[0] / 2]  # (x_center, y_center)
        max_distance = euclidean_distance([0, 0], image_center)
        centrality = 1 - (euclidean_distance(centerPoint, image_center) / max_distance)

        # Combine weights into final score
        score = temporal_weight * spatial_weight * centrality

        # Append to training data
        # Structure: [frameNum, coordinates, hnn_predictions, optical_flow, score]
        trainData.append([frameNum, coordinates[i], capHnn, cap, score])

    # SAVE TRAINING DATA TO CSV
    output_csv = video_dir + '/trainDataHnn3step.csv'
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['frameNum', 'coordinates', 'hnncoordinates', 'cap', 'score'])

        # Write data rows
        for row in trainData:
            writer.writerow(row)

    print(f"Training data saved to {output_csv} ({len(trainData)} rows)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Example: Process video 2, mask 1
runFlowForAll(str(2), str(1))
exit()

# Batch processing for multiple videos and masks
# Note: Currently commented out as the script exits after single example

for vid in range(22):  # Process videos 0 to 21
    for mask in range(10):  # Process masks 0 to 9 for each video
        coord_file = f"videos{vid}/mask{mask}/coordinates.csv"

        # Only process if coordinates file exists
        if os.path.exists(coord_file):
            print(f"\n--- Processing video {vid}, mask {mask} ---")
            runFlowForAll(str(vid), str(mask))
        else:
            print(f"Skipping video {vid}, mask {mask}: no {coord_file}")