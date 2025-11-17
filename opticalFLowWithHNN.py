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

# Get the script's directory and list all files for debugging/logging
script_dir = os.path.dirname(os.path.abspath(__file__))
files = os.listdir(script_dir)
from codeFromPaperHnn.Config import paths

#This script copies one column from a dataset to another dataset

videoSafeDir = paths['videoSafeFolder']
print(f"Files in script directory ({script_dir}): {files}")

"""
This script extends the optical flow processing with Hamiltonian Neural Network (HNN) predictions.
It computes sparse optical flow between consecutive frames, divides motion into a 3x3 grid,
averages flow per cell, and uses HNN to predict future positions based on historical movements.
Visualizations are saved as images, and training data (including HNN predictions) is exported to CSV.

Key enhancements over base flow:
- Tracks previous grid cell centers in 'oldGrid' to compute deltas for HNN input.
- HNNCleanPredict is called per cell to forecast next position from current and delta.
- Outputs both optical flow averages and HNN-predicted displacements per cell.
- Processes multiple videos and masks in batch mode.

Configuration:
- Videos: 0-21 (skipped if no coordinates.csv)
- Masks: 0-9 per video
- Outputs: trainDataHnn3step.csv per mask dir, and numbered output images in a fixed save dir.
"""

# Global index for naming saved output images
indeximg = 0

# Global 3x3 grid to track previous center positions [x, y] for each cell (row-major)
oldGrid = [[[], [], []], [[], [], []], [[], [], []]]


def euclidean_distance(point1, point2):
    """
    Compute the Euclidean distance between two 2D points.

    Parameters:
    -----------
    point1 : tuple of two floats
        First point (x1, y1).
    point2 : tuple of two floats
        Second point (x2, y2).

    Returns:
    --------
    float
        The Euclidean distance between the points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def count_image_files(directory):
    """
    Count the number of image files in a directory.

    Supports common image extensions: .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp.

    Parameters:
    -----------
    directory : str
        Path to the directory to scan.

    Returns:
    --------
    int
        Number of image files found. Returns 0 if directory not found or on error.

    Raises:
    -------
    Prints error messages to console for file system issues.
    """
    # List of common image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    # Initialize counter
    image_count = 0

    try:
        # Iterate through all files in the directory
        for file in os.listdir(directory):
            # Check if the file has an image extension
            if os.path.isfile(os.path.join(directory, file)) and \
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
    Calculate the angle (in degrees, 0-180 range) and magnitude of the vector from (x1,y1) to (x2,y2).

    Parameters:
    -----------
    x1, y1 : float
        Starting point coordinates.
    x2, y2 : float
        Ending point coordinates.

    Returns:
    --------
    list of two floats
        [angle_degrees, magnitude]
    """
    xDiff = x1 - x2
    yDiff = y1 - y2
    angle_rad = math.atan2(yDiff, xDiff)  # Compute angle in radians
    # Convert to degrees
    angle_deg = math.degrees(angle_rad)
    # Normalize to 0-180 range (note: %180 keeps it in 0-180)
    angle = angle_deg % 180
    mag = math.sqrt(xDiff * xDiff + yDiff * yDiff)  # Magnitude (Euclidean distance)
    return [angle, mag]


def calculate_grid_flow(old_points, new_points, image_width, image_height, mask, frame2, upMin, rightMin, upMax,
                        rightMax):
    """
    Calculate average optical flow and HNN-predicted flow in a 3x3 grid based on point displacements.

    Divides the image into 3x3 cells, computes average (dx, dy) flow for points in each cell,
    tracks previous centers in oldGrid, computes delta from prior, and uses HNN to predict next position.
    Draws HNN-predicted flow lines on mask and centers on frame.

    Parameters:
    -----------
    old_points : list of tuples
        List of (x, y) coordinates of points in the previous frame.
    new_points : list of tuples
        List of (x, y) coordinates of points in the current frame.
    image_width : float
        Effective width of the bounding box (rightMax - rightMin).
    image_height : float
        Effective height of the bounding box (upMax - upMin).
    mask : numpy.ndarray
        Image mask for drawing HNN flow lines.
    frame2 : numpy.ndarray
        Current frame for drawing center circles.
    upMin, rightMin, upMax, rightMax : float
        Bounding box coordinates (y_min, x_min, y_max, x_max).

    Returns:
    --------
    tuple of two lists
        - res: List of [fragmentNum (1-9), center_x, center_y, avg_dx, avg_dy] for non-empty cells (optical flow).
        - resHnn: List of [fragmentNum, pred_x, pred_y, dx_hnn, dy_hnn] for cells with HNN predictions.
    """
    # Calculate flow vectors (displacement: old - new; note: typically new - old for forward flow)
    flow_vectors = np.array(old_points) - np.array(new_points)

    # Initialize 3x3 grid structures for flow vectors, new points, and old points
    grid_flow = [[[] for _ in range(3)] for _ in range(3)]
    grid_flowPoints = [[[] for _ in range(3)] for _ in range(3)]
    grid_flowPointsOld = [[[] for _ in range(3)] for _ in range(3)]

    # Calculate cell dimensions
    cell_width = image_width / 3
    cell_height = image_height / 3
    if cell_height == 0:
        cell_height = 0.0001  # Avoid division by zero
    if cell_width == 0:
        cell_width = 0.0001  # Avoid division by zero
    if upMin == 0:
        upMin = 0.0001  # Avoid offset issues
    if rightMin == 0:
        rightMin = 0.0001  # Avoid offset issues

    # Assign each flow vector and points to grid cells based on old point positions
    for (x, y), (dx, dy), (nx, ny) in zip(old_points, flow_vectors, new_points):
        # Determine grid cell indices (0, 1, or 2) based on position relative to mins
        col = min(int((x - rightMin) // cell_width), 2)
        row = min(int((y - upMin) // cell_height), 2)
        # Store in corresponding grid cell
        grid_flow[row][col].append((dx, dy))
        grid_flowPoints[row][col].append((nx, ny))
        grid_flowPointsOld[row][col].append((x, y))

    # Calculate averages and HNN predictions for each grid cell
    avg_grid_flow = np.zeros((3, 3, 2))  # 3x3 grid, each with (avg_dx, avg_dy)
    res = []  # Optical flow results for non-empty cells
    resHnn = []  # HNN prediction results
    fragmentNum = 0
    for row in range(3):
        for col in range(3):
            fragmentNum += 1
            # Center of the cell for current position
            a = rightMin + (image_width / 6) * (2 * col + 1)  # x-center
            b = upMin + (image_height / 6) * (2 * row + 1)  # y-center

            # Initialize oldGrid for this cell if empty
            if len(oldGrid[row][col]) == 0:
                oldGrid[row][col] = [a, b]
                print(f"Initialized oldGrid[{row}][{col}]: [{a}, {b}]")
                print(oldGrid)
                continue

            # Get previous center from oldGrid
            aold = oldGrid[row][col][0]
            bold = oldGrid[row][col][1]
            # Update oldGrid to current center
            oldGrid[row][col] = [a, b]

            if grid_flow[row][col]:  # If cell has flow vectors
                # Average optical flow
                avg_dx = np.mean([f[0] for f in grid_flow[row][col]])
                avg_dy = np.mean([f[1] for f in grid_flow[row][col]])
                avg_grid_flow[row, col] = [avg_dx, avg_dy]
                res.append([fragmentNum, a, b, avg_dx, avg_dy])

                # HNN prediction: Input current pos (a,b) and delta from old (a-aold, b-bold)
                xhnn, yhnn = HNNCleanPredict(a, b, (a - aold), (b - bold), False)
                xhnn = xhnn.detach().numpy()  # Convert tensor to numpy
                yhnn = yhnn.detach().numpy()

                # HNN displacement
                dx_hnn = a - xhnn
                dy_hnn = b - yhnn
                resHnn.append([fragmentNum, xhnn, yhnn, dx_hnn, dy_hnn])

                # Draw HNN-predicted flow line on mask (from current to predicted)
                mask = cv2.line(mask, (int(a), int(b)), (int(xhnn), int(yhnn)), (120, 120, 255), 2)
                # Draw current center circle on frame
                frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (255, 0, 0), -1)

    return res, resHnn


def calcOpFlow(frame1_path, frame2_path):  # Fixed typo: was 'caclOpFlow'
    """
    Compute sparse optical flow and HNN predictions between two frames using Shi-Tomasi + Lucas-Kanade.

    Detects features in frame1, tracks to frame2, filters good matches, computes grid flow and HNN.
    Saves visualization image to a fixed directory and returns results.

    Parameters:
    -----------
    frame1_path : str
        Path to the previous frame image.
    frame2_path : str
        Path to the current frame image.

    Returns:
    --------
    tuple of two lists
        - res: Optical flow grid results.
        - resHnn: HNN prediction grid results.
        Empty lists on error/no points.
    """
    global indeximg

    # Load frames
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    if frame1 is None or frame2 is None:
        print(f"Error loading frames: {frame1_path}, {frame2_path}")
        return [], []

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

    if prev_pts is None:
        return [], []

    # Calculate optical flow
    try:
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, next_gray, prev_pts, None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    except Exception as e:
        print(f"Error in optical flow calculation: {e}")
        return [], []

    # Filter only good points (status == 1)
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    if good_new.size == 0:
        return [], []

    # Create a mask image for drawing
    mask = np.zeros_like(frame1)

    # Compute angles and magnitudes for all good points (computed but not used in return)
    angles = []
    magnitudes = []
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        res = calcAngleMag(a, b, c, d)
        angles.append(res[0])
        magnitudes.append(res[1])

    # Combine x and y coordinates to find bounding box
    combinedX = [pair[0] for pair in good_new] + [pair[0] for pair in good_old]
    combinedY = [pair[1] for pair in good_new] + [pair[1] for pair in good_old]
    upMin = min(combinedY)
    upMax = max(combinedY)
    rightMax = max(combinedX)
    rightMin = min(combinedX)

    # Calculate grid flow and HNN predictions using the bounding box dimensions
    res, resHnn = calculate_grid_flow(
        good_old, good_new, abs(rightMax - rightMin), abs(upMax - upMin),
        mask, frame2, upMin, rightMin, upMax, rightMax
    )

    # Overlay mask on frame2 for visualization
    output = cv2.add(frame2, mask)

    # Display briefly (waitKey(0) for manual close)
    cv2.imshow('Sparse Optical Flow with HNN', output)

    # Save output image to fixed directory
    save_dir = videoSafeDir
    filename = str(indeximg) + 'output_image.jpg'
    save_path = os.path.join(save_dir, filename)
    print(f"Attempting to save to: {save_path}")

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Save and verify
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

    indeximg += 1
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    return res, resHnn


def runFlowForAll(videoNum, maskNum):  # Renamed for clarity: was 'runFloeForall'
    """
    Process all consecutive frame pairs for a given video and mask, compute flow + HNN, and save to CSV.

    Loads bounding box coordinates from CSV, computes optical flow and HNN predictions per pair,
    calculates a normalized score, and appends to training data.

    Parameters:
    -----------
    videoNum : str
        Video identifier (e.g., "4" for videos/4/).
    maskNum : str
        Mask identifier (e.g., "1" for mask1/).

    Returns:
    --------
    None
        Saves 'trainDataHnn3step.csv' in the mask directory.
    """
    # Load the first frame to determine image size (height, width, channels)
    frameStart = cv2.imread("videos" + videoNum + "/Frames/0000.jpg")
    if frameStart is None:
        print(f"Error loading first frame for video {videoNum}")
        return
    size = frameStart.shape[:2]  # (height, width)

    # Count total frames (subtract 2 for padding/edge cases)
    countFrames = count_image_files("videos" + videoNum + "/Frames") - 2
    if countFrames <= 0:
        print(f"No frames found for video {videoNum}")
        return

    # Directory for masked video frames
    video_dir = "videos" + videoNum + "/mask" + maskNum

    # Load bounding box coordinates from CSV (one per frame: upMin, rightMin, upMax, rightMax)
    coordinates = []
    try:
        with open(video_dir + '/coordinates.csv', 'r', newline='') as file:
            reader = csv.reader(file)
            # Skip the header row
            header = next(reader)
            # Read each row
            for row in reader:
                # Convert strings to floats
                upMin, rightMin, upMax, rightMax = map(float, row)
                coordinates.append((upMin, rightMin, upMax, rightMax))
    except FileNotFoundError:
        print(f"Coordinates file not found: {video_dir}/coordinates.csv")
        return
    except Exception as e:
        print(f"Error reading coordinates: {e}")
        return

    if not os.path.exists(video_dir):
        print(f"Directory {video_dir} does not exist!")
        return

    # Scan all JPEG frame names in the mask directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if p.lower().endswith(('.jpg', '.jpeg'))
    ]

    if len(frame_names) < 2:
        print(f"Insufficient frames in {video_dir}: {len(frame_names)}")
        return

    # Initialize list to store training data
    trainData = []

    # Process each pair of consecutive frames
    for i in range(len(frame_names) - 1):
        print(f"Processing frames: {frame_names[i]}, {frame_names[i + 1]} (frame {frame_names[i][6:8]})")

        # Compute optical flow and HNN between consecutive frames
        cap, capHnn = calcOpFlow(
            video_dir + '/' + frame_names[i],
            video_dir + '/' + frame_names[i + 1]
        )

        # Extract frame number from filename (assuming format like 'frameXXXX.jpg', extract XX)
        frameNum = int(frame_names[i][6:8]) + 0.0001  # Small offset to avoid integer issues

        # Compute normalized score:
        # - Temporal weight: (frameNum / countFrames)^2
        # - Spatial weight: (region_area / total_image_area)
        # - Centrality: 1 - (distance_from_center / max_distance)
        region_area = coordinates[i][2] * coordinates[i][3]  # height * width
        image_area = size[0] * size[1]  # height * width
        centerPoint = [
            coordinates[i][0] + coordinates[i][2] / 2,  # y_center
            coordinates[i][1] + coordinates[i][3] / 2  # x_center
        ]
        image_center = [size[1] / 2, size[0] / 2]  # (x_center, y_center)
        max_distance = euclidean_distance([0, 0], image_center)
        centrality = 1 - (euclidean_distance(centerPoint, image_center) / max_distance)

        score = (
                (pow(frameNum, 2) / pow(countFrames, 2)) *
                (region_area / image_area) *
                centrality
        )

        # Append row: [frameNum, coordinates_tuple, hnn_results, flow_results, score]
        trainData.append([frameNum, coordinates[i], capHnn, cap, score])

    # Save training data to CSV
    output_csv = video_dir + '/trainDataHnn3step.csv'
    '''with open(output_csv, 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        # Write header
        writer.writerow(['frameNum', 'coordinates', 'hnncoordinates', 'cap', 'score'])
        # Write each row (CSV handles nested lists as strings)
        for row in trainData:
            writer.writerow(row)'''

    print(f"Training data saved to {output_csv} ({len(trainData)} rows)")


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