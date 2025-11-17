import cv2
import numpy as np
import os
import math
import csv
from codeFromPaperHnn.utils import choose_nonlinearity
from codeFromPaperHnn.nn_models import MLP
from codeFromPaperHnn.nn_models import *
from codeFromPaperHnn.hnn import *
from codeFromPaperHnn.TrainedModel import HNNPredict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

"""
This script processes video frames to compute sparse optical flow using OpenCV,
analyzes motion in masked regions, and generates training data for a model.
It calculates grid-based flow averages, angles, magnitudes, and scores based on
frame position and region centrality. Outputs are saved to CSV files.

Key steps:
1. Load initial frame to determine image dimensions.
2. Count available frames in the directory.
3. For each pair of consecutive masked frames:
   - Compute sparse optical flow (Shi-Tomasi corners + Lucas-Kanade).
   - Calculate angle and magnitude of displacements.
   - Divide flow into a 3x3 grid and compute average vectors per cell.
   - Compute a normalized score incorporating frame progress, region size,
      and centrality.
4. Save results (frame number, coordinates, grid flow, score) to CSV.
"""

# Configuration: Specify video and mask numbers
videoNum = "4"  # Video identifier (e.g., "4" for videos/4/)
maskNum = "1"  # Mask identifier (e.g., "1" for mask1/)

# Load the first frame to determine image size (height, width, channels)
frameStart = cv2.imread("videos" + videoNum + "/Frames/0000.jpg")
size = frameStart.shape[:2]  # Extract height and width as a tuple (height, width)


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


# Count the total number of frames in the Frames directory (subtract 2 for padding/edge cases)
countFrames = count_image_files("videos" + videoNum + "/Frames") - 2


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
    # Normalize to 0-360 range
    angle = angle_deg % 180
    mag = math.sqrt(xDiff * xDiff + yDiff * yDiff)  # Magnitude (Euclidean distance)
    return [angle, mag]


def calculate_grid_flow(old_points, new_points, image_width, image_height, mask, frame2, upMin, rightMin, upMax,
                        rightMax):
    """
    Calculate average optical flow in a 3x3 grid based on point displacements.

    Divides the image into 3x3 cells and computes average (dx, dy) flow for points in each cell.
    Also draws visualization on the mask and frame (lines for flow, circles for centers).

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
        Image mask for drawing flow lines.
    frame2 : numpy.ndarray
        Current frame for drawing center circles.
    upMin, rightMin, upMax, rightMax : float
        Bounding box coordinates (y_min, x_min, y_max, x_max).

    Returns:
    --------
    list of lists
        Each sublist: [fragmentNum (1-9), center_x, center_y, avg_dx, avg_dy] for non-empty cells.
    """
    # Calculate flow vectors (displacement: old - new? Note: typically new - old for forward flow)
    flow_vectors = np.array(old_points) - np.array(new_points)

    # Initialize grid structures: 3x3 lists for flow vectors and points
    grid_flow = [[[] for _ in range(3)] for _ in range(3)]
    grid_flowPoints = [[[] for _ in range(3)] for _ in range(3)]

    # Calculate cell dimensions
    cell_width = image_width / 3
    cell_height = image_height / 3
    if cell_height == 0:
        cell_height = 0.0001  # Avoid division by zero
    if cell_width == 0:
        cell_width = 0.0001  # Avoid division by zero
    if upMin == 0:
        upMin = 0.0001  # Avoid offset issues

    # Assign each flow vector to grid cells based on old point positions
    for (x, y), (dx, dy), (nx, ny) in zip(old_points, flow_vectors, new_points):
        # Determine grid cell indices (0, 1, or 2) based on position relative to mins
        col = min(int((x - rightMin) // cell_width), 2)
        row = min(int((y - upMin) // cell_height), 2)
        # Store flow vector and new point in corresponding grid cell
        grid_flow[row][col].append((dx, dy))
        grid_flowPoints[row][col].append((nx, ny))

    # Calculate average flow for each grid cell
    avg_grid_flow = np.zeros((3, 3, 2))  # 3x3 grid, each with (avg_dx, avg_dy)
    res = []  # Results list for non-empty cells
    fragmentNum = 0
    for row in range(3):
        for col in range(3):
            fragmentNum += 1
            # Center of the cell for visualization
            a = rightMin + (image_width / 6) * (2 * col + 1)  # x-center
            b = upMin + (image_height / 6) * (2 * row + 1)  # y-center
            if grid_flow[row][col]:  # If cell has flow vectors
                avg_dx = np.mean([f[0] for f in grid_flow[row][col]])
                avg_dy = np.mean([f[1] for f in grid_flow[row][col]])
                avg_grid_flow[row, col] = [avg_dx, avg_dy]
                res.append([fragmentNum, a, b, avg_dx, avg_dy])
                # Draw flow line on mask
                mask = cv2.line(mask, (int(a), int(b)), (int(a + avg_dx), int(b + avg_dy)), (120, 120, 255), 2)
                # Draw center circle on frame
                frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (255, 0, 0), -1)

    return res


def calcOpFlow(frame1_path, frame2_path):  # Fixed typo: was 'caclOpFlow'
    """
    Compute sparse optical flow between two frames using Shi-Tomasi corners and Lucas-Kanade method.

    Detects features in frame1, tracks them to frame2, filters good matches, and computes grid flow.
    Displays the result briefly and returns the grid flow results.

    Parameters:
    -----------
    frame1_path : str
        Path to the previous frame image.
    frame2_path : str
        Path to the current frame image.

    Returns:
    --------
    list of lists
        Grid flow results from calculate_grid_flow, or empty list on error/no points.
    """
    # Load frames
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    if frame1 is None or frame2 is None:
        print(f"Error loading frames: {frame1_path}, {frame2_path}")
        return []

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
        return []

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
        return []

    # Filter only good points (status == 1)
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    if good_new.size == 0:
        return []

    # Create a mask image for drawing tracks
    mask = np.zeros_like(frame1)

    # Compute angles and magnitudes for all good points (though not directly used in return)
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

    # Calculate grid flow using the bounding box dimensions
    res = calculate_grid_flow(
        good_old, good_new, abs(rightMax - rightMin), abs(upMax - upMin),
        mask, frame2, upMin, rightMin, upMax, rightMax
    )

    # Overlay mask on frame2 for visualization
    output = cv2.add(frame2, mask)

    # Display the result (briefly for inspection)
    cv2.imshow('Sparse Optical Flow', output)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    return res


# Directory for masked video frames
video_dir = "videos" + videoNum + "/mask" + maskNum

# Load bounding box coordinates from CSV (one per frame: upMin, rightMin, upMax, rightMax)
coordinates = []
if os.path.exists(video_dir + '/coordinates.csv'):
    with open(video_dir + '/coordinates.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header row
        header = next(reader)
        # Read each row of coordinates
        for row in reader:
            # Convert strings to floats
            upMin, rightMin, upMax, rightMax = map(float, row)
            coordinates.append((upMin, rightMin, upMax, rightMax))
else:
    print(f"Coordinates file not found: {video_dir}/coordinates.csv")
    coordinates = []  # Empty if file missing

if not os.path.exists(video_dir):
    print(f"Directory {video_dir} does not exist!")
    exit(1)  # Exit if directory missing

# Scan all JPEG frame names in the mask directory
frame_names = [
    p for p in os.listdir(video_dir)
    if p.lower().endswith(('.jpg', '.jpeg'))
]

# Initialize list to store training data
trainData = []

# Process each pair of consecutive frames
for i in range(len(frame_names) - 1):
    print(f"Processing frames: {frame_names[i]}, {frame_names[i + 1]} (frame {frame_names[i][6:8]})")

    # Compute optical flow between consecutive frames
    cap = calcOpFlow(
        video_dir + '/' + frame_names[i],
        video_dir + '/' + frame_names[i + 1]
    )

    # Extract frame number from filename (assuming format like 'frameXXXX.jpg', extract XX)
    frameNum = int(frame_names[i][6:8]) + 0.0001  # Small offset to avoid integer issues

    # Compute center point of the bounding box
    centerPoint = [
        coordinates[i][0] + coordinates[i][2] / 2,  # y_center = upMin + height/2
        coordinates[i][1] + coordinates[i][3] / 2  # x_center = rightMin + width/2
    ]

    # Compute normalized score:
    # - Temporal weight: (frameNum / countFrames)^2
    # - Spatial weight: (region_area / total_image_area)
    # - Centrality: 1 - (distance_from_center / max_distance)
    region_area = coordinates[i][2] * coordinates[i][3]  # height * width
    image_area = size[0] * size[1]  # height * width
    image_center = [size[1] / 2, size[0] / 2]  # (x_center, y_center) Note: size[0]=height (y), size[1]=width (x)
    max_distance = euclidean_distance([0, 0], image_center)
    centrality = 1 - (euclidean_distance(centerPoint, image_center) / max_distance)

    score = (
            (pow(frameNum, 2) / pow(countFrames, 2)) *
            (region_area / image_area) *
            centrality
    )

    # Append row: [frameNum, coordinates_tuple, grid_flow_list, score]
    trainData.append([frameNum, coordinates[i], cap, score])

# Save training data to CSV
output_csv = video_dir + '/trainData.csv'
with open(output_csv, 'w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    # Write header
    writer.writerow(['frameNum', 'coordinates', 'cap', 'score'])
    # Write each row
    for row in trainData:
        writer.writerow(row)

print(f"Training data saved to {output_csv}")
print(f"Processed {len(trainData)} frame pairs out of {len(frame_names) - 1} available.")