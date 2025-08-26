import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
from matplotlib.patches import Rectangle
matplotlib.use('Qt5Agg')
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

import torch
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
import torch
print(torch.version.cuda)  # Returns None if CPU-only
import cv2

vidoNum = "17"
start = "72"
frNum = "2"


if device.type == "cuda":
    print("stana")
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "configs/sam2/sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def create_next_mask_folder(base_path):
    # Get list of all folders in the base path
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    # Filter folders that match the 'mask' pattern and extract numbers
    mask_folders = [f for f in folders if f.startswith('mask') and f[4:].isdigit()]

    if not mask_folders:
        # If no mask folders exist, create mask1
        next_number = 1
    else:
        # Extract numbers from mask folders and find the highest
        numbers = [int(f[4:]) for f in mask_folders]
        next_number = max(numbers) + 1

    # Create new folder name
    new_folder = f"mask{next_number}"
    new_folder_path = os.path.join(base_path, new_folder)

    # Create the new folder
    os.makedirs(new_folder_path)
    print(f"Created folder: {new_folder_path}")

    return new_folder_path
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

video_dir = "videos"  + vidoNum + "/Frames" + frNum # Better to not include trailing slash
print(video_dir)

output_dir = create_next_mask_folder("videos"  + vidoNum)
# Ensure the directory exists
if not os.path.exists(video_dir):
    print(f"Directory {video_dir} does not exist!")
    frame_names = []
else:
    # Scan all JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if p.lower().endswith(('.jpg', '.jpeg'))
    ]

print(f"Found {len(frame_names)} JPEG frames in {video_dir}")
print(frame_names)
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
print(frame_names)
# take a look the first video frame
'''frame_idx = int(start)
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
print(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
plt.show()'''


inference_state = predictor.init_state(video_path=video_dir)

predictor.reset_state(inference_state)


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)


############################################
# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
box = np.array([300, 0, 500, 400], dtype=np.float32)
drawing = False  # True if mouse is pressed
ix, iy = -1, -1  # Initial coordinates
x_min, y_min, x_max, y_max = -1, -1, -1, -1  # Bounding box coordinates
img = None  # Image to draw on
img_copy = None  # Copy of the image to reset drawing

def draw_bounding_box(event, x, y, flags, param):
    global ix, iy, x_min, y_min, x_max, y_max, drawing, img, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Update the rectangle while dragging
        img = img_copy.copy()  # Reset to original image
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)  # Draw rectangle (green, thickness 2)
        cv2.imshow("Select Bounding Box", img)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing
        drawing = False
        x_min, x_max = min(ix, x), max(ix, x)
        y_min, y_max = min(iy, y), max(iy, y)
        # Draw final rectangle
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow("Select Bounding Box", img)
        print(f"Selected box: (x_min, y_min, x_max, y_max) = ({x_min}, {y_min}, {x_max}, {y_max})")

def get_bounding_box(image_path):
    global img, img_copy, x_min, y_min, x_max,\
        y_max

    # Load the image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]  # For color images (H, W, channels)
    # If grayscale: image.shape returns (H, W)

    print(f"Width (pixels): {width}")
    print(f"Height (pixels): {height}")
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    img_copy = img.copy()


    # Create a window and set the mouse callback
    cv2.namedWindow("Select Bounding Box")
    cv2.setMouseCallback("Select Bounding Box", draw_bounding_box)

    print("Instructions: Click and drag to draw a bounding box. Press 'q' to confirm, 'r' to reset, 'Esc' to cancel.")

    while True:
        cv2.imshow("Select Bounding Box", img)
        key = cv2.waitKey(1) & 0xFF

        # Press 'q' to confirm the bounding box
        if key == ord('q') and x_min != -1:
            break
        # Press 'r' to reset the bounding box
        elif key == ord('r'):
            img = img_copy.copy()
            x_min, y_min, x_max, y_max = -1, -1, -1, -1
            print("Reset bounding box. Draw again.")
        # Press 'Esc' to cancel
        elif key == 27:  # Escape key
            cv2.destroyAllWindows()
            raise ValueError("Bounding box selection canceled")

    cv2.destroyAllWindows()

    # Return the bounding box as a NumPy array
    if x_min == -1 or y_min == -1 or x_max == -1 or y_max == -1:
        print("No bounding box selected. Using default box.")
        return np.array([300, 0, 500, 400], dtype=np.float32)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

# Example usage
try:
    print("tri")
    # Replace 'path_to_your_image.jpg' with the path to your image
    image_path = video_dir +"/00" + start +".jpg"  # Update this path
    box = get_bounding_box(image_path)
    print(f"Final bounding box: {box}")
    cv2.destroyAllWindows()
except ValueError as e:
    print(f"Error: {e}")
    # Fallback to default box
    box = np.array([300, 0, 500, 400], dtype=np.float32)
    print(f"Using default box: {box}")
    cv2.destroyAllWindows()
###############################################


print("tri")
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_box(box, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])


# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 1
plt.close("all")
bboxes = []
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    #plt.figure(figsize=(6, 4))
    #plt.title(f"frame {out_frame_idx}")
    #plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    plt.show()
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        # Compute bounding box from mask
        mask = np.array(out_mask)  # Ensure mask is a NumPy array
        print(f"Mask dtype: {out_mask.dtype}")
        print(f"Unique values: {np.unique(out_mask)}")
        print(f"Number of True pixels: {np.sum(out_mask)}")

        if mask.ndim == 3:  # If mask is RGB, convert to binary
            print("RGB")
            mask = mask[0, :, :]
        print(f"Mask2 dtype: {mask.dtype}")
        print(f"Unique2 values: {np.unique(mask)}")
        print(f"Number2 of True pixels: {np.sum(mask)}")

        rows, cols = np.where(mask == True)  # Get coordinates of non-zero mask pixels
        print(rows, cols)
        if len(rows) > 0 and len(cols) > 0:  # Ensure mask is not empty
            print("vatre")
            x_min, x_max = cols.min(), cols.max()
            y_min, y_max = rows.min(), rows.max()
            width = x_max - x_min
            height = y_max - y_min

            # Draw bounding box
            bboxes.append([x_min, y_min, width, height])
            rect = Rectangle((x_min, y_min), width, height,
                             linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
        # Load the original image
        original_image = np.array(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))

        # Convert mask to NumPy array
        mask = np.array(out_mask)

        # Handle different mask formats
        if mask.ndim == 3:
            mask = mask[0, :, :]  # Take one channel
        elif mask.ndim != 2:
            print(f"Unexpected mask dimensions for saving: {mask.ndim}")
            continue

        # Ensure mask is binary
        if mask.dtype != np.bool_ and mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        # Check for empty mask
        if not np.any(mask):
            print(f"Empty mask for object {out_obj_id} in frame {out_frame_idx}")
            continue

        # Ensure mask and image have the same dimensions
        if mask.shape[:2] != original_image.shape[:2]:
            print(f"Mismatch: mask shape {mask.shape[:2]} vs image shape {original_image.shape[:2]}")
            continue

        # Apply mask to the original image
        masked_image = original_image.copy()
        if masked_image.ndim == 2:  # Grayscale image
            masked_image[mask == 0] = 0  # Set non-masked pixels to black
        else:  # RGB image
            for channel in range(masked_image.shape[2]):  # Apply mask to each channel
                masked_image[:, :, channel][mask == 0] = 0  # Set to black

        # Create a new figure for saving
        plt.figure(figsize=(6, 4))
        plt.axis('off')  # Turn off axes
        plt.imshow(masked_image)

        # Save the masked image as JPEG
        if out_frame_idx + int(start) < 10:
            output_path = os.path.join(output_dir, f"frame_0{out_frame_idx + int(start)}_obj_{out_obj_id}.jpg")
        else:
            output_path = os.path.join(output_dir, f"frame_{out_frame_idx + int(start)}_obj_{out_obj_id}.jpg")
        plt.savefig(output_path, format='jpg', bbox_inches='tight', pad_inches=0)
        print(f"Saved masked image: {output_path}")
        plt.close()  # Close the figure after saving
    plt.show(block=False)  # Non-blocking display
    plt.pause(1)  # Display for 5 seconds
    plt.close('all')  # Close all figures
with open(output_dir + '/coordinates.csv', 'w', newline='') as file:
        # Create a CSV writer object
    writer = csv.writer(file)

        # Write header
    writer.writerow(['x_min', 'y_min', 'width', 'height'])

        # Loop through the coordinates and write each row
    for coord in bboxes:
        writer.writerow(coord)