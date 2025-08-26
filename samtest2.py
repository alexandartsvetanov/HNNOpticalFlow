import os
import cv2


def extract_frames(video_path, output_dir, num_frames=52):
    """
    Extracts equally spaced frames from a video and saves them as JPEG images.

    Args:
        video_path (str): Path to the input MP4 video file.
        output_dir (str): Directory to save the extracted frames.
        num_frames (int): Number of frames to extract (default: 100).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"Video Info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")

    # Calculate frame indices to extract
    #frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frame_indices = [int(i) for i in range(num_frames)]
    frame_indices = [min(i, total_frames - 1) for i in frame_indices]  # Ensure we don't go beyond

    # Extract and save frames
    saved_count = 0
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            output_path = os.path.join(output_dir, f"{i:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        else:
            print(f"Warning: Could not read frame {frame_idx}")

    # Release resources
    cap.release()
    print(f"Successfully saved {saved_count}/{num_frames} frames to {output_dir}")


if __name__ == "__main__":
    # Example usage
    videonum = "5"
    video_path = "collision" + videonum + ".mp4"  # Replace with your video path
    #os.makedirs("videos" + videonum)
    os.makedirs("videos" + videonum + "/Frames4")
    output_dir = "videos" + videonum + "/Frames4" # Output directory

    extract_frames(video_path, output_dir)