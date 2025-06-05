"""
Video processing utilities for the object tracking project.
Includes functions for video I/O, frame processing, and format conversion.
"""
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging


class VideoProcessor:
    """Utility class for video processing operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_frames(self, video_path, output_dir, frame_interval=30):
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save extracted frames
            frame_interval (int): Extract every Nth frame
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = 0
        saved_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_filename = output_dir / f"frame_{saved_count:06d}.jpg"
                    cv2.imwrite(str(frame_filename), frame)
                    saved_count += 1
                
                frame_count += 1
            
            self.logger.info(f"Extracted {saved_count} frames from {frame_count} total frames")
            
        finally:
            cap.release()
    
    def create_video_from_frames(self, frames_dir, output_path, fps=30):
        """
        Create video from sequence of frame images.
        
        Args:
            frames_dir (str): Directory containing frame images
            output_path (str): Output video file path
            fps (int): Frames per second for output video
        """
        frames_dir = Path(frames_dir)
        frame_files = sorted(frames_dir.glob("*.jpg"))
        
        if not frame_files:
            raise ValueError(f"No frame files found in {frames_dir}")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        height, width, _ = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        try:
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                out.write(frame)
            
            self.logger.info(f"Created video: {output_path} from {len(frame_files)} frames")
            
        finally:
            out.release()
    
    def resize_video(self, input_path, output_path, target_width, target_height):
        """
        Resize video to target dimensions.
        
        Args:
            input_path (str): Input video path
            output_path (str): Output video path
            target_width (int): Target width
            target_height (int): Target height
        """
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get original video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                resized_frame = cv2.resize(frame, (target_width, target_height))
                out.write(resized_frame)
                frame_count += 1
            
            self.logger.info(f"Resized {frame_count} frames to {target_width}x{target_height}")
            
        finally:
            cap.release()
            out.release()
    
    def get_video_info(self, video_path):
        """
        Get detailed information about a video file.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Video information
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration_seconds': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
            }
            
            return info
            
        finally:
            cap.release()


def convert_bbox_format(bbox, from_format='xyxy', to_format='xywh'):
    """
    Convert bounding box between different formats.
    
    Args:
        bbox (list): Bounding box coordinates
        from_format (str): Source format ('xyxy', 'xywh', 'cxcywh')
        to_format (str): Target format ('xyxy', 'xywh', 'cxcywh')
        
    Returns:
        list: Converted bounding box coordinates
    """
    if from_format == to_format:
        return bbox
    
    # Convert to standard xyxy format first
    if from_format == 'xywh':
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
    elif from_format == 'cxcywh':
        cx, cy, w, h = bbox
        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    else:  # xyxy
        x1, y1, x2, y2 = bbox
    
    # Convert from xyxy to target format
    if to_format == 'xywh':
        return [x1, y1, x2 - x1, y2 - y1]
    elif to_format == 'cxcywh':
        return [(x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1]
    else:  # xyxy
        return [x1, y1, x2, y2]


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (list): First bounding box [x1, y1, x2, y2]
        box2 (list): Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def draw_detection_results(image, detections, labels=None, colors=None):
    """
    Draw detection results on image.
    
    Args:
        image (np.ndarray): Input image
        detections (list): List of detections with bounding boxes
        labels (list, optional): Labels for each detection
        colors (list, optional): Colors for each detection
        
    Returns:
        np.ndarray: Image with drawn detections
    """
    result_image = image.copy()
    
    for i, detection in enumerate(detections):
        bbox = detection.get('bbox', detection)
        label = labels[i] if labels and i < len(labels) else f"Object {i}"
        color = colors[i] if colors and i < len(colors) else (0, 255, 0)
        
        # Convert bbox to integers
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(result_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_image
