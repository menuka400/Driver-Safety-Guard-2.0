"""
Object Tracking Module
Handles tracking of detected objects (phones, faces) across frames
"""
import numpy as np
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ObjectTracker:
    def __init__(self, feature_dim=256, max_disappeared=60, max_distance=200):
        # Phone tracking
        self.next_phone_id = 1
        self.active_phone_tracks = {}
        self.phone_track_history = defaultdict(list)
        
        # Face tracking
        self.next_face_id = 1
        self.active_face_tracks = {}
        self.face_track_history = defaultdict(list)
        
        # Tracking parameters
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.feature_dim = feature_dim
        self.max_cosine_distance = 0.3
        
        # Visualization
        self.max_trail_length = 50
        self.trail_points = defaultdict(lambda: deque(maxlen=self.max_trail_length))
        self.show_trails = True
        self.trail_thickness = 2
        
        # Velocity tracking
        self.velocity_history = defaultdict(lambda: deque(maxlen=5))
        self.use_prediction = True
        
        # Colors for visualization
        self.phone_colors = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        self.face_colors = [
            (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 128), (255, 128, 128), (128, 128, 255), (255, 255, 128)
        ]
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1[:4]
        x2_min, y2_min, x2_max, y2_max = box2[:4]
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_cosine_similarity(self, feature1, feature2):
        """Calculate cosine similarity between two feature vectors"""
        if len(feature1) == 0 or len(feature2) == 0:
            return 0.0
        
        # Convert to numpy arrays
        f1 = np.array(feature1)
        f2 = np.array(feature2)
        
        # Calculate cosine similarity
        dot_product = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def predict_next_position(self, track_id, track_type="phone"):
        """Predict next position based on velocity history"""
        if not self.use_prediction:
            return None
        
        velocity_key = f"{track_type}_{track_id}"
        if velocity_key not in self.velocity_history or len(self.velocity_history[velocity_key]) < 2:
            return None
        
        # Calculate average velocity
        velocities = list(self.velocity_history[velocity_key])
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        
        # Get last known position
        if track_type == "phone" and track_id in self.active_phone_tracks:
            last_pos = self.active_phone_tracks[track_id]['centroid']
        elif track_type == "face" and track_id in self.active_face_tracks:
            last_pos = self.active_face_tracks[track_id]['centroid']
        else:
            return None
        
        # Predict next position
        predicted_x = last_pos[0] + avg_vx
        predicted_y = last_pos[1] + avg_vy
        
        return (predicted_x, predicted_y)
    
    def update_velocity(self, track_id, old_pos, new_pos, track_type="phone"):
        """Update velocity history for a track"""
        velocity_key = f"{track_type}_{track_id}"
        
        if old_pos and new_pos:
            vx = new_pos[0] - old_pos[0]
            vy = new_pos[1] - old_pos[1]
            self.velocity_history[velocity_key].append((vx, vy))
    
    def track_phones(self, phone_detections, phone_features):
        """Track mobile phones across frames"""
        if not phone_detections:
            # Update disappeared counts
            to_remove = []
            for track_id in self.active_phone_tracks:
                self.active_phone_tracks[track_id]['disappeared'] += 1
                if self.active_phone_tracks[track_id]['disappeared'] > self.max_disappeared:
                    to_remove.append(track_id)
            
            # Remove disappeared tracks
            for track_id in to_remove:
                del self.active_phone_tracks[track_id]
                if track_id in self.trail_points:
                    del self.trail_points[track_id]
            
            return []
        
        # Convert detections to centroids
        input_centroids = []
        for detection in phone_detections:
            x1, y1, x2, y2 = detection[:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            input_centroids.append((cx, cy))
        
        # If no existing tracks, create new ones
        if len(self.active_phone_tracks) == 0:
            for i, detection in enumerate(phone_detections):
                self.register_phone(detection, phone_features[i] if i < len(phone_features) else [])
        else:
            # Match detections to existing tracks
            self.match_phone_detections(phone_detections, phone_features, input_centroids)
        
        return list(self.active_phone_tracks.keys())
    
    def register_phone(self, detection, feature):
        """Register a new phone track"""
        x1, y1, x2, y2 = detection[:4]
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        self.active_phone_tracks[self.next_phone_id] = {
            'centroid': centroid,
            'bbox': detection,
            'feature': feature,
            'disappeared': 0,
            'total_frames': 1
        }
        
        # Add to trail
        self.trail_points[f"phone_{self.next_phone_id}"].append(centroid)
        
        self.next_phone_id += 1
    
    def match_phone_detections(self, detections, features, centroids):
        """Match current detections to existing phone tracks"""
        track_ids = list(self.active_phone_tracks.keys())
        
        # Calculate distance matrix
        distance_matrix = np.zeros((len(track_ids), len(centroids)))
        
        for i, track_id in enumerate(track_ids):
            track_centroid = self.active_phone_tracks[track_id]['centroid']
            
            for j, detection_centroid in enumerate(centroids):
                # Combine spatial distance and feature similarity
                spatial_dist = self.calculate_distance(track_centroid, detection_centroid)
                
                feature_sim = 0.0
                if j < len(features) and len(features[j]) > 0 and len(self.active_phone_tracks[track_id]['feature']) > 0:
                    feature_sim = self.calculate_cosine_similarity(
                        self.active_phone_tracks[track_id]['feature'],
                        features[j]
                    )
                
                # Combined distance (lower is better)
                combined_distance = spatial_dist * (1 - feature_sim)
                distance_matrix[i, j] = combined_distance
        
        # Hungarian algorithm would be ideal here, but using simple greedy matching
        used_detection_indices = set()
        used_track_indices = set()
        
        # Sort by distance and assign
        distances = []
        for i in range(len(track_ids)):
            for j in range(len(centroids)):
                distances.append((distance_matrix[i, j], i, j))
        
        distances.sort()
        
        for distance, track_idx, detection_idx in distances:
            if track_idx in used_track_indices or detection_idx in used_detection_indices:
                continue
            
            if distance < self.max_distance:
                track_id = track_ids[track_idx]
                
                # Update track
                old_centroid = self.active_phone_tracks[track_id]['centroid']
                new_centroid = centroids[detection_idx]
                
                self.active_phone_tracks[track_id]['centroid'] = new_centroid
                self.active_phone_tracks[track_id]['bbox'] = detections[detection_idx]
                self.active_phone_tracks[track_id]['disappeared'] = 0
                self.active_phone_tracks[track_id]['total_frames'] += 1
                
                if detection_idx < len(features):
                    self.active_phone_tracks[track_id]['feature'] = features[detection_idx]
                
                # Update velocity and trail
                self.update_velocity(track_id, old_centroid, new_centroid, "phone")
                self.trail_points[f"phone_{track_id}"].append(new_centroid)
                
                used_track_indices.add(track_idx)
                used_detection_indices.add(detection_idx)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detection_indices:
                feature = features[i] if i < len(features) else []
                self.register_phone(detection, feature)
        
        # Mark unmatched tracks as disappeared
        for i, track_id in enumerate(track_ids):
            if i not in used_track_indices:
                self.active_phone_tracks[track_id]['disappeared'] += 1
                
                # Remove if disappeared too long
                if self.active_phone_tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.active_phone_tracks[track_id]
                    if f"phone_{track_id}" in self.trail_points:
                        del self.trail_points[f"phone_{track_id}"]
    
    def track_faces(self, face_detections, face_features):
        """Track faces across frames (similar to phone tracking)"""
        # Similar implementation as phone tracking but for faces
        # This is a simplified version - the full implementation would be similar to track_phones
        
        if not face_detections:
            # Update disappeared counts for faces
            to_remove = []
            for track_id in self.active_face_tracks:
                self.active_face_tracks[track_id]['disappeared'] += 1
                if self.active_face_tracks[track_id]['disappeared'] > self.max_disappeared:
                    to_remove.append(track_id)
            
            for track_id in to_remove:
                del self.active_face_tracks[track_id]
            
            return []
        
        # For now, simple centroid-based tracking for faces
        face_centroids = []
        for detection in face_detections:
            x1, y1, x2, y2 = detection[:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            face_centroids.append((cx, cy))
        
        if len(self.active_face_tracks) == 0:
            for i, detection in enumerate(face_detections):
                self.register_face(detection, face_features[i] if i < len(face_features) else [])
        else:
            self.match_face_detections(face_detections, face_features, face_centroids)
        
        return list(self.active_face_tracks.keys())
    
    def register_face(self, detection, feature):
        """Register a new face track"""
        x1, y1, x2, y2 = detection[:4]
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        self.active_face_tracks[self.next_face_id] = {
            'centroid': centroid,
            'bbox': detection,
            'feature': feature,
            'disappeared': 0,
            'total_frames': 1
        }
        
        self.next_face_id += 1
    
    def match_face_detections(self, detections, features, centroids):
        """Match current detections to existing face tracks"""
        # Simplified matching for faces - similar logic as phones
        track_ids = list(self.active_face_tracks.keys())
        
        if not track_ids:
            return
        
        # Simple distance-based matching
        used_detections = set()
        
        for track_id in track_ids:
            track_centroid = self.active_face_tracks[track_id]['centroid']
            best_distance = float('inf')
            best_detection_idx = -1
            
            for i, detection_centroid in enumerate(centroids):
                if i in used_detections:
                    continue
                    
                distance = self.calculate_distance(track_centroid, detection_centroid)
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_detection_idx = i
            
            if best_detection_idx != -1:
                # Update track
                self.active_face_tracks[track_id]['centroid'] = centroids[best_detection_idx]
                self.active_face_tracks[track_id]['bbox'] = detections[best_detection_idx]
                self.active_face_tracks[track_id]['disappeared'] = 0
                self.active_face_tracks[track_id]['total_frames'] += 1
                
                if best_detection_idx < len(features):
                    self.active_face_tracks[track_id]['feature'] = features[best_detection_idx]
                
                used_detections.add(best_detection_idx)
            else:
                self.active_face_tracks[track_id]['disappeared'] += 1
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                feature = features[i] if i < len(features) else []
                self.register_face(detection, feature)
        
        # Remove disappeared tracks
        to_remove = []
        for track_id in list(self.active_face_tracks.keys()):
            if self.active_face_tracks[track_id]['disappeared'] > self.max_disappeared:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.active_face_tracks[track_id]
    
    def get_tracking_info(self):
        """Get current tracking information"""
        return {
            "active_phone_tracks": len(self.active_phone_tracks),
            "active_face_tracks": len(self.active_face_tracks),
            "total_phone_ids": self.next_phone_id - 1,
            "total_face_ids": self.next_face_id - 1
        }
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.active_phone_tracks.clear()
        self.active_face_tracks.clear()
        self.phone_track_history.clear()
        self.face_track_history.clear()
        self.trail_points.clear()
        self.velocity_history.clear()
        self.next_phone_id = 1
        self.next_face_id = 1
        print("🔄 Object tracking reset")
