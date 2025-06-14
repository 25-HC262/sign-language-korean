"""
Korean Sign Language Recognition Utility Functions

This module contains utility functions for KSL recognition using OpenPose.
"""

from .config import ROWS_PER_FRAME, SEQ_LEN
import json
import cv2
import numpy as np
import os
import sys

# OpenPose 설정
try:
    # OpenPose Python API path 설정
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError:
    print("OpenPose Python API not found. Please install OpenPose with Python API.")
    raise

class CFG:
    """
    Configuration class for KSL recognition.
    """
    sequence_length = SEQ_LEN
    rows_per_frame = ROWS_PER_FRAME
    
    # OpenPose parameters
    openpose_params = {
        "model_folder": "/path/to/openpose/models/",
        "face": True,
        "hand": True,
        "hand_detector": 2,
        "body": 1,
        "net_resolution": "-1x368",
        "face_net_resolution": "368x368",
        "hand_net_resolution": "368x368"
    }


class OpenPoseDetector:
    def __init__(self, params=None):
        """
        Initialize OpenPose detector
        """
        if params is None:
            params = CFG.openpose_params
            
        self.params = params
        
        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
    
    def process(self, image):
        """
        Process image with OpenPose
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            dict: Dictionary containing keypoints
        """
        datum = op.Datum()
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        
        return {
            'pose_keypoints': datum.poseKeypoints,
            'face_keypoints': datum.faceKeypoints,
            'hand_keypoints': [datum.handKeypoints[0], datum.handKeypoints[1]]  # [left, right]
        }


def openpose_detection(image, detector):
    """
    Perform landmark detection using OpenPose.
    
    Args:
        image (numpy.ndarray): Input image
        detector: OpenPose detector instance
        
    Returns:
        tuple: (image, results)
    """
    results = detector.process(image)
    return image, results


def extract_coordinates_openpose(results):
    """
    Extract coordinates from OpenPose results.
    
    Args:
        results: OpenPose detection results
        
    Returns:
        numpy.ndarray: Array of extracted coordinates [137, 3]
    """
    # Initialize arrays with NaN
    body = np.zeros((25, 3)) * np.nan
    face = np.zeros((70, 3)) * np.nan
    lh = np.zeros((21, 3)) * np.nan
    rh = np.zeros((21, 3)) * np.nan
    
    # Extract body keypoints
    if results['pose_keypoints'] is not None and len(results['pose_keypoints']) > 0:
        pose_kp = results['pose_keypoints'][0]  # First person
        if pose_kp.shape[0] == 25:
            body[:, :2] = pose_kp[:, :2]  # x, y
            body[:, 2] = pose_kp[:, 2]    # confidence as z
    
    # Extract face keypoints  
    if results['face_keypoints'] is not None and len(results['face_keypoints']) > 0:
        face_kp = results['face_keypoints'][0]
        if face_kp.shape[0] == 70:
            face[:, :2] = face_kp[:, :2]
            face[:, 2] = face_kp[:, 2]
    
    # Extract hand keypoints
    if results['hand_keypoints'][0] is not None and len(results['hand_keypoints'][0]) > 0:
        lh_kp = results['hand_keypoints'][0][0]
        if lh_kp.shape[0] == 21:
            lh[:, :2] = lh_kp[:, :2]
            lh[:, 2] = lh_kp[:, 2]
            
    if results['hand_keypoints'][1] is not None and len(results['hand_keypoints'][1]) > 0:
        rh_kp = results['hand_keypoints'][1][0]
        if rh_kp.shape[0] == 21:
            rh[:, :2] = rh_kp[:, :2]
            rh[:, 2] = rh_kp[:, 2]
    
    # Normalize coordinates to [0, 1] if they contain pixel values
    # Assuming image dimensions are available
    # You may need to pass image dimensions as parameters
    
    # Concatenate in order: body, face, left_hand, right_hand
    return np.concatenate([body, face, lh, rh])


def draw_openpose(image, results):
    """
    Draw OpenPose keypoints on image
    """
    # OpenPose already provides visualization
    # This is a placeholder for custom drawing if needed
    pass


def load_json_file(json_path):
    """
    Load a JSON file and return it as a dictionary.
    """
    with open(json_path, 'r') as f:
        sign_map = json.load(f)
    return sign_map