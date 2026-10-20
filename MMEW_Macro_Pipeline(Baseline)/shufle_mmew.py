# -*- coding: utf-8 -*-
"""
ShuffleNetV2 Feature Extractor for MMEW Macro Expressions (Full 1024-dim)
"""
import os
import cv2
import numpy as np
import torch
import json
from tqdm import tqdm
import mediapipe as mp
import math
from collections import defaultdict
from torchvision import models, transforms

class ShuffleNetV2FaceProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self._init_mediapipe()
        
        # Load ShuffleNetV2 with pre-trained weights
        self.model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity() # Outputs 1024-dim features
        self.model = self.model.to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _init_mediapipe(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.7)

    def _align_face(self, frame, bbox):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = (landmarks[33].x, landmarks[33].y)
        right_eye = (landmarks[263].x, landmarks[263].y)
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))
        
        center = ((left_eye[0] + right_eye[0])/2 * frame.shape[1], (left_eye[1] + right_eye[1])/2 * frame.shape[0])
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        x, y, w, h = bbox
        return rotated[y:y+h, x:x+w]

    def _detect_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw = frame.shape[:2]
            x = max(0, int(bboxC.xmin * iw) - int(bboxC.width * iw * 0.2))
            y = max(0, int(bboxC.ymin * ih) - int(bboxC.height * ih * 0.2))
            w = min(iw - x, int(bboxC.width * iw * 1.4))
            h = min(ih - y, int(bboxC.height * ih * 1.4))
            return self._align_face(frame, (x, y, w, h)), (x, y, w, h)
        return frame, None

    def _preprocess_frame(self, frame):
        face_roi, _ = self._detect_face(frame)
        return self.transform(face_roi).unsqueeze(0).to(self.device)

    def _extract_features(self, tensor):
        with torch.no_grad():
            return self.model(tensor).squeeze().cpu().numpy()

    def process_sequence(self, image_paths, target_frames=100):
        features = []
        last_valid = None
        
        frames = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is not None:
                frames.append(img)
                
        if len(frames) == 0:
            return None
            
        for i in range(target_frames):
            idx = min(i, len(frames) - 1)
            frame = frames[idx]
            tensor = self._preprocess_frame(frame)
            frame_features = self._extract_features(tensor)
            last_valid = frame_features.copy()
            features.append(frame_features)
            
        return np.stack(features) # Shape: (100, 1024)

def process_mmew_dataset(input_dir, output_json):
    processor = ShuffleNetV2FaceProcessor()
    data = {"features": [], "labels": []}
    
    emotion_map = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'sadness': 4, 'surprise': 5}
    
    subjects = sorted(os.listdir(input_dir))
    for subject in subjects:
        subj_dir = os.path.join(input_dir, subject)
        if not os.path.isdir(subj_dir):
            continue
            
        for emotion, label_idx in emotion_map.items():
            emo_dir = os.path.join(subj_dir, emotion)
            if not os.path.isdir(emo_dir):
                continue
                
            sub_entries = sorted(os.listdir(emo_dir))
            clip_groups = defaultdict(list)
            for f in sub_entries:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    parts = f.split('-')
                    if len(parts) >= 2:
                        clip_prefix = f"{parts[0]}-{parts[1]}"
                        clip_groups[clip_prefix].append(os.path.join(emo_dir, f))
            
            for clip_prefix, img_files in clip_groups.items():
                img_files = sorted(img_files)
                features = processor.process_sequence(img_files, target_frames=100)
                if features is not None:
                    data["features"].append(features.tolist())
                    data["labels"].append(label_idx)

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Processing completed. Total samples extracted: {len(data['features'])}")
    print(f"Results saved to {output_json}")

if __name__ == "__main__":
    INPUT_DIR = r"/home/ab3379/work/data/MMEW_Final/Macro_Expression"
    OUTPUT_JSON = "Shufle_MMEW_1024d.json"
    process_mmew_dataset(INPUT_DIR, OUTPUT_JSON)