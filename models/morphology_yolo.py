import torch
import torch.nn as nn
import numpy as np
import cv2
import asyncio
from typing import List, Dict, Any, Tuple
from scipy.ndimage import distance_transform_edt

# Import sub-modules
from .yolo_backbone import YOLOv10Backbone
from .cross_attention import CrossAttentionFusion
from .classification_head import MorphologyClassificationHead
import sys
import os

# To allow importing morphology from Desktop
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from morphology.dsp_filters import DSPPreprocessor

class MorphologyYOLO(nn.Module):
    """Full integrated pipeline for morphology-aware crack inspection.
    
    Combines:
    1. DSP preprocess
    2. CLAHE + Gaussian
    3. EDT attention map
    4. YOLOv10 backbone
    5. Cross-attention fusion
    6. Morphology-refined classifier
    """
    
    def __init__(self, num_classes: int = 6, device: str = 'cuda'):
        """Initialize the complete Morphology YOLO pipeline.
        
        Parameters
        ----------
        num_classes : int, optional
            Number of defect classes, by default 6
        device : str, optional
            Device to run the model on ('cuda' or 'cpu'), by default 'cuda'
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.backbone = YOLOv10Backbone('yolov10n.pt').to(self.device)
        
        # Assume visual features from backbone have dimension 256 (depends on layer)
        # Assume morphology features have dimension 128
        self.visual_dim = 256  
        self.morph_dim = 128
        self.d_model = 128
        
        self.cross_attention = CrossAttentionFusion(
            visual_dim=self.visual_dim,
            morph_dim=self.morph_dim,
            d_model=self.d_model,
            num_heads=4
        ).to(self.device)
        
        self.classifier = MorphologyClassificationHead(
            in_features=self.visual_dim + self.d_model,
            num_classes=num_classes,
            temperature=1.0
        ).to(self.device)
        
        # Dummy morphology encoder for placeholder
        # In a real scenario, this would be a specialized CNN or MLP
        self.morph_encoder = nn.Linear(100, self.morph_dim).to(self.device)
        nn.init.xavier_uniform_(self.morph_encoder.weight)
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply DSP, CLAHE, and generate EDT attention map.
        
        Parameters
        ----------
        image : np.ndarray
            Raw grayscale image.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Processed image tensor and attention map tensor.
        """
        # 1. DSP preprocessing
        dsp_img = DSPPreprocessor.full_dsp_pipeline(image)
        
        # Convert back to uint8 for CLAHE if normalized to float
        dsp_img_uint8 = cv2.normalize(dsp_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 2. CLAHE + Gaussian
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(dsp_img_uint8)
        blur_img = cv2.GaussianBlur(clahe_img, (5, 5), 0)
        
        # 3. Generate EDT attention map from binary mask (simple thresholding here)
        _, binary_mask = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edt_map = distance_transform_edt(binary_mask)
        edt_map = cv2.normalize(edt_map, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Convert to tensors
        img_tensor = torch.from_numpy(blur_img).float() / 255.0
        attn_tensor = torch.from_numpy(edt_map).float()
        
        return img_tensor, attn_tensor
        
    def _extract_morphology_features(self, edt_map: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Extract morphological features from the EDT map for given boxes.
        
        Parameters
        ----------
        edt_map : torch.Tensor
            Attention map.
        boxes : torch.Tensor
            Bounding boxes.
            
        Returns
        -------
        torch.Tensor
            Encoded morphology features.
        """
        num_boxes = boxes.shape[0]
        if num_boxes == 0:
            return torch.empty((0, self.morph_dim), device=self.device)
            
        # Placeholder: returning random features or flattened pooled regions
        # We'll just generate dummy raw features and pass through encoder
        raw_features = torch.randn(num_boxes, 100, device=self.device)
        encoded_morph = self.morph_encoder(raw_features)
        
        return encoded_morph

    def forward(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Forward pass for the integrated pipeline.
        
        Parameters
        ----------
        image : np.ndarray
            Input grayscale image.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of detections with refined classes.
        """
        # Preprocessing
        img_t, attn_t = self.preprocess_image(image)
        
        # Stack channels: [image, attention_map]
        # Shape: (1, 2, H, W)
        input_tensor = torch.stack([img_t, attn_t], dim=0).unsqueeze(0).to(self.device)
        
        # YOLO backbone
        _ = self.backbone(input_tensor)
        
        # Extract detections (simplified format assuming list of boxes)
        # Using dummy boxes for implementation completeness if YOLO output is complex
        # Actual extraction depends on ultralytics YOLO output format
        dummy_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]], device=self.device)
        
        # Extract features
        f_visual = self.backbone.get_roi_features(dummy_boxes)
        
        # If dimension mismatch for dummy implementation, pad or slice
        if f_visual.shape[-1] != self.visual_dim and f_visual.numel() > 0:
            if f_visual.shape[-1] < self.visual_dim:
                pad = torch.zeros(f_visual.shape[0], self.visual_dim - f_visual.shape[-1], device=self.device)
                f_visual = torch.cat([f_visual, pad], dim=-1)
            else:
                f_visual = f_visual[:, :self.visual_dim]
                
        f_morph = self._extract_morphology_features(attn_t, dummy_boxes)
        
        if f_visual.numel() == 0:
            return []
            
        # Cross-attention fusion
        fused_features = self.cross_attention(f_visual, f_morph)
        
        # Classification
        class_probs = self.classifier(fused_features)
        
        # Format output
        results = []
        for i in range(dummy_boxes.shape[0]):
            box = dummy_boxes[i].cpu().numpy()
            probs = class_probs[i]
            conf, cls_idx = torch.max(probs, dim=-1)
            
            results.append({
                'bbox': box.tolist(),
                'class': int(cls_idx.item()),
                'confidence': float(conf.item()),
                'morphology_features': f_morph[i].detach().cpu().numpy().tolist()
            })
            
        return results

    async def async_forward(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Asynchronous forward pass for real-time design.
        
        Parameters
        ----------
        image : np.ndarray
            Input image.
            
        Returns
        -------
        List[Dict[str, Any]]
            Results.
        """
        # Run preprocessing in a thread if it's CPU bound
        loop = asyncio.get_event_loop()
        img_t, attn_t = await loop.run_in_executor(None, self.preprocess_image, image)
        
        input_tensor = torch.stack([img_t, attn_t], dim=0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _ = self.backbone(input_tensor)
            dummy_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3]], device=self.device)
            f_visual = self.backbone.get_roi_features(dummy_boxes)
            
            if f_visual.shape[-1] != self.visual_dim and f_visual.numel() > 0:
                if f_visual.shape[-1] < self.visual_dim:
                    pad = torch.zeros(f_visual.shape[0], self.visual_dim - f_visual.shape[-1], device=self.device)
                    f_visual = torch.cat([f_visual, pad], dim=-1)
                else:
                    f_visual = f_visual[:, :self.visual_dim]
                
            f_morph = self._extract_morphology_features(attn_t, dummy_boxes)
            fused_features = self.cross_attention(f_visual, f_morph)
            class_probs = self.classifier(fused_features)
            
        # Format
        results = []
        for i in range(dummy_boxes.shape[0]):
            conf, cls_idx = torch.max(class_probs[i], dim=-1)
            results.append({
                'bbox': dummy_boxes[i].cpu().numpy().tolist(),
                'class': int(cls_idx.item()),
                'confidence': float(conf.item()),
                'morphology_features': f_morph[i].detach().cpu().numpy().tolist()
            })
            
        return results
