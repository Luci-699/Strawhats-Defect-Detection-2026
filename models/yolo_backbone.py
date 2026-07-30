import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOv10Backbone(nn.Module):
    """Wrapper class for YOLOv10 backbone.
    
    Handles:
    - Forward pass through backbone
    - Modifying first conv layer for 2-channel input (grayscale + attention map)
    - ROI feature extraction for detected boxes
    - Feature map access for fusion module
    """
    
    def __init__(self, pretrained_weights: str = 'yolov10n.pt'):
        """Initialize YOLOv10 backbone and adapt it for 2-channel input.
        
        Parameters
        ----------
        pretrained_weights : str, optional
            Path to pretrained YOLOv10 weights, by default 'yolov10n.pt'
        """
        super().__init__()
        # Load YOLO model cleanly without triggering Ultralytics trainer engine
        import os
        if os.path.exists(pretrained_weights) and pretrained_weights != 'yolov10n.pt':
            self.yolo = YOLO('yolov10n.pt')
            try:
                ckpt = torch.load(pretrained_weights, map_location='cpu')
                if 'model' in ckpt:
                    model_obj = ckpt['model']
                    if hasattr(model_obj, 'state_dict'):
                        self.yolo.model.load_state_dict(model_obj.state_dict(), strict=False)
                    elif isinstance(model_obj, dict):
                        self.yolo.model.load_state_dict(model_obj, strict=False)
                elif 'state_dict' in ckpt:
                    self.yolo.model.load_state_dict(ckpt['state_dict'], strict=False)
            except Exception as e:
                print(f"[WARN] Failed to load custom state_dict from {pretrained_weights}: {e}")
        else:
            self.yolo = YOLO(pretrained_weights)
            
        self.model = self.yolo.model
        
        # Modify the first convolutional layer for 2-channel input
        # Original is likely 3-channel (RGB)
        first_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                first_layer = module
                break
                
        if first_layer is not None and first_layer.in_channels == 3:
            # Create a new conv layer with 2 input channels
            new_first_layer = nn.Conv2d(
                in_channels=2, 
                out_channels=first_layer.out_channels,
                kernel_size=first_layer.kernel_size,
                stride=first_layer.stride,
                padding=first_layer.padding,
                bias=(first_layer.bias is not None)
            )
            
            # Initialize weights: copy weights from first 2 channels or average them
            with torch.no_grad():
                new_first_layer.weight[:, :2, :, :] = first_layer.weight[:, :2, :, :]
                if first_layer.bias is not None:
                    new_first_layer.bias = first_layer.bias
                    
            # Replace the layer in the model
            if hasattr(self.model.model[0], 'conv'):
                self.model.model[0].conv = new_first_layer
                
        # To extract intermediate feature maps, we can use forward hooks
        self.feature_maps = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to extract intermediate features."""
        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output
            return hook
            
        # Hook into a deep layer for rich semantic features
        target_layer = None
        for idx, (name, module) in enumerate(self.model.named_modules()):
            if module.__class__.__name__ == 'SPPF':
                module.register_forward_hook(get_activation(f'feat_{idx}'))
                target_layer = f'feat_{idx}'
        self.target_layer = target_layer
                
    def forward(self, x: torch.Tensor):
        """Forward pass through the backbone.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 2, H, W).
            
        Returns
        -------
        list
            YOLO detections.
        """
        self.feature_maps = {}
        return self.model(x)
        
    def get_roi_features(self, boxes: torch.Tensor, batch_idx: int = 0) -> torch.Tensor:
        """Extract ROI features for detected boxes from the intermediate feature map.
        
        Parameters
        ----------
        boxes : torch.Tensor
            Bounding boxes of shape (N, 4) in format (x1, y1, x2, y2). Expected to be normalized [0, 1].
        batch_idx : int, optional
            Batch index to extract from, by default 0
            
        Returns
        -------
        torch.Tensor
            Extracted features of shape (N, C), where C is feature dimension.
        """
        import torchvision

        if not self.feature_maps or self.target_layer not in self.feature_maps:
            raise RuntimeError("No feature maps extracted. Run forward pass first.")
            
        feat_map = self.feature_maps[self.target_layer] # Shape: (B, C, H, W)
        B, C, H, W = feat_map.shape
        
        if boxes.shape[0] == 0:
            return torch.empty((0, C), device=feat_map.device)
            
        # Scale to feature map dimensions
        scaled_boxes = boxes.clone()
        scaled_boxes[:, [0, 2]] *= W
        scaled_boxes[:, [1, 3]] *= H
        
        # Add batch index for roi_align
        batch_indices = torch.full((boxes.shape[0], 1), batch_idx, device=boxes.device, dtype=boxes.dtype)
        roi_boxes = torch.cat([batch_indices, scaled_boxes], dim=1)
        
        # Use roi_align to pool a 1x1 region (Adaptive Average Pooling equivalent over the box)
        roi_features = torchvision.ops.roi_align(feat_map, roi_boxes, output_size=(1, 1), spatial_scale=1.0)
        
        return roi_features.squeeze(-1).squeeze(-1) # Shape: (N, C)
