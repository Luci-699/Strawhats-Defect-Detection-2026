import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MorphologyEncoder(nn.Module):
    """
    MLP encoder to transform the 11-dimensional morphological feature vector 
    into a 128-dimensional embedding (f_morph).
    """
    def __init__(self):
        super(MorphologyEncoder, self).__init__()
        
        self.fc1 = nn.Linear(11, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 128)
        self.layernorm = nn.LayerNorm(128)

        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights using Xavier normalization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.
        
        Args:
            x (torch.Tensor): 11-dimensional feature vector.
            
        Returns:
            torch.Tensor: 128-dimensional embedding.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
            
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        f_morph = self.layernorm(out)
        
        return f_morph
