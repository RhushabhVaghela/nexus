import torch
import torch.nn as nn

class CrossModalAlignment(nn.Module):
    """
     aligns embedding spaces of disparate encoders (Vision, Audio) 
    into the shared Student Latent Space (Core).
    """
    def __init__(self, core_dim: int, vision_dim: int = 1024, audio_dim: int = 2048, video_dim: int = 1024, tool_dim: int = 1024):
        super().__init__()
        # Vision: SigLIP (1024) -> Core (2048) | Handles Images
        self.vision_proj = nn.Linear(vision_dim, core_dim)
        
        # Audio: Qwen3-TTS (2048) -> Core (2048) | Handles Speech/Sounds
        self.audio_proj = nn.Linear(audio_dim, core_dim)
        
        # Video: VideoMAE (1024) -> Core (2048) | Handles Video
        self.video_proj = nn.Linear(video_dim, core_dim)
        
        # Tools: Action Embeddings (1024) -> Core (2048) | Handles Tool/Function States
        self.tool_proj = nn.Linear(tool_dim, core_dim)
        
        self.norm_v = nn.LayerNorm(core_dim)
        self.norm_a = nn.LayerNorm(core_dim)
        self.norm_vid = nn.LayerNorm(core_dim)
        self.norm_tool = nn.LayerNorm(core_dim)
        self.act = nn.GELU()
        
    def forward(self, vision_feats=None, audio_feats=None, video_feats=None, tool_feats=None):
        """
        Multimodal Fusion Layer.
        Args:
            vision_feats: [B, S_v, D_v] - Images
            audio_feats:  [B, S_a, D_a] - Speech/Audio
            video_feats:  [B, S_vd, D_vd] - Video
            tool_feats:   [B, S_t, D_t] - Tool Outputs/Action States
        Returns:
            aligned_combined: [Batch, Total_Seq, core_dim]
        """
        aligned_outputs = []
        
        if vision_feats is not None:
            aligned_outputs.append(self.act(self.norm_v(self.vision_proj(vision_feats))))
            
        if audio_feats is not None:
            aligned_outputs.append(self.act(self.norm_a(self.audio_proj(audio_feats))))
            
        if video_feats is not None:
             aligned_outputs.append(self.act(self.norm_vid(self.video_proj(video_feats))))
             
        if tool_feats is not None:
             aligned_outputs.append(self.act(self.norm_tool(self.tool_proj(tool_feats))))
             
        if not aligned_outputs:
            return None
        
        # Concatenate sequences to form a multimodal context 
        return torch.cat(aligned_outputs, dim=1)
