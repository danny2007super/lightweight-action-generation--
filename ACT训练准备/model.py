"""Build the author's ACT architecture without ALOHA, ROS or simulator imports."""
from argparse import Namespace
from pathlib import Path
import sys

import torch
from torch import nn

UPSTREAM = Path(__file__).parent / "upstream"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / "detr"))
from policy import ACTPolicy
from detr.models import backbone as backbone_module
from detr.models.detr_vae import DETRVAE, build_encoder
from detr.models.transformer import build_transformer


class PreparedACTPolicy(ACTPolicy):
    """Use upstream forward, CVAE, L1 and KL loss; adapt construction only."""
    def __init__(self, config, device):
        nn.Module.__init__(self)
        state_dim, action_dim = int(config["state_dim"]), int(config["action_dim"])
        if state_dim != action_dim or state_dim not in (7, 14):
            raise ValueError("supported dimensions: matching 7/7 or 14/14")
        args = Namespace(backbone="resnet18", dilation=False, masks=False,
                         lr_backbone=config["backbone_learning_rate"], position_embedding="sine",
                         hidden_dim=config["hidden_dim"], nheads=config["attention_heads"],
                         dim_feedforward=config["dim_feedforward"], enc_layers=config["encoder_layers"],
                         dec_layers=config["decoder_layers"], dropout=config["dropout"], pre_norm=False,
                         num_queries=config["chunk_size"], camera_names=config["camera_names"])
        # Upstream's backbone queries is_main_process() to choose pretrained weights.
        # Keep the vendored source intact and control that choice during construction.
        original = backbone_module.is_main_process
        backbone_module.is_main_process = lambda: bool(config["pretrained_backbone"])
        try:
            backbone = backbone_module.build_backbone(args)
        finally:
            backbone_module.is_main_process = original
        self.model = DETRVAE([backbone], build_transformer(args), build_encoder(args),
                             state_dim=state_dim, num_queries=args.num_queries, camera_names=args.camera_names)
        if state_dim != 14:
            self.model.input_proj_robot_state = nn.Linear(state_dim,args.hidden_dim)
            self.model.encoder_action_proj = nn.Linear(action_dim,args.hidden_dim)
            self.model.encoder_joint_proj = nn.Linear(state_dim,args.hidden_dim)
        self.model.to(device)
        self.kl_weight = config["kl_weight"]
        groups = [dict(params=[p for n,p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]),
                  dict(params=[p for n,p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                       lr=config["backbone_learning_rate"])]
        self.optimizer = torch.optim.AdamW(groups,lr=config["learning_rate"],weight_decay=config["weight_decay"])


def configure_cache(cache_dir):
    torch.hub.set_dir(str(Path(cache_dir).resolve()))
