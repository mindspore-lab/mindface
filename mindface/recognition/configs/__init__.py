"""
init
"""
from .train_config_casia_mobile import combination_cfg as casia_mobile
from .train_config_casia_r50 import combination_cfg as casia_r50
from .train_config_casia_r100 import combination_cfg as casia_r100
from .train_config_ms1mv2_mobile import combination_cfg as ms1mv2_mobile
from .train_config_ms1mv2_r50 import combination_cfg as ms1mv2_r50
from .train_config_ms1mv2_r100 import combination_cfg as ms1mv2_r100
from .train_config_casia_vit_t import combination_cfg as casia_vit_t
from .train_config_casia_vit_s import combination_cfg as casia_vit_s
from .train_config_casia_vit_b import combination_cfg as casia_vit_b
from .train_config_casia_vit_l import combination_cfg as casia_vit_l
from .train_config_ms1mv2_vit_t import combination_cfg as ms1mv2_vit_t
from .train_config_ms1mv2_vit_s import combination_cfg as ms1mv2_vit_s
from .train_config_ms1mv2_vit_b import combination_cfg as ms1mv2_vit_b
from .train_config_ms1mv2_vit_l import combination_cfg as ms1mv2_vit_l

config_combs = {
    "casia_mobile": casia_mobile,
    "casia_r50": casia_r50,
    "casia_r100": casia_r100,
    "ms1mv2_mobile": ms1mv2_mobile,
    "ms1mv2_r50": ms1mv2_r50,
    "ms1mv2_r100": ms1mv2_r100,
    "casia_vit_t": casia_vit_t,
    "casia_vit_s": casia_vit_s,
    "casia_vit_b": casia_vit_b,
    "casia_vit_l": casia_vit_l,
    "ms1mv2_vit_t": ms1mv2_vit_t,
    "ms1mv2_vit_s": ms1mv2_vit_s,
    "ms1mv2_vit_b": ms1mv2_vit_b,
    "ms1mv2_vit_l": ms1mv2_vit_l,
}
