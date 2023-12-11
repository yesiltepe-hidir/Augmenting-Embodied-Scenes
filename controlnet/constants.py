from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DDIMScheduler
import os

SCHEDULER_DICT = {
    'euler_ancestral': EulerAncestralDiscreteScheduler,
    'euler': EulerDiscreteScheduler,
    'ddim': DDIMScheduler,
}

PREPROCESSOR_DICT = {
    'lineart_realistic': "lllyasviel/control_v11p_sd15_lineart",
    'lineart_coarse': "lllyasviel/control_v11p_sd15_lineart",
    'lineart_standard': "lllyasviel/control_v11p_sd15_lineart",
    'lineart_anime': "lllyasviel/control_v11p_sd15s2_lineart_anime",
    'lineart_anime_denoise': "lllyasviel/control_v11p_sd15s2_lineart_anime",
    'softedge_hed': 'lllyasviel/control_v11p_sd15_softedge',
    'softedge_hedsafe': 'lllyasviel/control_v11p_sd15_softedge',
    'softedge_pidinet': 'lllyasviel/control_v11p_sd15_softedge',
    'softedge_pidsafe': 'lllyasviel/control_v11p_sd15_softedge',
    'canny': 'lllyasviel/control_v11p_sd15_canny',
    'depth_leres': 'lllyasviel/control_v11f1p_sd15_depth',
    'depth_leres++': 'lllyasviel/control_v11f1p_sd15_depth',
    'depth_midas': 'lllyasviel/control_v11f1p_sd15_depth',
    'depth_zoe': 'lllyasviel/control_v11f1p_sd15_depth',
}