MODEL_TYPE = {"TYPE": 'swin'}
DATA_TYPE = {"DATA_TYPE": '3DABUS'}
IMG_SIZE = (16, 128, 128)
NUMBER_WORKERS = 0
NUM_CLASSES = 1

# Model settings
# Dropout rate
DROP_RATE = 0.0
# Drop path rate
DROP_PATH_RATE = 0.1
# Label Smoothing
LABEL_SMOOTHING = 0#0.1 no need to do label smoothing in binary classification 

# train
TRAIN = {
         "AUGMENT": False,
         "BATCH_SIZE": 6,
         #DEFAULT EPOCHS
         "EPOCHS": 200,
         "WARMUP_EPOCHS": 20, #20,
         "WEIGHT_DECAY": 5e-2, #5e-2,
         "BASE_LR": 5e-4, #5e-4,
         "WARMUP_LR": 5e-7,
         "MIN_LR": 5e-6,
         # Clip gradient norm
         "CLIP_GRAD": 5.0,
         # Auto resume from latest checkpoint
         "AUTO_RESUME": True,
         # Gradient accumulation steps
         # could be overwritten by command line argument
         "ACCUMULATION_STEPS": 0,
         # Whether to use gradient checkpointing to save memory
         # could be overwritten by command line argument
         "USE_CHECKPOINT": False,

         # LR scheduler
         "LR_SCHEDULER_NAME": 'cosine',
         # Epoch interval to decay LR, used in StepLRScheduler
         "LR_SCHEDULER_DECAY_EPOCHS": 30,
         # LR decay rate, used in StepLRScheduler
         "LR_SCHEDULER_DECAY_RATE": 0.1,

         # Optimizer
         "OPTIMIZER_NAME": 'adamw',
         # Optimizer Epsilon
         "OPTIMIZER_EPS": 1e-8, #default
         # Optimizer Betas
         "OPTIMIZER_BETAS": (0.9, 0.999), #default
         # SGD momentum
         "OPTIMIZER_MOMENTUM": 0.9
         }

# val
VAL = {
        "BATCH_SIZE": 1,
        "CONF_THRESH": 0.01, #0.005,
        "MULTI_SCALE_VAL": True,
        "FLIP_VAL": True
        }

# Data augmentation parameters
AUG = {
        # Color jitter factor
        "COLOR_JITTER": 0.4,
        # Use AutoAugment policy. "v0" or "original"
        "AUTO_AUGMENT": 'rand-m9-mstd0.5-inc1',
        # Random erase prob
        "REPROB": 0.25,
        # Random erase mode
        "REMODE": 'pixel',
        # Random erase count
        "RECOUNT": 1,
        # Mixup alpha, mixup enabled if > 0
        "MIXUP": 0.5,#0.8,
        # Cutmix alpha, cutmix enabled if > 0
        "CUTMIX": 1.0,
        # Cutmix min/max ratio, overrides alpha and enables cutmix if set
        "CUTMIX_MINMAX": None,
        # Probability of performing mixup or cutmix when either/both is enabled
        "MIXUP_PROB": 1.0,
        # Probability of switching to cutmix when both mixup and cutmix enabled
        "MIXUP_SWITCH_PROB": 0.5,
        # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
        "MIXUP_MODE": 'batch'
        }
# Swin Transformer parameters
DEPTH_PATCH_SIZE = 2
PATCH_SIZE = 2 #4
IN_CHANS = 3 # original, histogram, mask

EMBED_DIM = 24 #48
DEPTHS = [2, 2, 2, 2]#[2, 2, 6, 2]
NUM_HEADS = [2, 2, 2, 2]#[3, 6, 12, 24]
"""
#primary structure settings
EMBED_DIM = 48
DEPTHS = [2, 2, 6, 2]
NUM_HEADS = [3, 6, 12, 24]
"""
WINDOW_SIZE = 8
MLP_RATIO = 4.
QKV_BIAS = True
QK_SCALE = None
APE = False
PATCH_NORM = True