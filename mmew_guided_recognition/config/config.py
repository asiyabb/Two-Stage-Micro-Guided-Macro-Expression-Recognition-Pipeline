import os

class Config:
    DATA_ROOT = "/home/ab3379/work/data/MMEW_Final"
    MACRO_DIR = os.path.join(DATA_ROOT, "Macro_Expression")
    MICRO_DIR = os.path.join(DATA_ROOT, "Micro_Expression")
    EXCEL_ANNOTATIONS = os.path.join(DATA_ROOT, "MMEW_Micro_Exp.xlsx")

    EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    NUM_CLASSES = len(EMOTIONS)
    LABEL_MAP = {e: i for i, e in enumerate(EMOTIONS)}

    # Sequence and Window Specs (Aligned with pipeline diagram: 8 frames per window, stride 4)
    SEQ_LEN = 32
    WINDOW_SIZE = 8
    WINDOW_STRIDE = 4
    IMAGE_SIZE = (224, 224)

    # Training Specs
    BATCH_SIZE = 8
    EPOCHS_PRETRAIN = 20
    EPOCHS_MACRO = 20
    LR = 1e-4

    # Loss Weights
    ALPHA_CONTRASTIVE = 0.1
    BETA_TEMPORAL = 0.05

    DEVICE = "cuda"