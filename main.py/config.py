# Dataset Paths
TRAIN_DIR = "../dataset/train"
VAL_DIR = "../dataset/val"
TEST_DIR = "../dataset/test"

# Image Settings
IMAGE_SIZE = 224

# Training Hyperparameters
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 0.0001

# Number of Classes
NUM_CLASSES = 6

# Model Save Paths - Use absolute paths
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "outputs", "logs")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")

BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "last_model.pth")
LOG_FILE = os.path.join(LOGS_DIR, "training_log.csv")

# Device
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
