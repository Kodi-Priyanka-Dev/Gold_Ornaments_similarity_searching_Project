import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

# ==========================================
# PATHS
# ==========================================

MODEL_PATH = r"D:\Gold searching\models\best_model.pth"

IMAGE_PATH = r"D:\Gold searching\dataset\test\Bangles\020_0014.png"

# ==========================================
# SETTINGS
# ==========================================

NUM_CLASSES = 4

CLASS_NAMES = [
    "Bangles",
    "Bracelets",
    "Ear Rings",
    "necklace"
]

CONFIDENCE_THRESHOLD = 0.70

# ==========================================
# DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================
# LOAD MODEL
# ==========================================

model = timm.create_model(
    "convnext_base",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)
model.eval()

# ==========================================
# CLASSIFICATION
# ==========================================

image = Image.open(IMAGE_PATH).convert("RGB")

image = transform(image)

image = image.unsqueeze(0).to(device)

with torch.no_grad():

    output = model(image)

    probability = F.softmax(output, dim=1)

confidence, prediction = torch.max(probability, dim=1)

confidence = confidence.item()
prediction = prediction.item()

print("--------------------------------")
print("Prediction :", CLASS_NAMES[prediction])
print("Confidence :", round(confidence*100,2), "%")
print("--------------------------------")

if confidence < CONFIDENCE_THRESHOLD:

    print("Rejected")
    print("Low confidence image.")

else:

    print("Accepted")
    print("Proceed to Similarity Search")