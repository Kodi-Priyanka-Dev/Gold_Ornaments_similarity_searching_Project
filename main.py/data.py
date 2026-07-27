import os
import random
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Prefer the workspace's actual dataset folder, but keep the older layout as a fallback.
SOURCE_DIR_CANDIDATES = [
    os.path.join(BASE_DIR, "Jewellery_Data"),
    os.path.join(BASE_DIR, "dataset", "original"),
    os.path.join(BASE_DIR, "dataset"),
]

SOURCE_DIR = next((path for path in SOURCE_DIR_CANDIDATES if os.path.isdir(path)), None)

if SOURCE_DIR is None:
    raise FileNotFoundError(
        "Could not find a dataset folder. Expected either 'Jewellery_Data', 'dataset/original', or 'dataset'."
    )

OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

random.seed(42)

classes = os.listdir(SOURCE_DIR)

for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(split_dir, exist_ok=True)

for cls in classes:

    source_folder = os.path.join(SOURCE_DIR, cls)

    if not os.path.isdir(source_folder):
        continue

    images = [img for img in os.listdir(source_folder)
              if img.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split_name, split_imgs in [
        ("train", train_imgs),
        ("val", val_imgs),
        ("test", test_imgs),
    ]:

        destination = os.path.join(OUTPUT_DIR, split_name, cls)
        os.makedirs(destination, exist_ok=True)

        for img in split_imgs:
            shutil.copy(
                os.path.join(source_folder, img),
                os.path.join(destination, img),
            )

print("Dataset split completed successfully!")