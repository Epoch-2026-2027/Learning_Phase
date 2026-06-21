from pathlib import Path
from PIL import Image
import kagglehub
from tqdm import tqdm


IMG_SIZE = 512

DIR = Path(kagglehub.dataset_download("balraj98/massachusetts-roads-dataset"))
PROCESSED_DIR = Path(r"C:\Users\tilak\OneDrive\Documents\Tilak Asodariya\AI ML\Epoch\CORES\Task-2\SubTask-1\processed")

DATA = DIR / "tiff"

for folder in ["train","train_labels","val","val_labels","test","test_labels"]:
    (PROCESSED_DIR / folder).mkdir(parents=True,exist_ok=True)


def process_split(image_dir, mask_dir, out_image_dir, out_mask_dir):
    image_files = sorted(image_dir.glob("*.tiff"))
    print(f"{image_dir.name}: {len(image_files)} images")
    for img_path in tqdm(image_files, desc=image_dir.name):
        mask_path = mask_dir / f"{img_path.stem}.tif"
        
        #image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        image.save(out_image_dir / f"{img_path.stem}.png")

        #mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
        mask.save(out_mask_dir / f"{mask_path.stem}.png")

process_split(DATA / "train",DATA / "train_labels",PROCESSED_DIR / "train",PROCESSED_DIR / "train_labels")
process_split(DATA / "val",DATA / "val_labels",PROCESSED_DIR / "val",PROCESSED_DIR / "val_labels")
process_split(DATA / "test",DATA / "test_labels",PROCESSED_DIR / "test",PROCESSED_DIR / "test_labels")

for split in ["train", "val", "test"]:
    imgs = len(list((PROCESSED_DIR / split).glob("*.png")))
    masks = len(list((PROCESSED_DIR / f"{split}_labels").glob("*.png")))

    print(f"{split}: {imgs} images | {masks} masks")