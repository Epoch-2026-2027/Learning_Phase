import shutil
from pathlib import Path

folder = Path('./archive_deepglobe/train')
for file in folder.iterdir():
    if file.is_file():
        if "_mask" in file.name:
            shutil.move('./archive_deepglobe/train/'+file.name, './sorted_data/masks/'+file.name)
        if "_sat" in file.name:
            shutil.move('./archive_deepglobe/train/'+file.name, './sorted_data/sats/'+file.name)

print("Done!")