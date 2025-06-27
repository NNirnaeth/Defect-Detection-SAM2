import os
import random
import shutil

def select_200_samples(src_dir="datasets/severstal/train", dst_dir="datasets/severstal/train_200", n=200):
    ann_dir = os.path.join(src_dir, "ann")
    img_dir = os.path.join(src_dir, "img")

    dst_ann = os.path.join(dst_dir, "ann")
    dst_img = os.path.join(dst_dir, "img")

    os.makedirs(dst_ann, exist_ok=True)
    os.makedirs(dst_img, exist_ok=True)

    # Get list of valid JSONs with their corresponding image
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
    paired_files = []
    for ann_file in ann_files:
        img_file = ann_file.replace(".json", "")
        if os.path.exists(os.path.join(img_dir, img_file)):
            paired_files.append((img_file, ann_file))

    # Randomly select 200 pairs
    selected = random.sample(paired_files, min(n, len(paired_files)))

    for img_file, ann_file in selected:
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(dst_img, img_file))
        shutil.copy(os.path.join(ann_dir, ann_file), os.path.join(dst_ann, ann_file))

    print(f" Copied {len(selected)} pairs to {dst_dir}")

if __name__ == "__main__":
    select_200_samples()

