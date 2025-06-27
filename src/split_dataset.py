import os
import shutil
import random

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def split_train_test(src_img_dir, src_ann_dir, train_ratio=0.7):
    all_imgs = sorted(os.listdir(src_img_dir))
    paired_imgs = [f for f in all_imgs if os.path.exists(os.path.join(src_ann_dir, f + ".json"))]
    random.shuffle(paired_imgs)

    split_idx = int(len(paired_imgs) * train_ratio)
    train_imgs = paired_imgs[:split_idx]
    test_imgs = paired_imgs[split_idx:]

    # Crear carpetas
    for name in ["train_split", "test_split"]:
        ensure_dir(f"datasets/severstal/{name}/img")
        ensure_dir(f"datasets/severstal/{name}/ann")

    # Copiar train
    for fname in train_imgs:
        shutil.copy(os.path.join(src_img_dir, fname), f"datasets/severstal/train_split/img/{fname}")
        shutil.copy(os.path.join(src_ann_dir, fname + ".json"), f"datasets/severstal/train_split/ann/{fname}.json")

    # Copiar test
    for fname in test_imgs:
        shutil.copy(os.path.join(src_img_dir, fname), f"datasets/severstal/test_split/img/{fname}")
        shutil.copy(os.path.join(src_ann_dir, fname + ".json"), f"datasets/severstal/test_split/ann/{fname}.json")

    print(f"✔ Train split: {len(train_imgs)} images")
    print(f"✔ Test split:  {len(test_imgs)} images")

def create_subsets(train_split_dir, sizes):
    src_img_dir = os.path.join(train_split_dir, "img")
    src_ann_dir = os.path.join(train_split_dir, "ann")

    all_imgs = sorted(os.listdir(src_img_dir))
    paired_imgs = [f for f in all_imgs if os.path.exists(os.path.join(src_ann_dir, f + ".json"))]

    for subset_size in sizes:
        subset_dir = f"datasets/severstal/train_{subset_size}"
        img_dest = os.path.join(subset_dir, "img")
        ann_dest = os.path.join(subset_dir, "ann")
        ensure_dir(img_dest)
        ensure_dir(ann_dest)

        selected = random.sample(paired_imgs, subset_size)

        for fname in selected:
            shutil.copy(os.path.join(src_img_dir, fname), os.path.join(img_dest, fname))
            shutil.copy(os.path.join(src_ann_dir, fname + ".json"), os.path.join(ann_dest, fname + ".json"))

        print(f"✔ Subset train_{subset_size}: {subset_size} images")

if __name__ == "__main__":
    random.seed(42)

    base_img_dir = "/home/ptp/sam2/datasets/severstal/train/img"
    base_ann_dir = "/home/ptp/sam2/datasets/severstal/train/ann"

    # Paso 1: dividir en train_split y test_split
    split_train_test(base_img_dir, base_ann_dir, train_ratio=0.7)

    # Paso 2: crear subsets desde train_split
    subset_sizes = [25, 50, 100, 200, 500]
    create_subsets("datasets/severstal/train_split", subset_sizes)
