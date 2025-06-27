import os
import pandas as pd
from sklearn.model_selection import train_test_split

## Prepare the Chest X-Ray segmentation dataset (images + masks from CSV)
def prepare_dataset(data_dir="datasets/chest"):
    """
    Prepares dataset with structure:
    - images in: data_dir/images/images/
    - masks in:  data_dir/masks/masks/
    - CSV: data_dir/train.csv with columns: ImageId, MaskId

    Returns:
        train_data: list of dicts {"image": path, "annotation": path}
        test_data: same
    """
    images_dir = os.path.join(data_dir, "images", "images")
    masks_dir = os.path.join(data_dir, "masks", "masks")
    csv_path = os.path.join(data_dir, "train.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    def build_list(df):
        return [
            {
                "image": os.path.join(images_dir, row["ImageId"]),
                "annotation": os.path.join(masks_dir, row["MaskId"])
            }
            for _, row in df.iterrows()
        ]

    train_data = build_list(train_df)
    test_data = build_list(test_df)

    print(f"Chest dataset loaded -> Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data

