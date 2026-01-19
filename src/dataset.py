import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class Ham10000Binary(Dataset):
    """
    HAM10000 binary dataset.

    Label mapping (edit if you want):
      suspicious (1): mel, akiec
      benign (0): nv, bkl, df, vasc, bcc
    """
    SUSPICIOUS = {"mel", "akiec"}
    BENIGN = {"nv", "bkl", "df", "vasc", "bcc"}
    ALLOWED = SUSPICIOUS | BENIGN

    def __init__(self, csv_path: str, images_dir: str, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        df = pd.read_csv(csv_path)

        # HAM10000 metadata columns
        if "image_id" not in df.columns or "dx" not in df.columns:
            raise ValueError("metadata CSV must contain columns: image_id, dx")

        df = df[df["dx"].isin(self.ALLOWED)].copy()

        # Map to binary
        df["label"] = df["dx"].apply(lambda x: 1 if x in self.SUSPICIOUS else 0)

        df["filename"] = df["image_id"].astype(str) + ".jpg"

        # Sanity check: ensure at least one expected file exists
        sample = df["filename"].iloc[0]
        if not os.path.exists(os.path.join(images_dir, sample)):
            raise FileNotFoundError(
                f"Example image not found: {sample}. "
                f"Check images_dir={images_dir} and that images are flattened."
            )

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label
