import torch
from PIL import Image
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, meta_df, transform, with_label=False):
        self.records = meta_df.reset_index(drop=True)
        self.transform = transform
        self.with_label = with_label

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.loc[idx]
        img = Image.open(row['out_path']).convert('RGB')
        img = self.transform(img)
        if self.with_label:
            label = torch.tensor(int(row['male']), dtype=torch.long)
            return img, label
        return img
