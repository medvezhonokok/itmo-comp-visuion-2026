import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# Цвета из реального датасета DVM (Unlisted/редкие исключены)
COLOR_CLASSES = [
    'beige', 'black', 'blue', 'bronze', 'brown', 'green',
    'grey', 'orange', 'purple', 'red', 'silver', 'white', 'yellow'
]

COLOR2IDX = {c: i for i, c in enumerate(COLOR_CLASSES)}

# Маппинг нестандартных названий из DVM -> наши классы
COLOR_ALIASES = {
    'gold': 'yellow',
    'turquoise': 'blue',
    'navy': 'blue',
    'indigo': 'blue',
    'maroon': 'red',
    'burgundy': 'red',
    'magenta': 'purple',
    'pink': 'purple',
    'multicolour': None,  # пропускаем
    'unlisted': None,     # пропускаем
}


def normalize_color(color_str):
    """Приводит цвет из имени файла DVM к одному из COLOR_CLASSES."""
    c = color_str.strip().lower()
    if c in COLOR2IDX:
        return c
    mapped = COLOR_ALIASES.get(c)
    if mapped is None:
        return None
    return mapped


def get_transforms(train=True, img_size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


class DVMDataset(Dataset):
    """
    DVM-CAR dataset loader.
    Expects either:
      - A CSV with columns [image_path, color] where image_path is absolute or relative to root_dir
      - OR a root_dir organized as root_dir/<color>/<image>.jpg (ImageFolder style)
    """

    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (path, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def load_dvm_confirmed_fronts(root_dir, test_size=0.2, val_size=0.1, seed=42):
    """
    Загружает DVM confirmed_fronts.
    Структура: root_dir/Make/Year/Make$$Model$$Year$$Color$$...$$image_N.jpg
    Цвет извлекается из имени файла (4-й элемент через $$).
    """
    samples = []
    for make in os.listdir(root_dir):
        make_dir = os.path.join(root_dir, make)
        if not os.path.isdir(make_dir):
            continue
        for year in os.listdir(make_dir):
            year_dir = os.path.join(make_dir, year)
            if not os.path.isdir(year_dir):
                continue
            for fname in os.listdir(year_dir):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                parts = fname.split('$$')
                if len(parts) < 4:
                    continue
                color = normalize_color(parts[3])
                if color is None:
                    continue
                samples.append((os.path.join(year_dir, fname), COLOR2IDX[color]))

    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    val_relative = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_relative, stratify=train_labels, random_state=seed
    )

    return (
        list(zip(train_paths, train_labels)),
        list(zip(val_paths, val_labels)),
        list(zip(test_paths, test_labels)),
    )


def load_from_folder(root_dir, test_size=0.2, val_size=0.1, seed=42):
    """Load dataset from ImageFolder-style directory (Make/Year/filename$$Color$$...)."""
    return load_dvm_confirmed_fronts(root_dir, test_size, val_size, seed)


def load_from_csv(csv_path, root_dir=None, test_size=0.2, val_size=0.1, seed=42):
    """Load dataset from CSV file with columns [image_path, color]."""
    df = pd.read_csv(csv_path)
    df['color'] = df['color'].str.lower()
    df = df[df['color'].isin(COLOR2IDX)]

    if root_dir:
        df['image_path'] = df['image_path'].apply(lambda p: os.path.join(root_dir, p))

    samples = list(zip(df['image_path'].tolist(), df['color'].map(COLOR2IDX).tolist()))
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    val_relative = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_relative, stratify=train_labels, random_state=seed
    )

    return (
        list(zip(train_paths, train_labels)),
        list(zip(val_paths, val_labels)),
        list(zip(test_paths, test_labels)),
    )


def get_loaders(train_samples, val_samples, test_samples, batch_size=32, num_workers=4):
    train_ds = DVMDataset(train_samples, transform=get_transforms(train=True))
    val_ds = DVMDataset(val_samples, transform=get_transforms(train=False))
    test_ds = DVMDataset(test_samples, transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
