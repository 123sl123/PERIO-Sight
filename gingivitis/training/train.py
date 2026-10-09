"""Public image-level gingivitis training; no clinical aggregation or CAM generation."""

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet34_Weights, ResNet18_Weights
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
DATA_DIR = os.environ.get('GINGIVITIS_DATA_DIR', 'dataset/gingivitis')
OUTPUT_DIR = os.environ.get('GINGIVITIS_OUTPUT_DIR', 'outputs/gingivitis/training')
EXP_NAME = 'exp01_rn50_split_by_case_BOP_PD_pretrained_refine'
MODEL_NAME = 'resnet50'
PRETRAINED = True
PRETRAINED_MODEL_PATH = os.environ.get('GINGIVITIS_INIT_WEIGHTS') or None
device = torch.device(os.environ.get('GINGIVITIS_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, EXP_NAME)
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 15
RANDOM_SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(RANDOM_SEED)
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

class Datasets(Dataset):
    """Load class-organized images or explicitly labeled flat splits."""

    def __init__(self, data_dir, split='train', transform=None, classes=None, recursive=True):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f'Split directory not found: {self.split_dir}')
        self.image_paths = []
        self.labels = []
        class_dirs = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d)) and (not d.startswith('.'))])
        if class_dirs:
            if classes is not None:
                missing = [c for c in classes if c not in class_dirs]
                if missing:
                    raise ValueError(f'Classes {missing} not found in {self.split_dir}. Found: {class_dirs}')
                self.classes = list(classes)
            else:
                self.classes = class_dirs
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            for class_name in self.classes:
                class_dir = os.path.join(self.split_dir, class_name)
                if recursive:
                    for root, _, files in os.walk(class_dir):
                        for f in sorted(files):
                            if is_image_file(f):
                                self.image_paths.append(os.path.join(root, f))
                                self.labels.append(self.class_to_idx[class_name])
                else:
                    for f in sorted(os.listdir(class_dir)):
                        fp = os.path.join(class_dir, f)
                        if os.path.isfile(fp) and is_image_file(f):
                            self.image_paths.append(fp)
                            self.labels.append(self.class_to_idx[class_name])
        else:
            self.classes = list(classes) if classes is not None else None
            self._load_flat_split()
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.split_dir}. Expected either 'split/class/.../image' or 'split/image' + labels.csv.")
        paired = sorted(zip(self.image_paths, self.labels), key=lambda x: x[0])
        self.image_paths = [p[0] for p in paired]
        self.labels = [p[1] for p in paired]
        self._print_stats()

    def _load_flat_split(self):
        label_files = [os.path.join(self.split_dir, name) for name in ['labels.csv', 'labels.txt', 'annotations.csv', 'annotations.txt'] if os.path.isfile(os.path.join(self.split_dir, name))]
        if label_files:
            self._load_label_file(label_files[0])
            return
        if self.classes is None:
            raise ValueError(f'No class subdirectories or labels file found in {self.split_dir}. Please provide labels.csv (columns: image, label) or pass classes=[...].')
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for f in sorted(os.listdir(self.split_dir)):
            fp = os.path.join(self.split_dir, f)
            if os.path.isfile(fp) and is_image_file(f):
                label = self._infer_label_from_filename(f)
                if label is not None:
                    self.image_paths.append(fp)
                    self.labels.append(label)

    def _load_label_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(path)
            img_col = None
            label_col = None
            for col in df.columns:
                low = col.lower().strip()
                if low in ('image', 'path', 'file', 'filename', 'img'):
                    img_col = col
                if low in ('label', 'class', 'class_name', 'target', 'y'):
                    label_col = col
            if img_col is None or label_col is None:
                raise ValueError(f'labels.csv must contain image and label columns, got {list(df.columns)}')
        else:
            records = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        records.append({'image': parts[0], 'label': parts[1]})
            df = pd.DataFrame(records)
            img_col, label_col = ('image', 'label')
        if self.classes is None:
            self.classes = sorted(df[label_col].astype(str).unique().tolist())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for _, row in df.iterrows():
            img_path = str(row[img_col])
            if not os.path.isabs(img_path):
                img_path = os.path.join(self.split_dir, img_path)
            if os.path.isfile(img_path) and is_image_file(img_path):
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[str(row[label_col])])

    def _infer_label_from_filename(self, filename):
        filename_lower = filename.lower()
        for cls in self.classes:
            if cls.lower() in filename_lower:
                return self.class_to_idx[cls]
        return None

    def _print_stats(self):
        counts = {}
        for l in self.labels:
            counts[l] = counts.get(l, 0) + 1
        class_counts = {self.classes[idx]: cnt for idx, cnt in sorted(counts.items())}
        print(f'[Dataset {self.split}] Total: {len(self.image_paths)}, per class: {class_counts}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return (image, label)

def get_transforms(img_size=224, mode='train'):
    if mode == 'train':
        return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def create_model(model_name='resnet50', num_classes=2, pretrained=True, pretrained_model_path=None):
    if pretrained:
        if model_name == 'resnet18':
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet34':
            model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet50':
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet101':
            model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f'Unknown model: {model_name}')
    elif model_name == 'resnet18':
        model = models.resnet18(weights=None)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=None)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
    elif model_name == 'resnet101':
        model = models.resnet101(weights=None)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if pretrained_model_path:
        pretrained_model_path = os.path.expanduser(pretrained_model_path)
        if os.path.isfile(pretrained_model_path):
            print(f'Loading pretrained model weights from: {pretrained_model_path}')
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f'  Warning - missing keys: {missing}')
            if unexpected:
                print(f'  Warning - unexpected keys: {unexpected}')
        else:
            raise FileNotFoundError(f'Pretrained model path not found: {pretrained_model_path}')
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = (images.to(device), labels.to(device))
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return (epoch_loss, epoch_acc)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = (images.to(device), labels.to(device))
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc_score = roc_auc_score(all_labels, all_probs)
    return (epoch_loss, epoch_acc, precision, recall, f1, auc_score, all_labels, all_probs)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_auc_curve(val_aucs, save_path):
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(val_aucs) + 1)
    plt.plot(epochs, val_aucs, 'g-', linewidth=2, label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC During Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_roc_curve(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return roc_auc

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_training_log(log_file, message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    train_dataset = Datasets(DATA_DIR, split='train', transform=get_transforms(IMG_SIZE, mode='train'))
    val_dataset = Datasets(DATA_DIR, split='val', transform=get_transforms(IMG_SIZE, mode='val'), classes=train_dataset.classes)
    if train_dataset.classes != ['Gingivitis', 'Normal'] or set(train_dataset.labels) != {0, 1} or set(val_dataset.labels) != {0, 1}:
        raise ValueError('Training requires Gingivitis/Normal classes in both training and validation splits.')
    classes = train_dataset.classes
    num_classes = len(classes)
    print(f'=' * 60)
    print(f'ResNet Training - Oral Disease Classification')
    print(f'=' * 60)
    print(f'Device: {device}')
    print(f'Random Seed: {RANDOM_SEED}')
    print(f'Data Path: {DATA_DIR}')
    print(f'Output Path: {OUTPUT_PATH}')
    print(f'Model: {MODEL_NAME}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}')
    print(f'Learning Rate: {LEARNING_RATE}, Early Stopping Patience: {EARLY_STOPPING_PATIENCE}')
    print(f'Pretrained (torchvision): {PRETRAINED}')
    print(f"Pretrained Model Path: {(PRETRAINED_MODEL_PATH if PRETRAINED_MODEL_PATH else 'None')}")
    print(f'Classes detected: {classes}')
    print(f'Num Classes: {num_classes}')
    print(f'=' * 60)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f'\nDataset: Train={len(train_dataset)}, Val={len(val_dataset)}\n')
    model = create_model(MODEL_NAME, num_classes=num_classes, pretrained=PRETRAINED, pretrained_model_path=PRETRAINED_MODEL_PATH)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    log_file = os.path.join(OUTPUT_PATH, 'training_log.txt')
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = ([], [])
    train_accs, val_accs = ([], [])
    val_aucs = []
    history = []
    save_training_log(log_file, f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    save_training_log(log_file, f'=' * 60)
    save_training_log(log_file, f'Data Path: {DATA_DIR}')
    save_training_log(log_file, f'Classes: {classes}')
    save_training_log(log_file, f'Hyperparameters:')
    save_training_log(log_file, f'  Model: {MODEL_NAME}, Epochs: {EPOCHS}')
    save_training_log(log_file, f'  Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}')
    save_training_log(log_file, f'  Weight Decay: {WEIGHT_DECAY}, Early Stopping Patience: {EARLY_STOPPING_PATIENCE}')
    save_training_log(log_file, f'  Random Seed: {RANDOM_SEED}, Image Size: {IMG_SIZE}')
    save_training_log(log_file, f'=' * 60)
    best_val_f1 = 0.0
    best_val_auc = float('-inf')
    best_metrics = {}
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_labels, val_probs = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        epoch_time = time.time() - epoch_start
        save_training_log(log_file, f'Epoch {epoch:03d}/{EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f} | Time: {epoch_time:.1f}s')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'val_precision': val_precision, 'val_recall': val_recall, 'val_f1': val_f1, 'val_auc': val_auc})
        is_best = False
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_val_f1 = val_f1
            is_best = True
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, 'best_model.pth'))
            save_training_log(log_file, f'  -> New best model saved! Val AUC: {best_val_auc:.4f}, Acc: {best_val_acc:.4f}, F1: {best_val_f1:.4f}')
        else:
            epochs_no_improve += 1
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'val_precision': val_precision, 'val_recall': val_recall, 'val_f1': val_f1, 'val_auc': val_auc}, os.path.join(OUTPUT_PATH, 'checkpoint_epoch_{:03d}.pth'.format(epoch)))
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            save_training_log(log_file, f'\nEarly stopping triggered after {epoch} epochs')
            save_training_log(log_file, f'Best Val - Acc: {best_val_acc:.4f}, Prec: {best_val_precision:.4f}, Rec: {best_val_recall:.4f}, F1: {best_val_f1:.4f}, AUC: {best_val_auc:.4f}')
            break
    save_training_log(log_file, f"\n{'=' * 60}")
    save_training_log(log_file, f'Training completed!')
    save_training_log(log_file, f'Best Val - Acc: {best_val_acc:.4f}, Prec: {best_val_precision:.4f}, Rec: {best_val_recall:.4f}, F1: {best_val_f1:.4f}, AUC: {best_val_auc:.4f}')
    save_training_log(log_file, f"{'=' * 60}\n")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, os.path.join(OUTPUT_PATH, 'training_curves.png'))
    plot_auc_curve(val_aucs, os.path.join(OUTPUT_PATH, 'auc_curve.png'))
    model.load_state_dict(torch.load(os.path.join(OUTPUT_PATH, 'best_model.pth')))
    recheck_loss, recheck_acc, recheck_precision, recheck_recall, test_f1, recheck_auc, recheck_labels, recheck_probs = validate_epoch(model, val_loader, criterion, device)
    recheck_pred = (recheck_probs > 0.5).astype(int)
    recheck_roc_auc = plot_roc_curve(recheck_labels, recheck_probs, os.path.join(OUTPUT_PATH, 'roc_curve.png'))
    plot_confusion_matrix(recheck_labels, recheck_pred, classes, os.path.join(OUTPUT_PATH, 'confusion_matrix.png'))
    report = classification_report(recheck_labels, recheck_pred, target_names=classes)
    summary = f"{'=' * 60}\nFINAL RESULTS\n{'=' * 60}\nClasses: {classes}\nTotal Epochs: {len(train_losses)}\nBest Validation:\n  - Accuracy:  {best_val_acc:.4f}\n  - Precision: {best_val_precision:.4f}\n  - Recall:    {best_val_recall:.4f}\n  - F1-Score:  {best_val_f1:.4f}\n  - AUC:       {best_val_auc:.4f}\nValidation re-evaluation:\n  - Accuracy:  {recheck_acc:.4f}\n  - Precision: {recheck_precision:.4f}\n  - Recall:    {recheck_recall:.4f}\n  - F1-Score:  {test_f1:.4f}\n  - AUC:       {recheck_auc:.4f}\n\n{'=' * 60}\nClassification Report:\n{report}\n{'=' * 60}\n"
    print(summary)
    with open(os.path.join(OUTPUT_PATH, 'final_summary.txt'), 'w') as f:
        f.write(summary)
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_PATH, 'training_history.csv'), index=False)
    metrics = {'metric': ['accuracy', 'precision', 'recall', 'f1_score', 'auc'], 'best_val': [best_val_acc, best_val_precision, best_val_recall, best_val_f1, best_val_auc], 'validation_recheck': [recheck_acc, recheck_precision, recheck_recall, test_f1, recheck_auc]}
    pd.DataFrame(metrics).to_csv(os.path.join(OUTPUT_PATH, 'best_metrics.csv'), index=False)
    print(f"\n{'=' * 60}")
    print(f'ALL RESULTS SAVED TO: {OUTPUT_PATH}')
    print(f"{'=' * 60}")
    print(f'Files saved:')
    print(f'  - best_model.pth          ')
    print(f'  - training_log.txt        ')
    print(f'  - training_history.csv    ')
    print(f'  - training_curves.png     ')
    print(f'  - roc_curve.png           ')
    print(f'  - confusion_matrix.png    ')
    print(f'  - final_summary.txt       ')
    print(f'  - best_metrics.csv        ')
    print(f'  - checkpoint_epoch_*.pth  ')
    print(f"{'=' * 60}")
if __name__ == '__main__':
    main()
