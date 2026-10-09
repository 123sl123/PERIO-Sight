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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
DATA_DIR = os.environ.get('GINGIVITIS_DATA_DIR', 'dataset/gingivitis')
OUTPUT_DIR = os.environ.get('GINGIVITIS_OUTPUT_DIR', 'outputs/gingivitis/training')
EXP_NAME = 'v4_balanced_sampler_focal_labelsmooth_rot'
MODEL_NAME = 'resnet50'
PRETRAINED = True
PRETRAINED_MODEL_PATH = os.environ.get('GINGIVITIS_INIT_WEIGHTS') or None
device = torch.device(os.environ.get('GINGIVITIS_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, EXP_NAME)
LOSS_TYPE = 'FocalLoss_gamma2_LabelSmoothing'
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 15
RANDOM_SEED = 42
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1

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

class FocalLossWithLabelSmoothing(nn.Module):
    """Apply class-weighted focal modulation to smoothed cross entropy."""

    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (num_classes - 1) if num_classes > 1 else 0)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        ce_loss = -(true_dist * log_probs).sum(dim=1)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_weight = focal_weight * alpha_t
        loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class Datasets(Dataset):

    def __init__(self, data_dir, split='train', transform=None, classes=None, recursive=True):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f'Split directory not found: {self.split_dir}')
        self.image_paths, self.labels = ([], [])
        class_dirs = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d)) and (not d.startswith('.'))])
        if class_dirs:
            self.classes = list(classes) if classes else class_dirs
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
            raise RuntimeError(f'No images found in {self.split_dir}.')
        paired = sorted(zip(self.image_paths, self.labels), key=lambda x: x[0])
        self.image_paths = [p[0] for p in paired]
        self.labels = [p[1] for p in paired]
        self._print_stats()

    def _load_flat_split(self):
        label_files = [os.path.join(self.split_dir, n) for n in ['labels.csv', 'labels.txt', 'annotations.csv', 'annotations.txt'] if os.path.isfile(os.path.join(self.split_dir, n))]
        if label_files:
            self._load_label_file(label_files[0])
            return
        if self.classes is None:
            raise ValueError('No class subdirectories or labels file found.')
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
            img_col, label_col = (None, None)
            for col in df.columns:
                low = col.lower().strip()
                if low in ('image', 'path', 'file', 'filename', 'img'):
                    img_col = col
                if low in ('label', 'class', 'class_name', 'target', 'y'):
                    label_col = col
            if img_col is None or label_col is None:
                raise ValueError('labels.csv must contain image and label columns.')
        else:
            records = []
            with open(path) as f:
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

    def _infer_label_from_filename(self, f):
        fl = f.lower()
        for cls in self.classes:
            if cls.lower() in fl:
                return self.class_to_idx[cls]
        return None

    def _print_stats(self):
        cts = {}
        for l in self.labels:
            cts[l] = cts.get(l, 0) + 1
        print(f'[Dataset {self.split}] Total: {len(self.image_paths)}, per class: { {self.classes[i]: cts.get(i, 0) for i in range(len(self.classes))}}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return (img, label)

def get_transforms(img_size=224, mode='train'):
    if mode == 'train':
        return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(degrees=15), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def create_model(model_name='resnet50', num_classes=2, pretrained=True, pretrained_model_path=None):
    if pretrained:
        if model_name == 'resnet50':
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f'Unknown model: {model_name}')
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if pretrained_model_path:
        pp = os.path.expanduser(pretrained_model_path)
        if os.path.isfile(pp):
            print(f'Loading pretrained model weights from: {pp}')
            ckpt = torch.load(pp, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f'  Warning - missing keys: {missing}')
            if unexpected:
                print(f'  Warning - unexpected keys: {unexpected}')
        else:
            raise FileNotFoundError(f'Pretrained model path not found: {pp}')
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    rloss, corr, tot = (0.0, 0, 0)
    for x, y in dataloader:
        x, y = (x.to(device), y.to(device))
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        rloss += loss.item() * x.size(0)
        corr += (out.argmax(1) == y).sum().item()
        tot += y.size(0)
    return (rloss / tot, corr / tot)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    rloss, corr, tot = (0.0, 0, 0)
    labs, probs = ([], [])
    with torch.no_grad():
        for x, y in dataloader:
            x, y = (x.to(device), y.to(device))
            out = model(x)
            loss = criterion(out, y)
            rloss += loss.item() * x.size(0)
            corr += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
            labs.extend(y.cpu().numpy())
            probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
    labs = np.array(labs)
    probs = np.array(probs)
    preds = (probs > 0.5).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    return (rloss / tot, corr / tot, precision_score(labs, preds, zero_division=0), recall_score(labs, preds, zero_division=0), f1_score(labs, preds, zero_division=0), roc_auc_score(labs, probs), labs, probs)

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
    plt.plot(range(1, len(val_aucs) + 1), val_aucs, 'g-', lw=2, label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_roc_curve(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ra = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={ra:.4f})')
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return ra

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
    labels = np.array(train_dataset.labels)
    class_counts = np.bincount(labels)
    class_weights_sample = 1.0 / class_counts
    sample_weights = class_weights_sample[labels]
    sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.float64), num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    cw_tensor = torch.tensor(class_weights_sample / class_weights_sample.sum() * num_classes, dtype=torch.float32)
    criterion = FocalLossWithLabelSmoothing(alpha=cw_tensor, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTHING)
    print(f'\nDataset: Train={len(train_dataset)}, Val={len(val_dataset)}')
    print(f'  [V4] Class counts: {dict(zip(classes, class_counts.tolist()))}')
    print(f'  [V4] Focal gamma: {FOCAL_GAMMA}, Label Smoothing: {LABEL_SMOOTHING}')
    print(f'  [V4] Alpha: {dict(zip(classes, cw_tensor.tolist()))}')
    print(f'  [V4] Loss: {LOSS_TYPE}, Augmentation: HFlip+VFlip+Rotation(15)\n')
    model = create_model(MODEL_NAME, num_classes=num_classes, pretrained=PRETRAINED, pretrained_model_path=PRETRAINED_MODEL_PATH)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    log_file = os.path.join(OUTPUT_PATH, 'training_log.txt')
    epochs_no_improve = 0
    train_losses, val_losses, train_accs, val_accs, val_aucs = ([], [], [], [], [])
    history = []
    save_training_log(log_file, f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    save_training_log(log_file, f'Version: V4 - Balanced Sampler + Focal Loss + LabelSmoothing + Rotation')
    save_training_log(log_file, f'Loss: {LOSS_TYPE} | Gamma: {FOCAL_GAMMA} | LabelSmoothing: {LABEL_SMOOTHING}')
    save_training_log(log_file, f'Model: {MODEL_NAME} | Pretrained: {PRETRAINED}')
    save_training_log(log_file, f'LR: {LEARNING_RATE}, WD: {WEIGHT_DECAY}, Patience: {EARLY_STOPPING_PATIENCE}')
    save_training_log(log_file, f'=' * 60)
    best_val_f1 = 0.0
    best_val_auc = float('-inf')
    best_val_acc = best_val_loss = best_val_precision = best_val_recall = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_labels, val_probs = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        cm = confusion_matrix(val_labels, (val_probs > 0.5).astype(int))
        g_rec = cm[0, 0] / cm[0].sum() if cm.shape == (2, 2) and cm[0].sum() > 0 else 0
        n_rec = cm[1, 1] / cm[1].sum() if cm.shape == (2, 2) and cm[1].sum() > 0 else 0
        save_training_log(log_file, f'Epoch {epoch:03d}/{EPOCHS} | Train L:{train_loss:.4f} Acc:{train_acc:.4f} | Val L:{val_loss:.4f} Acc:{val_acc:.4f} Prec:{val_precision:.4f} Rec:{val_recall:.4f} F1:{val_f1:.4f} AUC:{val_auc:.4f} | G-R:{g_rec:.4f} N-R:{n_rec:.4f} | {time.time() - t0:.0f}s')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'val_precision': val_precision, 'val_recall': val_recall, 'val_f1': val_f1, 'val_auc': val_auc})
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, 'best_model.pth'))
            save_training_log(log_file, f'  -> New best! AUC:{best_val_auc:.4f} Acc:{best_val_acc:.4f} F1:{best_val_f1:.4f}')
        else:
            epochs_no_improve += 1
        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(OUTPUT_PATH, f'checkpoint_epoch_{epoch:03d}.pth'))
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            save_training_log(log_file, f'\nEarly stopping at epoch {epoch}')
            save_training_log(log_file, f'Best - Acc:{best_val_acc:.4f} Prec:{best_val_precision:.4f} Rec:{best_val_recall:.4f} F1:{best_val_f1:.4f} AUC:{best_val_auc:.4f}')
            break
    save_training_log(log_file, f"\n{'=' * 60}\nTraining completed!\nBest Val - Acc:{best_val_acc:.4f} AUC:{best_val_auc:.4f}\n{'=' * 60}\n")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, os.path.join(OUTPUT_PATH, 'training_curves.png'))
    plot_auc_curve(val_aucs, os.path.join(OUTPUT_PATH, 'auc_curve.png'))
    model.load_state_dict(torch.load(os.path.join(OUTPUT_PATH, 'best_model.pth')))
    recheck_loss, recheck_acc, recheck_precision, recheck_recall, test_f1, recheck_auc, recheck_labels, recheck_probs = validate_epoch(model, val_loader, criterion, device)
    recheck_pred = (recheck_probs > 0.5).astype(int)
    plot_roc_curve(recheck_labels, recheck_probs, os.path.join(OUTPUT_PATH, 'roc_curve.png'))
    plot_confusion_matrix(recheck_labels, recheck_pred, classes, os.path.join(OUTPUT_PATH, 'confusion_matrix.png'))
    report = classification_report(recheck_labels, recheck_pred, target_names=classes)
    summary = f"{'=' * 60}\nFINAL RESULTS (V4: Balanced + Focal GS={LABEL_SMOOTHING} R=15)\n{'=' * 60}\nClasses: {classes}\nTotal Epochs: {len(train_losses)}\nBest Val - Acc:{best_val_acc:.4f} Prec:{best_val_precision:.4f} Rec:{best_val_recall:.4f} F1:{best_val_f1:.4f} AUC:{best_val_auc:.4f}\nValidation re-evaluation - Acc:{recheck_acc:.4f} Prec:{recheck_precision:.4f} Rec:{recheck_recall:.4f} F1:{test_f1:.4f} AUC:{recheck_auc:.4f}\n\n{'=' * 60}\nClassification Report:\n{report}\n{'=' * 60}\n"
    print(summary)
    with open(os.path.join(OUTPUT_PATH, 'final_summary.txt'), 'w') as f:
        f.write(summary)
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_PATH, 'training_history.csv'), index=False)
    pd.DataFrame({'metric': ['accuracy', 'precision', 'recall', 'f1_score', 'auc'], 'best_val': [best_val_acc, best_val_precision, best_val_recall, best_val_f1, best_val_auc], 'validation_recheck': [recheck_acc, recheck_precision, recheck_recall, test_f1, recheck_auc]}).to_csv(os.path.join(OUTPUT_PATH, 'best_metrics.csv'), index=False)
    print(f"\n{'=' * 60}\nV4 ALL RESULTS SAVED TO: {OUTPUT_PATH}\n{'=' * 60}")
if __name__ == '__main__':
    main()
