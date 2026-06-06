import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import nbformat as nbf

# 1. Configuration & Vocabulary
DATA_DIR = 'CIG_PS_AIML/cig_ps'
TRAIN_LABELS = os.path.join(DATA_DIR, 'train-labels.csv')
TRAIN_DIR = os.path.join(DATA_DIR, 'train_images')
TEST_DIR = os.path.join(DATA_DIR, 'test_images')

df = pd.read_csv(TRAIN_LABELS)

# Extract vocabulary
all_chars = set()
for text in df['text'].values:
    all_chars.update(list(str(text)))

vocab = sorted(list(all_chars))
char2idx = {c: i + 1 for i, c in enumerate(vocab)} # 0 is reserved for CTC blank
idx2char = {i + 1: c for i, c in enumerate(vocab)}
idx2char[0] = '-'
num_classes = len(vocab) + 1

# 2. Dataset
class CaptchaDataset(Dataset):
    def __init__(self, df, img_dir, char2idx, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.char2idx = char2idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        
        text = str(row['text'])
        target = [self.char2idx[c] for c in text]
        return image, torch.tensor(target, dtype=torch.long), row['image']

transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Use small subset for quick training since this is an example
train_dataset = CaptchaDataset(df, TRAIN_DIR, char2idx, transform=transform)

def collate_fn(batch):
    images, targets, paths = zip(*batch)
    images = torch.stack(images)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets)
    return images, targets, target_lengths, paths

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# 3. Model
class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        # 64 / 16 = 4, 128 / 4 = 32
        self.rnn = nn.LSTM(256 * 4, hidden_size, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv_out = self.cnn(x)
        b, c, h, w = conv_out.size()
        conv_out = conv_out.view(b, c * h, w)
        conv_out = conv_out.permute(0, 2, 1) # b, seq_len, features
        rnn_out, _ = self.rnn(conv_out)
        out = self.fc(rnn_out)
        return nn.functional.log_softmax(out, dim=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(num_classes).to(device)

# 4. Training
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5 # 5 for fast demo
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for images, targets, target_lengths, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images) # [b, w, c]
        outputs = outputs.permute(1, 0, 2) # [w, b, c]
        
        input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long).to(device)
        
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}')

# 5. Testing and Submission
test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.png')]
test_df = pd.DataFrame({'image': test_files, 'text': [''] * len(test_files)})
test_dataset = CaptchaDataset(test_df, TEST_DIR, char2idx, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

model.eval()
predictions = []
with torch.no_grad():
    for images, _, _, paths in tqdm(test_loader, desc='Testing'):
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(2) # b, seq_len
        preds = preds.cpu().numpy()
        
        for i in range(len(preds)):
            pred_chars = []
            for j in range(len(preds[i])):
                if preds[i][j] != 0 and (not (j > 0 and preds[i][j - 1] == preds[i][j])):
                    pred_chars.append(idx2char[preds[i][j]])
            predictions.append((paths[i], ''.join(pred_chars)))

sub_df = pd.DataFrame(predictions, columns=['image', 'prediction'])
sub_df.to_csv('submission.csv', index=False)

# 6. Generate Notebook
nb = nbf.v4.new_notebook()
code = open(__file__).read()
nb['cells'] = [
    nbf.v4.new_markdown_cell("# Distorted Visual Sequence Pattern Recognition\nThis notebook demonstrates how to load the dataset, build a CRNN model, train it with CTC loss, and predict sequences."),
    nbf.v4.new_code_cell(code)
]
with open('train_submission.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Saved submission.csv and train_submission.ipynb")
