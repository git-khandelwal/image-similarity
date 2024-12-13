import sqlite3
import torch
import torch.optim as optim
import torch.nn as nn
from model import EncoderCNN
from dataset import ImageDataset, transform
from torch.utils.data import DataLoader, random_split
import numpy as np
import faiss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
np.random.seed(42)

# Initialize the database
conn = sqlite3.connect('features.db')
c = conn.cursor()

# Create a table to store features
c.execute('''
CREATE TABLE IF NOT EXISTS image_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature BLOB,
    image_path TEXT,
    prediction BLOB
)
''')

dataset = ImageDataset(folder_path='animals/animals', transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, _ = random_split(dataset, [train_size, test_size])

data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = EncoderCNN(embed_size=256, num_classes=90).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# FAISS
vectors = []

# Training
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    for idx, (images, image_paths, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        features, predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        # Store features in the database in the last epoch
        if epoch==num_epochs-1:
            with torch.no_grad():
                for i in range(images.size(0)):
                    feature = features[i].cpu().numpy()
                    feature_bytes = feature.tobytes()
                    prediction_index = torch.argmax(predictions[i]).item()
                    c.execute('INSERT INTO image_features (feature, image_path, prediction) VALUES (?, ?, ?)', (feature_bytes, image_paths[i], prediction_index))
                    
                    # Adding feature vectors to vectors list
                    vectors.append(feature)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'trained_model.pth')

vectors = np.array(vectors, dtype='float32')
index = faiss.IndexFlatL2(256)
index.add(vectors)
print(f"Number of vectors in the index: {index.ntotal}")
faiss.write_index(index, 'vector_index.faiss')

conn.commit()
conn.close()
