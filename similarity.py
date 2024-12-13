import sqlite3
import numpy as np
import torch
import cv2
from model import EncoderCNN
from dataset import ImageDataset, transform, animal_array
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import faiss

torch.manual_seed(42)
np.random.seed(42)

def load_model(model_path, device):
    model = EncoderCNN(embed_size=256, num_classes=90)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_top_similar(feature, conn, top_n=5, threshold=0.6):
    c = conn.cursor()
    c.execute('SELECT feature, image_path, prediction FROM image_features')
    rows = c.fetchall()

    # Calculate similarities using cosine similarity
    similarities = []
    for row in rows:
        db_feature = np.frombuffer(row[0], dtype=np.float32)
        similarity = np.dot(feature, db_feature) / (np.linalg.norm(feature) * np.linalg.norm(db_feature))
        if similarity >= threshold:
            similarities.append((similarity, row[1], row[2]))  # Store similarity, image path and label

    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_n]

def get_top_similar_faiss(feature, faiss_index="vector_index.faiss", top_n=5):
    index = faiss.read_index(faiss_index)
    _, indices = index.search(feature, top_n)
    
    conn = sqlite3.connect('features.db')
    cursor = conn.cursor()
    nearest_indices = [int(idx)+1 for idx in indices[0]]
    placeholders = ','.join(['?'] * len(nearest_indices))
    query = f"SELECT image_path, prediction FROM image_features WHERE id IN ({placeholders})"
    cursor.execute(query, nearest_indices)
    results = cursor.fetchall()
    results = [list(result) for result in results]
    return results

def evaluate_model(model, test_loader, device, conn):
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for images, _, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features, predictions = model(images)
            
            predicted_classes = torch.argmax(predictions, dim=1)
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    # Calculate precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_true_labels, all_predictions, average='weighted'
    )
    
    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # Visualize confusion matrix
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate similarity scores for each test image
    similarity_results = []
    for i, (images, image_paths, labels) in enumerate(test_loader):
        images = images.to(device)
        with torch.no_grad():
            features, predictions = model(images)
            
        for j in range(len(images)):
            feature = features[j].cpu().numpy()
            true_label = labels[j].item()
            predicted_label = torch.argmax(predictions[j]).item()
            
            top_similar = get_top_similar(feature, conn)
            
            similar_label_matches = [
                sim_label == true_label for _, _, sim_label in top_similar
            ]
            
            similarity_results.append({
                'image_path': image_paths[j],
                'true_label': true_label,
                'predicted_label': predicted_label,
                'top_similar': top_similar,
                'similar_label_matches': similar_label_matches
            })
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'similarity_results': similarity_results
    }

def get_test_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    _, test_dataset = random_split(dataset, [train_size, test_size])
    return test_dataset

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('trained_model.pth', device)
    
    dataset = ImageDataset(folder_path='animals/animals', transform=transform)
    
    test_dataset = get_test_dataset(dataset)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    conn = sqlite3.connect('features.db')
    
    results = evaluate_model(model, test_loader, device, conn)
    
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    for result in results['similarity_results'][:10]:  # Print first 10 results
        print("\nImage Path:", result['image_path'])
        print("True Label:", animal_array[result['true_label']])
        print("Predicted Label:", animal_array[result['predicted_label']])
        print("Similar Images:")
        for sim, img_path, sim_label in result['top_similar']:
            print(f"  - Similarity: {sim:.4f}, Path: {img_path}, Label: {animal_array[sim_label]}")
    
    conn.close()