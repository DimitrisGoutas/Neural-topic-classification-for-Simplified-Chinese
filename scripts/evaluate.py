import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# evaluation loop
def evaluate(model, dataloader, device):
    # set model to evaluation mode
    model.eval()  
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)  
            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(y_batch.numpy())  

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, cm

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained classifier on test set")
    parser.add_argument('--embeddings_file', required=True, help="Path to .npz embeddings file")
    parser.add_argument('--model_file', required=True, help="Path to trained PyTorch model (.pt)")
    args = parser.parse_args()

    # load test embeddings and labels
    data = np.load(args.embeddings_file)
    X_test = data["embeddings"]
    y_test = data["labels"]

    checkpoint = torch.load(args.model_file, map_location=torch.device('cpu'), weights_only=False)
    vector_size = checkpoint['vector_size']
    hidden_dim = checkpoint['hidden_dim']
    num_labels = checkpoint['num_labels']
    label_to_idx = checkpoint['label_to_idx']

    # convert string labels to indices using the saved mapping
    y_idx = np.array([label_to_idx[label] for label in y_test])

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_idx, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # initialize model and load saved weights
    model = Classifier(vector_size, hidden_dim, num_labels).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # evaluate
    acc, cm = evaluate(model, dataloader, device)

    print(f"Test Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()