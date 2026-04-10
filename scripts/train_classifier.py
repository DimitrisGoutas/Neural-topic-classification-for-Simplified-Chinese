import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# defining the neural network class
class Classifier(nn.Module):
    def __init__(self, vector_size, hidden_dim, num_labels):
        super(Classifier, self).__init__()

        # connected layers
        self.fc1 = nn.Linear(vector_size, hidden_dim)         
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)    
        self.fc3 = nn.Linear(hidden_dim // 2, num_labels)  


    # define how data will move through the network
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # give score for each class
        x = self.fc3(x)  

        return x


# training loop
def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

# validation loop
def eval_loop(dataloader, model, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument('--embeddings_file', required=True, help="Path to the file with embeddings and labels")
    parser.add_argument('--output_model', required=True, help="Create the file to save the trained model")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Number of neurons in hidden layer")
    parser.add_argument('--val_embeddings_file', type=str, help="Validation embeddings")
    parser.add_argument('--plot_file', default='training_plot.png', help='Path to save the training plot image')
    args = parser.parse_args()


    # load embeddings and labels
    data = np.load(args.embeddings_file)
    X = data["embeddings"] 
    y = data["labels"]     

    # get all unique labels
    unique_labels = sorted(list(set(y)))     
    # map labels with numbers             
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    # replace labels with numbers
    y_idx = np.array([label_to_idx[label] for label in y]) 


    # convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_idx, dtype=torch.long)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # number of features in each input
    vector_size = X.shape[1]  
    # number of classes   
    num_labels = len(unique_labels) 

    model = Classifier(vector_size, args.hidden_dim, num_labels).to(device)

    # create DataLoader for batching
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # measuring how wrong predictions are and use Adam optimizer to reduce loss
    criterion = nn.CrossEntropyLoss()    
    optimizer = optim.Adam(model.parameters())  

    # load validation data 
    val_loader = None
    if args.val_embeddings_file:
        val_data = np.load(args.val_embeddings_file)
        X_val = val_data["embeddings"]
        y_val = val_data["labels"]

        y_val_idx = np.array([label_to_idx[label] for label in y_val])

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_idx, dtype=torch.long)

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    train_losses = []
    val_accuracies = []


    # training loop
    for epoch in range(args.epochs):
        train_loss = train_loop(dataloader, model, criterion, optimizer, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")
        if val_loader:
            val_acc = eval_loop(val_loader, model, device)
            val_accuracies.append(val_acc)
            print(f"Validation Accuracy={val_acc:.4f}")

    if val_loader:
        plt.plot(range(1, args.epochs+1), val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy over Epochs")
        plt.legend()
        plt.savefig(args.plot_file)
        print("Saved training_plot file")


    torch.save({
        'model_state_dict': model.state_dict(),
        'label_to_idx': label_to_idx,
        'vector_size': vector_size,
        'hidden_dim': args.hidden_dim,
        'num_labels': num_labels
    }, args.output_model)

    print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    main()
