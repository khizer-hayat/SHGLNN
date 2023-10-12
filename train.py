import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.metrics import f1_score
from model import SHGLNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters and settings
EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 0.005
NUM_RUNS = 10

criterion = torch.nn.CrossEntropyLoss()

def train_and_evaluate(dataset_name):
    dataset = TUDataset(root='./data', name=dataset_name)
    
    # Instantiate the model
    model = SHGLNN(in_channels=dataset.num_node_features, hidden_channels=256, out_channels=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Split the dataset
    train_dataset = dataset[:int(0.3 * len(dataset))]
    val_dataset = dataset[int(0.3 * len(dataset)):int(0.4 * len(dataset))]
    test_dataset = dataset[int(0.4 * len(dataset)):]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    for run in range(NUM_RUNS):
        for epoch in range(EPOCHS):
            # --------- Training ---------
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_predictions = []
            train_labels = []

            for batch in train_loader:
                optimizer.zero_grad()
                data = batch.to(device)
                outputs = model(data)
                loss = criterion(outputs, data.y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += data.y.size(0)
                train_correct += (predicted == data.y).sum().item()
                train_predictions.extend(predicted.cpu().numpy())
                train_labels.extend(data.y.cpu().numpy())

            train_accuracy = 100 * train_correct / train_total
            train_f1 = f1_score(train_labels, train_predictions, average='weighted')
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f} %, Train F1: {train_f1:.4f}')

            # --------- Validation ---------
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_predictions = []
            val_labels = []
            
            for data in val_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data.y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += data.y.size(0)
                val_correct += (predicted == data.y).sum().item()
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(data.y.cpu().numpy())

            val_accuracy = 100 * val_correct / val_total
            val_f1 = f1_score(val_labels, val_predictions, average='weighted')
            print(f'Run [{run+1}/{NUM_RUNS}], Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f} %, Validation F1: {val_f1:.4f}')

        # --------- Testing ---------
        test_loss = 0
        test_correct = 0
        test_total = 0
        test_predictions = []
        test_labels = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data.y)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += data.y.size(0)
                test_correct += (predicted == data.y).sum().item()
                test_predictions.extend(predicted.cpu().numpy())
                test_labels.extend(data.y.cpu().numpy())

        test_accuracy = 100 * test_correct / test_total
        test_f1 = f1_score(test_labels, test_predictions, average='weighted')
        print(f'Run [{run+1}/{NUM_RUNS}], Testing Loss: {test_loss/len(test_loader):.4f}, Testing Accuracy: {test_accuracy:.2f} %, Testing F1: {test_f1:.4f}')

if __name__ == "__main__":
    dataset_names = ['MUTAG', 'NCI1', 'PROTEINS', 'ENZYMES', 'IMDB-BINARY', 'COLLAB']

    for dataset_name in dataset_names:
        print(f"\nTraining for {dataset_name}")
        train_and_evaluate(dataset_name)
