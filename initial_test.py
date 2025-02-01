import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

from dwave.system import DWaveSampler, EmbeddingComposite


def load_iris_data(file_path='iris.csv'):
    """
    Loads the iris dataset from a CSV file.
    Assumes the CSV has columns: sepal_length, sepal_width, petal_length, petal_width, species.
    """
    df = pd.read_csv(file_path)

    label_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    df['label'] = df['species'].map(label_mapping)

    # Extract features and labels.
    features = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values.astype(np.float32)
    labels = df['label'].values.astype(np.int64)
    return features, labels


class IrisNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super(IrisNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class QuantumAnnealingOptimizer(optim.Optimizer):
    """
    This optimizer uses D-Wave's quantum annealer to choose a binary update for each parameter.
    For each parameter element, a fixed update step (+lr or -lr) is chosen by solving a simple QUBO.
    """

    def __init__(self, params, lr=0.01, num_reads=10):
        defaults = dict(lr=lr, num_reads=num_reads)
        super(QuantumAnnealingOptimizer, self).__init__(params, defaults)
        self.sampler = EmbeddingComposite(DWaveSampler())

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            num_reads = group.get('num_reads', 10)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.detach().cpu().numpy().flatten()
                n = grad.shape[0]

                # Build a QUBO with only diagonal terms.
                Q = {}
                for i in range(n):
                    Q[(i, i)] = -2 * lr * grad[i]

                try:
                    sampleset = self.sampler.sample_qubo(Q, num_reads=num_reads)
                    best_sample = sampleset.first.sample
                except Exception as e:
                    # Fallback to a simple sign-based update if quantum sampling fails.
                    print("D-Wave sampling failed, falling back to classical update.")
                    update_direction = np.where(grad > 0, -1.0, 1.0)
                    best_sample = {i: 0 if update_direction[i] > 0 else 1 for i in range(n)}

                # Map binary decision to update: 0 -> +lr, 1 -> -lr.
                updates = lr * (1 - 2 * np.array([best_sample[i] for i in range(n)]))
                updates_tensor = torch.tensor(updates, dtype=p.data.dtype, device=p.data.device).view_as(p.data)
                p.data.add_(updates_tensor)
        return loss


# -----------------------------
# 4. Training Loop with Timing and Loss Metrics
# -----------------------------
def train_model(model, optimizer, loss_fn, dataloader, num_epochs):
    model.train()
    total_training_time = 0.0
    total_samples = len(dataloader.dataset)

    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        total_loss = 0.0

        for batch_data, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_data.size(0)

        epoch_time = time.perf_counter() - epoch_start_time
        avg_loss = total_loss / total_samples
        throughput = total_samples / epoch_time

        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}, Time = {epoch_time:.2f} sec, "
              f"Throughput = {throughput:.2f} samples/sec")
        total_training_time += epoch_time

    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")


# -----------------------------
# 5. Evaluation Function
# -----------------------------
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            _, preds = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
    return accuracy


# -----------------------------
# 6. Main Function
# -----------------------------
def main():
    features, labels = load_iris_data('iris.csv')

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, train_size=25, test_size=125, random_state=42, stratify=labels)

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = IrisNet(input_dim=4, hidden_dim=16, output_dim=3)

    loss_fn = nn.CrossEntropyLoss()

    # optimizer = QuantumAnnealingOptimizer(model.parameters(), lr=0.01, num_reads=10)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Before training:")
    evaluate_model(model, test_loader)

    num_epochs = 100
    train_model(model, optimizer, loss_fn, train_loader, num_epochs)

    print("\nAfter training:")
    evaluate_model(model, test_loader)


if __name__ == '__main__':
    main()
