import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# prepare data
def load_and_preprocess_adult_data(path):
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    df = pd.read_csv(path, names=columns, na_values=" ?", skipinitialspace=True)

    # cleaning
    df = df.dropna()

    #  the target column
    df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'income']
    df = pd.get_dummies(df, columns=categorical_cols)

    X = df.drop('income', axis=1).values
    y = df['income'].values

    # Normalisazion
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return (X_train, y_train), (X_test, y_test)


# load data
(train_X, train_y), (test_X, test_y) = load_and_preprocess_adult_data("C:/Users/HP/Documents/adult/adult.data")

# convert PyTorch
train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.long)
test_X = torch.tensor(test_X, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


model = SimpleNN(input_dim=train_X.shape[1])

# Confidentiality
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# training
model.train()
for epoch in range(10):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f}")

# evaluation
model.eval()
with torch.no_grad():
    outputs = model(test_X)
    preds = outputs.argmax(dim=1)
    accuracy = (preds == test_y).float().mean()
    print(f"\nTest Accuracy with DP: {accuracy:.4f}")

# epsilon after training
epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"Epsilon (ε) after training: {epsilon:.2f}")
