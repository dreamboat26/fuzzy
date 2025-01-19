import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Load and Preprocess the Data
def load_data():
    categories = ['comp.graphics', 'sci.space', 'rec.sport.baseball', 'talk.politics.mideast']
    data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    return data.data, data.target, data.target_names

def preprocess_data(texts, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(texts).toarray()
    return X, vectorizer

texts, labels, target_names = load_data()
X, vectorizer = preprocess_data(texts)
y = np.array(labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define the Lagrangean Neural Network (LNN) with Dynamic Transformations
class LNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LNN, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size)  # New projection layer
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        
        # Self-attention mechanism for dynamic adjacency
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        self.lambda_reg = 0.1  # Regularization parameter
    
    def forward(self, x):
        # Project the input to the hidden size
        x = torch.relu(self.input_to_hidden(x))
        x = self.dropout(x)
        return self.fc2(x)
    
    def compute_dynamic_adjacency(self, batch_data):
        """
        Constructs a dynamic adjacency matrix using self-attention.
        """
        # Project the input to the hidden size before using self-attention
        batch_data = torch.relu(self.input_to_hidden(batch_data))
        batch_data = batch_data.unsqueeze(1)  # Add sequence dimension for attention
        attn_output, attn_weights = self.attention(batch_data, batch_data, batch_data)
        adjacency_matrix = torch.mean(attn_weights, dim=1).squeeze(1)  # Average across heads
        return adjacency_matrix
    
    def lagrangian_loss(self, output, target, batch_data):
        """
        Combines Cross-Entropy Loss with Structural Regularization.
        """
        # Cross-Entropy loss for classification
        ce_loss = nn.CrossEntropyLoss()(output, target)
        
        # Compute dynamic adjacency matrix using attention
        adjacency_matrix = self.compute_dynamic_adjacency(batch_data)
        
        # Apply transformations to outputs and enforce consistency with the adjacency matrix
        transformed_output = torch.softmax(output, dim=1)
        
        # Structural regularization using dynamic adjacency matrix
        structural_reg = torch.mean(torch.abs(adjacency_matrix @ transformed_output))
        
        # Combined loss
        return ce_loss + self.lambda_reg * structural_reg

# Step 3: Training the LNN Model with Dynamic Transformations
def train_model(model, X_train, y_train, X_test, y_test, epochs=20, lr=0.01, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    test_data = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)
    
    num_batches = len(train_data) // batch_size
    for epoch in range(epochs):
        model.train()
        for i in range(num_batches):
            # Get the current batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_data = train_data[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            # Calculate Lagrangean loss with dynamic adjacency matrix
            loss = model.lagrangian_loss(outputs, batch_labels, batch_data)
            loss.backward()
            optimizer.step()
        
        # Periodic evaluation
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data)
        _, predictions = torch.max(test_outputs, 1)
        print("\nClassification Report:\n", classification_report(test_labels, predictions, target_names=target_names))

# Step 4: Initialize and Train the Model with Dynamic Transformations
input_size = X_train.shape[1]  # 1000 (from TF-IDF)
hidden_size = 100
output_size = len(set(y))

model = LNN(input_size, hidden_size, output_size)
train_model(model, X_train, y_train, X_test, y_test, epochs=20, lr=0.01, batch_size=64)
