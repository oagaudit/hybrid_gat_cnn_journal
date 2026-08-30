"""
src/models/cnn_model.py
CNN Architecture for Bid Rotation Image Classification (Stage 1)
Input: 96x96 grayscale images
Output: Binary classification logits (0 = competitive, 1 = collusive)
Note: No sigmoid in forward() - use BCEWithLogitsLoss for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BidRotationCNN(nn.Module):
    """
    CNN for classifying bid rotation images.
    Extracts 64-dimensional embeddings before final classification.
    """
    
    def __init__(self, embedding_dim=64, dropout_rate=0.5):
        """
        Args:
            embedding_dim: Dimension of embedding vector (default: 64)
            dropout_rate: Dropout probability for regularization
        """
        super(BidRotationCNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size after convolutions
        # Input: 96x96
        # After 4 pooling layers (stride=2 each): 96 -> 48 -> 24 -> 12 -> 6
        self.flattened_size = 256 * 6 * 6  # = 9216
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Embedding layer (output for Stage 2)
        self.embedding = nn.Linear(256, embedding_dim)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)
        
        # Classification layer (returns logits, NOT sigmoid)
        self.classifier = nn.Linear(embedding_dim, 1)
        
    def forward(self, x, return_embedding=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 96, 96)
            return_embedding: If True, return embedding vector instead of prediction
        
        Returns:
            If return_embedding=False: (batch_size, 1) - raw logits
            If return_embedding=True: (batch_size, embedding_dim) - embedding vector
        """
        # Convolutional blocks with BatchNorm and ReLU
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 96 -> 48
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 48 -> 24
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 24 -> 12
        x = self.pool(F.relu(self.bn4(self.conv4(x))))   # 12 -> 6
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Embedding
        embedding = self.bn_embedding(self.embedding(x))
        
        if return_embedding:
            return embedding
        
        # Classification - return raw logits (no sigmoid)
        logits = self.classifier(embedding)
        return logits
    
    def extract_embedding(self, x):
        """
        Convenience method to extract embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 96, 96)
        
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        return self.forward(x, return_embedding=True)
    
    def predict_probability(self, x):
        """
        Convenience method for inference (after training).
        Returns probability between 0 and 1.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 96, 96)
        
        Returns:
            probability: (batch_size, 1) between 0 and 1
        """
        logits = self.forward(x, return_embedding=False)
        return torch.sigmoid(logits)


class SimpleCNN(nn.Module):
    """
    Simpler CNN for faster training (use when computational resources are limited)
    """
    
    def __init__(self, embedding_dim=64, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 3 poolings: 96 -> 48 -> 24 -> 12
        self.flattened_size = 128 * 12 * 12  # = 18432
        
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.embedding = nn.Linear(256, embedding_dim)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)
        
        self.classifier = nn.Linear(embedding_dim, 1)
        
    def forward(self, x, return_embedding=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        
        embedding = self.bn_embedding(self.embedding(x))
        
        if return_embedding:
            return embedding
        
        logits = self.classifier(embedding)
        return logits
    
    def extract_embedding(self, x):
        return self.forward(x, return_embedding=True)
    
    def predict_probability(self, x):
        logits = self.forward(x, return_embedding=False)
        return torch.sigmoid(logits)


# Model factory function
def create_cnn_model(model_type='default', embedding_dim=64, dropout_rate=0.5):
    """
    Factory function to create CNN model.
    
    Args:
        model_type: 'default' or 'simple'
        embedding_dim: Dimension of embedding vector
        dropout_rate: Dropout probability
    
    Returns:
        PyTorch model
    """
    if model_type == 'simple':
        return SimpleCNN(embedding_dim, dropout_rate)
    else:
        return BidRotationCNN(embedding_dim, dropout_rate)


# Quick test
if __name__ == "__main__":
    print("Testing CNN architecture...")
    
    # Test default model
    model = create_cnn_model('default', embedding_dim=64)
    print(f"Default model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test simple model
    model_simple = create_cnn_model('simple', embedding_dim=64)
    print(f"Simple model: {sum(p.numel() for p in model_simple.parameters()):,} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 96, 96)  # batch_size=4
    
    # Test classification output (logits, not sigmoid)
    logits = model(dummy_input)
    print(f"Logits output shape: {logits.shape}")      # Expected: (4, 1)
    print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")  # Should be unbounded
    
    # Test embedding extraction
    embeddings = model.extract_embedding(dummy_input)
    print(f"Embedding output shape: {embeddings.shape}")   # Expected: (4, 64)
    
    # Test probability prediction (after training)
    probs = model.predict_probability(dummy_input)
    print(f"Probability output shape: {probs.shape}")      # Expected: (4, 1)
    
    print("\nArchitecture test passed!")
    print("\nNote: Use nn.BCEWithLogitsLoss() for training (sigmoid is included in loss)")