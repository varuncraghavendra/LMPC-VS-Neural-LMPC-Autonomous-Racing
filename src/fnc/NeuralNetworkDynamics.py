"""
Improved Neural Network Dynamics Model
Better architecture, dropout, early stopping, proper validation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class VehicleDynamicsNN(nn.Module):
    """Improved Neural Network with regularization"""
    def __init__(self, state_dim=6, input_dim=2, hidden_dim=256):
        super(VehicleDynamicsNN, self).__init__()
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        # Improved architecture with dropout
        self.input_layer = nn.Linear(state_dim + input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)
        
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(0.1)
        
        self.output_layer = nn.Linear(hidden_dim, state_dim)
        
        self.residual = True
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.ln1(x)
        x = self.dropout1(x)
        
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.ln2(x)
        x = self.dropout2(x)
        
        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.ln3(x)
        x = self.dropout3(x)
        
        delta = self.output_layer(x)
        
        if self.residual:
            return state + delta
        else:
            return delta


class TrajectoryDataset(Dataset):
    """Dataset for trajectory data"""
    def __init__(self, states, actions, next_states):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        self.next_states = torch.FloatTensor(next_states)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]


class NNDynamicsPredictor:
    """Improved Neural Network predictor"""
    def __init__(self, state_dim=6, input_dim=2, hidden_dim=256, 
                 learning_rate=1e-3, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = VehicleDynamicsNN(state_dim, input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.5)
        self.criterion = nn.MSELoss()
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        self.all_states = []
        self.all_actions = []
        self.all_next_states = []
        
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        
    def add_trajectory(self, states, actions):
        """Add trajectory data"""
        for t in range(len(states) - 1):
            self.all_states.append(states[t])
            self.all_actions.append(actions[t])
            self.all_next_states.append(states[t + 1])
    
    def compute_normalization(self):
        """Compute normalization"""
        if len(self.all_states) == 0:
            return
        
        states = np.array(self.all_states)
        actions = np.array(self.all_actions)
        
        self.state_mean = states.mean(axis=0)
        self.state_std = states.std(axis=0) + 1e-6
        self.action_mean = actions.mean(axis=0)
        self.action_std = actions.std(axis=0) + 1e-6
    
    def train(self, epochs=200, batch_size=128, verbose=True):
        """Train with early stopping and validation"""
        if len(self.all_states) == 0:
            print("No training data!")
            return
        
        self.compute_normalization()
        
        states = np.array(self.all_states)
        actions = np.array(self.all_actions)
        next_states = np.array(self.all_next_states)
        
        # 90/10 train/val split
        n_train = int(0.9 * len(states))
        indices = np.random.permutation(len(states))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        train_dataset = TrajectoryDataset(states[train_idx], actions[train_idx], next_states[train_idx])
        val_dataset = TrajectoryDataset(states[val_idx], actions[val_idx], next_states[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_states, batch_actions, batch_next_states in train_loader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_next_states = batch_next_states.to(self.device)
                
                pred = self.model(batch_states, batch_actions)
                loss = self.criterion(pred, batch_next_states)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_states, batch_actions, batch_next_states in val_loader:
                    batch_states = batch_states.to(self.device)
                    batch_actions = batch_actions.to(self.device)
                    batch_next_states = batch_next_states.to(self.device)
                    
                    pred = self.model(batch_states, batch_actions)
                    loss = self.criterion(pred, batch_next_states)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            self.scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
            
            if patience_counter >= 20:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    def predict(self, state, action):
        """Predict next state"""
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            pred = self.model(state_tensor, action_tensor)
            return pred.cpu().numpy().squeeze()
    
    def save_model(self, filepath):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std,
        }, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        self.action_mean = checkpoint['action_mean']
        self.action_std = checkpoint['action_std']
        print(f"✓ Model loaded from {filepath}")
