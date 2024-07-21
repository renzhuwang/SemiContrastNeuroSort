import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

# Define TimeEncoder
class TimeEncoder(nn.Module):
    def __init__(self):
        super(TimeEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 50, 128)  # Assuming input sequence length is 100

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 50)
        x = F.relu(self.fc1(x))
        return x

# Define FrequencyEncoder
class FrequencyEncoder(nn.Module):
    def __init__(self):
        super(FrequencyEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 50, 128)  # Adjust based on frequency domain representation length

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 50)
        x = F.relu(self.fc1(x))
        return x

# Define Projector
class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x

# Define TFContrastiveModel
class TFContrastiveModel(nn.Module):
    def __init__(self):
        super(TFContrastiveModel, self).__init__()
        self.time_encoder = TimeEncoder()
        self.frequency_encoder = FrequencyEncoder()
        self.time_projector = Projector(input_dim=128, output_dim=64)
        self.frequency_projector = Projector(input_dim=128, output_dim=64)

    def forward(self, x_time, x_freq):
        h_time = self.time_encoder(x_time)
        h_freq = self.frequency_encoder(x_freq)

        z_time = self.time_projector(h_time)
        z_freq = self.frequency_projector(h_freq)

        return z_time, z_freq

# Define ContrastiveLoss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        mask = mask - torch.eye(mask.size(0)).to(device)
        positive_samples = mask * similarity_matrix

        neg_mask = 1 - mask
        negative_samples = neg_mask * similarity_matrix

        positive_loss = torch.sum(-F.log_softmax(positive_samples, dim=-1), dim=-1)
        negative_loss = torch.sum(-F.log_softmax(negative_samples, dim=-1), dim=-1)

        loss = positive_loss.mean() + negative_loss.mean()
        return loss

# FFT and inverse FFT transform functions
def fft_transform(x):
    x_fft = torch.fft.fft(x, dim=-1)
    return x_fft.real, x_fft.imag

def ifft_transform(x_real, x_imag):
    x_complex = torch.complex(x_real, x_imag)
    x_ifft = torch.fft.ifft(x_complex, dim=-1)
    return x_ifft.real

# Data augmentation functions
def time_augment(x):
    x_aug = x + 0.05 * torch.randn_like(x)
    return x

def frequency_augment(x_real, x_imag):
    E = 5
    indices = np.random.choice(x_real.shape[-1], E, replace=False)
    x_real[:, :, indices] = 0
    x_imag[:, :, indices] = 0
    return x_real, x_imag

# Define PseudoLabelGenerator
class PseudoLabelGenerator:
    def __init__(self, encoder, num_clusters=10, gamma=0.5):
        self.encoder = encoder
        self.num_clusters = num_clusters
        self.gamma = gamma

    def compute_features(self, data_loader):
        features = []
        self.encoder.eval()
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.cuda()
                feature_time, feature_freq = self.encoder(data, data)
                feature = (feature_time + feature_freq) / 2  # Combine time and frequency features
                features.append(feature.cpu())
        features = torch.cat(features, dim=0)
        return features

    def generate_pseudo_labels(self, features):
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(features.numpy())
        pseudo_labels = torch.tensor(kmeans.labels_)

        # Compute confidence scores
        num_samples = features.size(0)
        num_top = int(self.gamma * num_samples / self.num_clusters)
        confidence_scores = torch.zeros(num_samples)

        for c in range(self.num_clusters):
            cluster_features = features[pseudo_labels == c]
            distances = torch.norm(cluster_features - torch.tensor(kmeans.cluster_centers_[c]), dim=1)
            sorted_distances, _ = distances.sort()
            threshold_distance = sorted_distances[min(num_top, len(sorted_distances) - 1)]
            cluster_indices = torch.nonzero(pseudo_labels == c, as_tuple=False).squeeze()
            for idx in cluster_indices:
                if idx < distances.size(0):
                    confidence_scores[idx] = 1 - (distances[idx] / threshold_distance)

        return pseudo_labels, confidence_scores

# Define SemiSupervisedModel
class SemiSupervisedModel(nn.Module):
    def __init__(self, encoder, num_clusters=10):
        super(SemiSupervisedModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(64, num_clusters)  # Assuming output dimension of projector is 64

    def forward(self, x):
        features_time, features_freq = self.encoder(x, x)
        features = (features_time + features_freq) / 2
        outputs = self.classifier(features)
        return outputs
