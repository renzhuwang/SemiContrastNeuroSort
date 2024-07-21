import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import TFContrastiveModel, ContrastiveLoss, time_augment, fft_transform, ifft_transform, frequency_augment

# Generate random data for testing
def generate_test_data(num_samples=7200, sequence_length=100):
    signals = torch.randn(num_samples, 1, sequence_length)
    labels = torch.randint(0, 2, (num_samples,))
    return signals, labels

# Test data loading and preprocessing
def train_data_loading():
    signals, labels = generate_test_data()
    dataset = TensorDataset(signals, labels)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return data_loader, signals, labels

# Unsupervised pre-training with contrastive learning
def run_pretraining():
    data_loader, _, _ = train_data_loading()
    model = TFContrastiveModel().cuda()
    contrastive_loss = ContrastiveLoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, labels in data_loader:
            data = data.cuda()
            labels = labels.cuda()

            # Time and frequency domain augmentations
            data_time_aug = time_augment(data)
            data_real, data_imag = fft_transform(data)
            data_real_aug, data_imag_aug = frequency_augment(data_real, data_imag)
            data_freq_aug = ifft_transform(data_real_aug, data_imag_aug)

            # Forward pass
            z_time, z_freq = model(data, data_freq_aug)
            z_time_aug, _ = model(data_time_aug, data_freq_aug)

            # Compute contrastive loss
            batch_size = z_time.size(0)
            labels = labels[:batch_size]  # Ensure labels match the batch size
            loss_time = contrastive_loss(z_time, labels)
            loss_freq = contrastive_loss(z_freq, labels)
            consistency_loss = F.mse_loss(z_time, z_freq) + F.mse_loss(z_time_aug, z_freq)
            loss = loss_time + loss_freq + consistency_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Contrastive Loss: {total_loss / len(data_loader)}")

    torch.save(model.state_dict(), 'contrastive_model.pth')

if __name__ == "__main__":
    run_pretraining()
