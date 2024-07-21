import torch
from torch.utils.data import DataLoader, TensorDataset
from model import TFContrastiveModel, PseudoLabelGenerator
from Semi import train_semi_supervised_model

# Generate random data for testing
def generate_test_data(num_samples=7200, sequence_length=100):
    signals = torch.randn(num_samples, 1, sequence_length)
    labels = torch.randint(0, 2, (num_samples,))
    return signals, labels

# Test data loading and preprocessing
def test_data_loading():
    signals, labels = generate_test_data()
    dataset = TensorDataset(signals, labels)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return data_loader, signals, labels

# Finetuning with semi-supervised learning
def run_finetuning():
    data_loader, _, _ = test_data_loading()
    model = TFContrastiveModel().cuda()
    model.load_state_dict(torch.load('contrastive_model.pth'))

    pseudo_label_generator = PseudoLabelGenerator(model, num_clusters=10, gamma=0.5)
    features = pseudo_label_generator.compute_features(data_loader)
    pseudo_labels, confidence_scores = pseudo_label_generator.generate_pseudo_labels(features)

    semi_supervised_model = train_semi_supervised_model(data_loader, pseudo_labels, confidence_scores)
    torch.save(semi_supervised_model.state_dict(), 'semi_supervised_model.pth')

if __name__ == "__main__":
    run_finetuning()
