import torch
from torch.utils.data import DataLoader, TensorDataset
from model import TFContrastiveModel, SemiSupervisedModel

# Generate random data for testing
def generate_test_data(num_samples=7200, sequence_length=100):
    signals = torch.randn(num_samples, 1, sequence_length)
    labels = torch.randint(0, 2, (num_samples,))
    return signals, labels

# Test data loading and preprocessing
def test_data_loading():
    signals, labels = generate_test_data()
    dataset = TensorDataset(signals, labels)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    return data_loader, signals, labels

# Evaluate model performance
def run_testing():
    data_loader, _, _ = test_data_loading()
    model = TFContrastiveModel().cuda()
    semi_supervised_model = SemiSupervisedModel(model).cuda()
    semi_supervised_model.load_state_dict(torch.load('semi_supervised_model.pth'))

    semi_supervised_model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.cuda()
            labels = labels.cuda()
            outputs = semi_supervised_model(data)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    run_testing()
