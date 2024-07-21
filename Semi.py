import torch
import torch.nn.functional as F
from model import SemiSupervisedModel, TFContrastiveModel

def train_semi_supervised_model(data_loader, pseudo_labels, confidence_scores):
    semi_supervised_model = SemiSupervisedModel(TFContrastiveModel()).cuda()
    optimizer = torch.optim.Adam(semi_supervised_model.parameters(), lr=0.001)
    num_epochs = 10

    for epoch in range(num_epochs):
        semi_supervised_model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.cuda()
            batch_size = data.size(0)

            # Ensure the batch size for pseudo_labels and confidence_scores matches the data batch size
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            if end_idx > len(pseudo_labels):
                end_idx = len(pseudo_labels)

            batch_pseudo_labels = pseudo_labels[start_idx:end_idx]
            batch_confidence_scores = confidence_scores[start_idx:end_idx]

            if batch_pseudo_labels.size(0) != batch_size:
                continue

            outputs = semi_supervised_model(data)
            pseudo_labels_tensor = batch_pseudo_labels.clone().detach().cuda()
            confidence_scores_tensor = batch_confidence_scores.clone().detach().cuda()

            # Use confidence-weighted cross-entropy loss
            loss = F.cross_entropy(outputs, pseudo_labels_tensor, reduction='none')
            loss = (loss * confidence_scores_tensor).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Semi-supervised Loss: {total_loss / len(data_loader)}")

    return semi_supervised_model

