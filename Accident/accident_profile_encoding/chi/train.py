import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import device
import numpy as np
import csv
def train_model(num_epochs, model, train_loader, test_loader, criterion, optimizer, device):
    best_accuracy = 0.0
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            X_BERT, X_weekday, X_time, X_location, X_traffic, labels = data
            inputs = (
            X_BERT.to(device), X_weekday.to(device), X_time.to(device), X_location.to(device), X_traffic.to(device))
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(*inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            # 使用梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

        # 使用test_loader评估模型
        accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device,'chi_pre_vs_true.csv')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()

    print(f'Best validation accuracy: {best_accuracy:.2f}%')
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), 'chi_model_weights.pth')  # 使用正确的文件路径

    return best_accuracy  # 可以选择返回最佳准确率


def evaluate_model(model, test_loader, device, output_filename):
    model.eval()
    total_samples = 0
    correct_preds = 0
    all_predicted = []
    all_labels = []
    all_embeddings = []  # 用于存储嵌入向量的列表
    with torch.no_grad():
        for data in test_loader:
            X_BERT, X_weekday, X_time, X_location, X_traffic, labels = data
            inputs = (X_BERT.to(device), X_weekday.to(device), X_time.to(device), X_location.to(device), X_traffic.to(device))
            labels = labels.to(device)

            # 获得模型的输出，这里假设模型返回了预测结果和嵌入表示
            outputs, embeddings = model(*inputs)
            _, predicted = torch.max(outputs, 1)

            total_samples += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            all_embeddings.extend(embeddings.cpu().numpy())
    accuracy = correct_preds / total_samples
    print(f'Test Accuracy: {accuracy * 100:.5f}%')

    all_predicted = np.array(all_predicted)
    all_labels = np.array(all_labels)

    precision = precision_score(all_labels, all_predicted, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predicted, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predicted, average='macro', zero_division=0)

    print(f'Precision: {precision:.5f}')
    print(f'Recall: {recall:.5f}')
    print(f'F1 Score: {f1:.5f}')

    # 将嵌入向量保存到文件
    embeddings_path = '../data/Chicago/Chicago_accident.npy'  # 替换为您想要的文件路径
    np.save(embeddings_path, np.array(all_embeddings))
    # 将预测结果和真实标签写入CSV文件
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['TrueLabel', 'PredictedLabel'])  # 写入标题行
        for true_label, predicted_label in zip(all_labels, all_predicted):
            writer.writerow([true_label, predicted_label])  # 写入数据行

    return accuracy, precision, recall, f1