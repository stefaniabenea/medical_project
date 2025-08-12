from model import CNN
from dataset import prepare_data
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import plot_confusion_matrix, set_seed, get_model, classification_report_per_class
import os
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)
set_seed(42)



parser = argparse.ArgumentParser(description="Train script")
parser.add_argument("--model_name", required=True, type=str, choices= ["CNN", "resnet18"], help="Choose between 'CNN' (custom model) and pretrained 'resnet18' model")
args = parser.parse_args()
model_name = args.model_name
train_loader, test_loader, class_names = prepare_data(data_dir = "data", model_name = model_name, batch_size=64)
model = get_model(model_name,len(class_names),fine_tune=False, unfrozen_blocks=0)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

all_train_losses =[]
all_train_accuracies = []
all_test_losses =[]
all_test_accuracies = []

best_loss = float('inf')
patience = 5
counter = 0
best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

nr_epochs = 30
for epoch in range(nr_epochs):
    
    model.train()
    running_loss = 0.0
    correct = 0
    total =0
    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        
        loss = loss_fn(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        correct +=(predictions==labels).sum().item()
        total+=images.size(0)

    avg_loss = running_loss/len(train_loader)
    accuracy = 100*correct/total
    all_train_losses.append(avg_loss)
    all_train_accuracies.append(accuracy)

    
    all_preds_test = []
    all_labels_test = []
    misclassified_images = []
    correct_test = 0
    total_test = 0
    running_loss_test = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            running_loss_test+=loss.item()
            correct_test += (predictions==labels).sum().item()
            total_test += images.size(0)

            all_labels_test.extend(labels.cpu().numpy())
            all_preds_test.extend(predictions.cpu().numpy())
            wrong_mask = predictions!=labels
            if wrong_mask.any():
                misclassified_images.append((images[wrong_mask].cpu(),labels[wrong_mask].cpu(), predictions[wrong_mask].cpu()))


        accuracy_test = 100* correct_test/total_test
        avg_loss_test = running_loss_test/len(test_loader)
        all_test_losses.append(avg_loss_test)
        all_test_accuracies.append(accuracy_test)
        scheduler.step(avg_loss_test)

        #early stopping
        if avg_loss_test<best_loss:
            best_loss=avg_loss_test
            counter = 0
            torch.save(model.state_dict(),f"models/{model_name}.pth")
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            counter+=1
            if counter>=patience:
                print("Early stopping triggered")
                break
        
        print(f"Epoch {epoch+1}/{nr_epochs} \n Train loss: {avg_loss:.4f} | Train accuracy: {accuracy:.2f} | Test loss: {avg_loss_test:.4f} | Test accuracy: {accuracy_test:.2f}")


model.load_state_dict(best_model_state)

fig, axs = plt.subplots(1, 2, figsize=(12,5))
axs[0].plot(all_train_losses, label="Train")
axs[0].plot(all_test_losses, label="Test")
axs[0].legend()
axs[0].set_title("Train vs test loss")
axs[0].set_ylabel("Loss")
axs[0].set_xlabel("Epoch")

axs[1].plot(all_train_accuracies,label="Train")
axs[1].plot(all_test_accuracies,label="Test")
axs[1].legend()
axs[1].set_title("Train vs test accuracy")
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("Epoch")

plt.tight_layout()
plt.savefig(f"plots/{model_name}_loss_accuracy_plot.png")
plt.show()

plot_confusion_matrix(all_labels_test, all_preds_test, class_names, save_path="plots", model_name=model_name)

all_wrong_images = [x[0] for x in misclassified_images]
all_wrong_labels = [x[1] for x in misclassified_images]
all_wrong_preds = [x[2] for x in misclassified_images]

all_wrong_images = torch.cat(all_wrong_images, dim=0)
all_wrong_labels = torch.cat(all_wrong_labels, dim=0)
all_wrong_preds  = torch.cat(all_wrong_preds, dim=0)

N = 10
selected_images = all_wrong_images[:N]
selected_labels = all_wrong_labels[:N]
selected_preds = all_wrong_preds[:N]

plt.figure(figsize=(12,6))
for i in range(N):
    plt.subplot(2,5,i+1)
    plt.imshow(selected_images[i].permute(1,2,0))
    plt.axis('off')
    plt.title(f"Real: {class_names[selected_labels[i]]}\nPred: {class_names[selected_preds[i]]}", fontsize=8)
plt.tight_layout()
plt.show()

classification_report_per_class(all_labels_test, all_preds_test, class_names)


