import torch
import torch.nn as nn
from tqdm import tqdm  # Progress bar
import time
import copy


def get_loss_function():
    
    criterion = nn.CrossEntropyLoss() #suitable for multi-class classification tasks, works on probabilities (after softmax) or raw logits
    return criterion


def train_one_epoch(model, train_loader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Training", ncols=100)


    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # reset gradients to zero

        outputs = model(images)
        loss = criterion(outputs, labels) # compute loss (cross-entropy)
        loss.backward() # backpropagation
        optimizer.step() # update weights

        _, predicted = torch.max(outputs, 1) # select most probable class

        correct = (predicted == labels).sum().item()

        total_samples += labels.size(0)
        correct_predictions += correct
        running_loss += loss.item() * images.size(0)

        current_acc = correct_predictions / total_samples
        pbar.set_postfix({
            'Loss': f"{running_loss / total_samples:.4f}", 
            'Acc': f"{current_acc:.4f}"
        })

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc


def validate_one_epoch(model, val_loader, criterion, device):
    
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    pbar = tqdm(val_loader, desc="Validation", ncols=100)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            correct = (predicted == labels).sum().item()

            total_samples += labels.size(0)
            correct_predictions += correct
            running_loss += loss.item() * images.size(0)

            current_acc = correct_predictions / total_samples
            pbar.set_postfix({
                'Loss': f"{running_loss / total_samples:.4f}", 
                'Acc': f"{current_acc:.4f}"
            })

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, optimizer, 
                scheduler, device, num_epochs=25, save_path='best_model.pth', class_weights=None):

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'\nResults:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}')
        print(f'  Learning Rate: {current_lr:.6f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, save_path)
            print(f'  ✓ New best model saved! (Val Acc: {best_acc:.4f})')
        else:
            print(f'  Best Val Acc so far: {best_acc:.4f}')

    elapsed_time = time.time() - start_time
    print(f'\nTraining complete in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    
    return model, history

