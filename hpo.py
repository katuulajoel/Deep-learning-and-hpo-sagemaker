import argparse
import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, device):
    '''
    This function takes a PyTorch model and a DataLoader for the test set, computes the test loss and accuracy, and prints the results.
    
    Args:
    model: PyTorch model to be evaluated.
    test_loader: DataLoader for the test set.
    device: The device to train on ('cpu' or 'cuda').
    '''
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=1):
    '''
    This function trains a PyTorch model on the provided training data and validates it
    on the validation data.

    Args:
    model: PyTorch model to be trained.
    train_loader: DataLoader for the training data.
    val_loader: DataLoader for the validation data.
    criterion: Loss function to be used for training.
    optimizer: Optimizer to be used for training.
    device: The device to train on ('cpu' or 'cuda').
    num_epochs: Number of epochs to train the model (default is 1).
    '''
    
    # Set the model to training mode
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        # Training pass
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                logger.info(f'[Epoch {epoch + 1}, Batch {batch_idx}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        # Validation pass
        val_loss = 0.0
        val_steps = 0
        correct = 0
        total = 0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disabling gradient calculation
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, target).item()
                
                # calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                val_steps += 1
        
        # Calculate average losses
        average_train_loss = running_loss / len(train_loader)
        average_val_loss = val_loss / val_steps

        # Print validation results
        logger.info(f'Epoch {epoch + 1}: Avg. Train Loss: {average_train_loss:.3f}, '
                    f'Avg. Val Loss: {average_val_loss:.3f}, '
                    f'Accuracy: {100 * correct / total:.2f}%')

    logger.info('Finished Training')
    return model
    
def net(num_classes):
    '''
    Initializes a pre-trained ResNet model and modifies the final fully connected layer.

    Args:
    num_classes: Number of classes in the dataset.

    Returns:
    model: Initialized PyTorch model.
    '''
    
    # Load a pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    
    # freeze all the convolutional layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Get the input dimension of the last layer
    in_features = model.fc.in_features
    
    # Replace the last fully connected layer with a new one with the correct number of classes
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def create_data_loaders(train_dir, valid_dir, test_dir, batch_size):
    '''
    Creates data loaders for training, validation, and test datasets.

    Args:
    train_dir: Directory of the training dataset.
    valid_dir: Directory of the validation dataset.
    test_dir: Directory of the test dataset.
    batch_size: Batch size for the data loaders.

    Returns:
    A dictionary containing the data loaders for the 'train', 'valid', and 'test' datasets.
    '''

    # Define image transformations
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # Load datasets using the specified directories for train, valid, and test
    image_datasets = {
        'train': torchvision.datasets.ImageFolder(root=train_dir, transform=transform['train']),
        'valid': torchvision.datasets.ImageFolder(root=valid_dir, transform=transform['valid']),
        'test': torchvision.datasets.ImageFolder(root=test_dir, transform=transform['test'])
    }
    
    # Create data loaders
    data_loaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'valid': DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=True, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    return data_loaders


def main(args):
    train_data_dir = '/opt/ml/input/data/train'  # SageMaker's local path for training data
    valid_data_dir = '/opt/ml/input/data/valid'  # SageMaker's local path for validation data
    test_data_dir = '/opt/ml/input/data/test'   # SageMaker's local path for test data
    
    # Create data loaders for training, validation, and testing datasets
    data_loaders = create_data_loaders(train_data_dir, valid_data_dir, test_data_dir, args.batch_size)
    
    model = net(num_classes=args.num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    model = model.to(device)
    
    model = train(model, data_loaders['train'], data_loaders['valid'], loss_criterion, optimizer, device, num_epochs=args.epochs)
    
    test(model, data_loaders['test'], device)
    
    torch.save(model.state_dict(), args.save_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train and Evaluate a Dog Breed Classifier')
    
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--save-path', type=str, default='model.pth', help='Path to save the trained model (default: model.pth)')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of classes in the dataset')
    parser.add_argument('--s3-bucket', type=str, required=True, help='S3 bucket to upload the trained model')
    parser.add_argument('--s3-model-path', type=str, required=True, help='S3 path to save the trained model')
    
    args=parser.parse_args()
    
    main(args)