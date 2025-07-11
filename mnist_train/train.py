import numpy as np
import cv2
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import cpu_count
import sys


def write_weights_to_h_file(model, filename="mnist_weights.h"):
    with open(filename, "w") as f:
        f.write("#ifndef MNIST_WEIGHTS_H\n")
        f.write("#define MNIST_WEIGHTS_H\n\n")
        f.write("// Neural Network Weights for MNIST Classifier\n")
        f.write("// Architecture: 100-32-16-10 (no biases)\n\n")
        
        # Layer dimensions
        f.write("// Layer Dimensions\n")
        f.write("#define INPUT_DIM 100\n")
        f.write("#define L1_DIM 32\n")
        f.write("#define L2_DIM 16\n")
        f.write("#define OUTPUT_DIM 10\n\n")
        
        # Combined weights array (flattened)
        f.write("// Combined weights array (flattened)\n")
        f.write("const float nn_weights[] = {\n")
        
        # Layer 1 weights (100x32)
        weights = model.layer1.weight.T.cpu().detach().numpy().flatten()
        f.write("    // Layer 1 (100x32)\n    ")
        f.write(",\n    ".join([f"{x:.8f}f" for x in weights]))
        f.write(",\n\n")
        
        # Layer 2 weights (32x16)
        weights = model.layer2.weight.T.cpu().detach().numpy().flatten()
        f.write("    // Layer 2 (32x16)\n    ")
        f.write(",\n    ".join([f"{x:.8f}f" for x in weights]))
        f.write(",\n\n")
        
        # Layer 3 weights (16x10)
        weights = model.layer3.weight.T.cpu().detach().numpy().flatten()
        f.write("    // Layer 3 (16x10)\n    ")
        f.write(",\n    ".join([f"{x:.8f}f" for x in weights]))
        f.write("\n};\n\n")
        
        # Weight offsets for each layer
        f.write("// Weight offsets for each layer\n")
        f.write("#define L1_WEIGHT_OFFSET 0\n")
        f.write("#define L2_WEIGHT_OFFSET (INPUT_DIM * L1_DIM)\n")
        f.write("#define L3_WEIGHT_OFFSET (L2_WEIGHT_OFFSET + L1_DIM * L2_DIM)\n")
        f.write("#define TOTAL_WEIGHTS (L3_WEIGHT_OFFSET + L2_DIM * OUTPUT_DIM)\n\n")
        
        f.write("#endif // MNIST_WEIGHTS_H\n")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(10*10, 32, bias=False)
        self.layer2 = nn.Linear(32, 16, bias=False)
        self.layer3 = nn.Linear(16, 10, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)  # No softmax - included in CrossEntropyLoss

def load_images_parallel(folder_path, dims=(10,10)):
    """Optimized parallel image loading"""
    from concurrent.futures import ThreadPoolExecutor
    
    def load_single_image(path):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)
        return img / 255.0
    
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_single_image, image_paths))
    return images

def main():
    # GPU optimization settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    torch.set_float32_matmul_precision('high')  # For newer GPUs
    
    print(f"Using device: {device}")
    print("Loading dataset")
    
    # Load data with parallel processing
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    dims = (10,10)
    for i in range(10):  # 0-9 digits
        # Training data
        train_folder = f'MNIST_Dataset_JPG/MNIST_JPG_training/{i}/'
        train_images.extend(load_images_parallel(train_folder, dims))
        train_labels.extend([i]*len(os.listdir(train_folder)))
        
        # Test data
        test_folder = f'MNIST_Dataset_JPG/MNIST_JPG_testing/{i}/'
        test_images.extend(load_images_parallel(test_folder, dims))
        test_labels.extend([i]*len(os.listdir(test_folder)))

    # Convert to tensors
    train_images = torch.tensor(np.array(train_images), dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_images = torch.tensor(np.array(test_images), dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create optimized DataLoaders
    num_workers = min(4, cpu_count())  # Use 4 workers or available cores
    batch_size = 4000  # Increased batch size for GPU efficiency
    
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    # Model and training setup
    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    epochs = 50
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Faster zero_grad
            
            with torch.cuda.amp.autocast():  # Mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}')
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    inference_start = time.time()
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    total_time = time.time() - start_time
    inference_time = time.time() - inference_start
    
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Inference time for {total} images: {inference_time:.4f} seconds")

    # Save weights (unchanged from your original function)
    write_weights_to_h_file(model)
    print("Weights saved to mnist_weights.h")

if __name__ == "__main__":
    main()