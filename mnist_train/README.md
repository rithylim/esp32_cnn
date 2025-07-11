# Training Procedure

After download the dataset, make sure that you follow the directory format in other to run the [train.py](train.py).

```
mnist_train
|- MNIST_Dataset_JPG
|- train.py
```

If you follow the above directory format, you don't need to change the below code. Otherwise, you have to check on this part:

```python
for i in range(10):  # 0-9 digits
        # Training data
        train_folder = f'MNIST_Dataset_JPG/MNIST_JPG_training/{i}/'
        train_images.extend(load_images_parallel(train_folder, dims))
        train_labels.extend([i]*len(os.listdir(train_folder)))
        
        # Test data
        test_folder = f'MNIST_Dataset_JPG/MNIST_JPG_testing/{i}/'
        test_images.extend(load_images_parallel(test_folder, dims))
        test_labels.extend([i]*len(os.listdir(test_folder)))
```

In my case, I've use CUDA to speed up my training and I'm using RTX 4060 to train this model. If you want to switch to CPU mode to train, you can modify here:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

To run the file just follow:

```
python .\train.py 
```

My training result:

```
PS C:\Users\rithy\learn_ai\esp32_mnist_dl\mnist_train> python .\train.py
Using device: cuda
Loading dataset
Starting training...
Epoch 1/50, Loss: 2.2935
Epoch 2/50, Loss: 2.2622
...
Epoch 49/50, Loss: 0.2906
Epoch 50/50, Loss: 0.2882

Training completed in 111.81 seconds
Test accuracy: 92.17%
Inference time for 10000 images: 2.1555 seconds
Weights saved to mnist_weights.h
```

After sucessfully of training, we will get [this file](mnist_weights.h) in other to implement with with c programming file.