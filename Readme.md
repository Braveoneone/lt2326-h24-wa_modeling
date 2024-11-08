# Bonus A
#### Improved Convolutional neural Network structure (file: wikiart.py) :   
The current model structure is too simple, consisting of only one convolutional layer and one fully connected layer. Convolution layers, Batch Normalization, and Dropout can be added to improve the feature extraction capability of the model and avoid overfitting.  
#### Optimize hyperparameters (files: train.py and config.json) :   
Reduce the learning rate a bit and try to increase the number of training cycles. The current learning rate is high, which may affect the stable convergence of the model.
In the WikiArtModel class, the original combination of convolutional layers and fully connected layers is replaced with a multi-layer convolutional neural network, which adds multiple convolutional layers and pooled layers to capture more complex image features. The addition of batch normalization and Dropout layers helps mitigate overfitting problems.
In train.py, adjust the learning rate to 0.001 for more stable training. The training time can also be increased depending on the results.

```
Learning rate = 0.001
Dropout = 0.01
epochs = 50 
batch_size = 32
```

Finally, accuracy is improved to 0.07301587611436844.

# Part 1
I solve the imbalance by calculating class weights. The get_class_weights function in the code calculates class weights based on the number of samples for each class in the data set. Weights are calculated by the total number of samples/(number of categories * Number of samples per category) to ensure that categories with fewer samples have higher weights, thus balancing the impact of class imbalances.  
Then I make use of weighted losses. Although the get_class_weights function calculates class weights, it is not used directly to adjust the loss function in the current code. In theory, the calculated weights can be passed to loss functions such as CrossEntropyLoss, which can be used to weight the losses of different classes and improve the learning effect of the model on a few classes.

# Part 2
#### Implement  
```
1. The autoencoder was used to learn image features from the WikiArt dataset.
2. The autoencoder contains an encoder and a decoder, where the encoder compresses the image into a low-dimensional representation.
3. The mean square error (MSE) is used as a loss function and the model is optimized by the Adam optimizer.
4. After training, the underlying features of the image are extracted and reduced to two dimensions by PCA.
5. The KMeans clustering algorithm is used to cluster the features after dimensionality reduction.
6. The clustering result is compared with the real label, and the clustering effect is visualized.
7. The results show that KMeans can classify images according to artistic style well.
8. The PCA distribution shows that images of different artistic styles are separated in the feature space.
9. The autoencoder successfully learns effective features and supports clustering tasks.
10. Through this method, the effective classification and analysis of artistic images can be realized.
```

# Part 3
#### Implement  
```
The AutoencoderWithStyle class defines an autoencoder model that includes an encoder (which encodes images into features), a style embedding layer (which converts style information into features), and a decoder (which decodes features back into images).

Encoder: Consists of multiple convolutional layers that gradually compress the input image from 3 channels (RGB) to a 1024 dimensional feature vector.
Style embedding layer: The style vector (128 dimensional) is transformed into 1024 dimensional features through the full connection layer, and the image features are combined.

Decoder: The combined feature is restored to an image through the deconvolution layer, and the RGB image of the same size as the input image is output.
Forward propagation: In the process of forward propagation, the input image and style embedding are combined after processing by the encoder and style embedding layer, and finally the image is reconstructed by the decoder.

save_images function: Defines a function to save images before and after processing. The image is saved in PNG format and named.

Data set loading: Use the ImageFolder to load the data set from the specified directory and bulk load through the DataLoader.

Image conversion: Preprocessing of images, including resizing (64x64) and converting to Tensor format.

Test phase: In a batch of data, the model is propagated forward and the image after style fusion is output.

Save the image: The processed image and the original image are saved to the specified path for visual comparison.

