# Denoising-Image

---

### **Objectives**
1. **Purpose**: Build and train an autoencoder model to denoise images.
2. **Steps**:
   - Load and preprocess image data.
   - Add artificial noise to the images.
   - Train an autoencoder to reconstruct the original images from noisy inputs.
   - Evaluate and visualize the model's performance.

---

### **Skills Used**
1. **Data Preprocessing**:
   - Loading image data from directories.
   - Resizing and normalizing images.
   - Adding Gaussian noise to simulate noisy data.

2. **Deep Learning**:
   - Building and compiling an autoencoder using Keras.
   - Training the model on noisy input images.

3. **Visualization**:
   - Displaying images using `matplotlib`.
   - Plotting model loss and accuracy trends.

4. **Evaluation**:
   - Generating predictions from the model.
   - Visualizing noisy vs. reconstructed images.

---

### **Libraries Imported**
1. **os**: Navigate through directories to load image files.
2. **cv2 (OpenCV)**: Load, preprocess, and manipulate image data.
3. **numpy**: Handle numerical operations and array manipulations.
4. **random**: Select random indices for visualization.
5. **matplotlib.pyplot**: Visualize images, training loss, and accuracy.
6. **tensorflow.keras**:
   - `Model`, `Input`, and various layers for building the autoencoder.
   - `Adam` optimizer for training the model.

---

### **Explanation of the Code**

#### **1. Data Loading and Preprocessing**
- The `load_data` function:
  - Traverses the specified folder (`train_data_path` or `test_data_path`).
  - Reads images using `cv2.imread`.
  - Converts them to RGB format using `cv2.cvtColor`.
  - Resizes images to a uniform shape of \(128 \times 128\).
  - Normalizes pixel values to the range [0, 1].
- The data is returned as a NumPy array for further processing.

#### **2. Adding Noise**
- The `add_noise` function:
  - Adds Gaussian noise to the image dataset.
  - Clips pixel values to ensure they remain within the [0, 1] range.

#### **3. Visualizing Data**
- Random images from the original dataset and their noisy counterparts are displayed using `matplotlib`.

#### **4. Autoencoder Model Architecture**
- **Encoder**:
  - Input layer accepts images of shape \(128 \times 128 \times 3\).
  - Three convolutional layers (`Conv2D`) with ReLU activation and max pooling (`MaxPooling2D`) progressively downsample the image to extract key features.
- **Decoder**:
  - Three convolutional layers with upsampling (`UpSampling2D`) reconstruct the image to its original shape.
  - A final convolutional layer with sigmoid activation ensures the output image has pixel values in the range [0, 1].

#### **5. Compiling and Training**
- The model uses the **Adam optimizer** with a learning rate of 0.001.
- The loss function is **mean squared error (MSE)**, suitable for pixel-wise image reconstruction.
- Training is done for 20 epochs with a batch size of 16. The noisy images are the input, and the original clean images are the target.

#### **6. Visualizing Training Performance**
- Loss and accuracy trends for training and validation datasets are plotted.

#### **7. Model Evaluation**
- Predictions are generated for noisy test images.
- A randomly selected noisy image and its corresponding reconstructed (denoised) image are visualized.

---
Code:https://github.com/JIMMYULAHANNAN/Denoising-Image/blob/main/CNN%20Model%20Image%20denoising%20.ipynb


This workflow provides a foundation for image denoising using convolutional neural networks (CNNs) and autoencoders. The code emphasizes deep learning techniques for feature extraction, image reconstruction, and noise reduction.
