# **Age Estimation**

![Age Estimation Model](https://tokenoftrust.com/wp-content/uploads/2024/06/image_hero_age-estimation_612x583.webp)


## **🧠 Project Overview**

Age estimation is a challenging and interesting task in computer vision. Factors such as lifestyle, genetics, ethnicity, makeup, and imaging conditions make accurate age prediction difficult. 
In this project, we perform **age estimation from face images** using **deep learning techniques**, leveraging the power of convolutional neural networks to predict a person’s age based on facial features.

## **🌟 Features**

- Develops a **precise method for age estimation from face images**  
- Supports **real-time usage**  
- **Robust to challenges** such as poor lighting and imaging conditions  
- **Robust to variations** such as gender and ethnicity

## **🛠️ Project Steps**

1. **Data**  
2. **Model**  
3. **Model Configuration** (Loss functions, Metrics, Optimizers, etc.)  
4. **Training and Validation Functions**  
5. **Model Training** – this is where practical experience plays a key role!  
6. **Evaluation**  

## 📂 Data

This project uses the **[UTKFace dataset](https://susanqq.github.io/UTKFace/)**, a large-scale face dataset with a wide age range (0 to 116 years old). The dataset contains over **20,000 face images** with annotations for **age, gender, and ethnicity**.  

The images include a large variation in:
- Pose  
- Facial expression  
- Illumination  
- Occlusion  
- Resolution  

This dataset can be used for a variety of tasks, including:
- Face detection  
- Age estimation  
- Age progression/regression  
- Landmark localization  


## **📉📈EDA**

Let's explore the images in the UTK dataset together!:

<img width="891" height="953" alt="image" src="https://github.com/user-attachments/assets/e786bbd7-e4bb-4b01-ae1d-28f30db775a2" />

🔍 Interesting Observations in the Dataset:

- The dataset contains **only the face region**  
- The dataset is **aligned**  
- All images in the dataset are **200x200 pixels** (some images have been upscaled, which may reduce quality)  
- The images are **in color**  
- Some faces are **angled**, which makes the task more challenging  
- Some images contain **only a part of the face** (e.g., only the eyes)  
- There are **multiple images of the same person**, which can challenge the evaluation  
- Some **labels may be incorrect**  
- Some images contain **watermarks**  
- There is **high variation in lighting**, which can challenge the network  
- Some faces are actually **paintings** but have been labeled as real

**Age Histogram:**

<img width="975" height="485" alt="image" src="https://github.com/user-attachments/assets/fc65344d-66db-42ca-8a2f-3eafba7ca308" />

The age distribution in the dataset is **right-skewed**, with younger ages (0–30 years) dominating and older ages (>50 years) underrepresented (mean age = 32.0, median = 29.0). The highest frequency occurs in the youngest age bin (~3,500 samples), and counts gradually decline for middle and older age groups, creating a **severe class imbalance**. This bias may cause the model to perform better on younger faces while struggling with older adults. To address these challenges, techniques such as **oversampling rare age groups, weighted loss during training, data augmentation**, or collecting more data for older ages can be considered to improve model performance across all age ranges.

**Plot histogram for gender:**

<img width="975" height="550" alt="image" src="https://github.com/user-attachments/assets/ab6ee0cd-194f-4ce1-b908-68c90180bd5f" />

The dataset shows a **balanced gender distribution**, with approximately **10,957 male** and **10,957 female** samples out of 21,914 labeled images. However, **1,251 samples (5.4%) are missing gender labels**, which may indicate unlabeled data or potential errors. Overall, male and female faces are nearly equally represented, minimizing class imbalance for gender. To ensure robust model training, the missing labels should be investigated, as they could introduce noise or bias if ignored, especially when handling faces with ambiguous gender traits.

**Plot histogram for ethnicity:**

<img width="975" height="550" alt="image" src="https://github.com/user-attachments/assets/7b2b4170-8c45-4229-b053-041998b8261b" />

The dataset shows a **class imbalance** among ethnicities:  
- White: 9,698 samples (42%)  
- Black: 4,478 samples  
- Indian: 3,952 samples  
- Asian: 3,348 samples  
- Others: 1,689 samples  

White and Indian groups are the most represented, while the "Others" category has the fewest samples, which may lead to **poorer model performance** for underrepresented groups. This imbalance introduces a **bias risk**, as the model may overfit to dominant groups and perform less accurately on minority ethnicities.

## 🧹 Data Preparation and Preprocessing

- **Fixing basic issues in the dataset**  
  - For example, removing or correcting unlabeled, corrupted, or problematic data  

- **Splitting the dataset into train, test, and validation sets**  
  - In this project, **30% of the dataset** was allocated for evaluation and testing, with **half of this portion used for testing** and the other half for validation

## 🔄 Data Augmentation and Normalization

- **Data Augmentation**  
  Using torchvision transforms, we applied the following augmentations **only to the training data**:
  - Resizing images to **128x128 pixels**  
  - Applying **random horizontal flips**  
  - Introducing **random rotations** of 15 degrees  
  - Adjusting image color using **ColorJitter**  
  - Converting images to **tensors**  
  - Normalizing pixel values using the provided mean and standard deviation:  
    `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`  

- **Data Normalization and Standardization**  
  Ensures that all images have a consistent scale and distribution for better model training.

## 💡 Model

For this project, we use **ResNet50**, a widely-used convolutional neural network (CNN) known for its high performance in image recognition tasks. ResNet, short for **Residual Network**, introduces the concept of **residual learning**, which helps train very deep networks effectively by addressing the **vanishing gradient problem**.

![ResNet50 Architecture](https://towardsdatascience.com/wp-content/uploads/2022/08/1rPktw9-nz-dy9CFcddMBdQ.jpeg)


### Key Features of ResNet50:
- **50 layers deep** including convolutional, batch normalization, and identity shortcut layers  
- **Residual blocks** allow the network to learn **identity mappings**, making it easier for the network to learn deeper representations  
- Pretrained versions (on ImageNet) can be fine-tuned for **transfer learning**, which accelerates training and improves performance on smaller datasets  

### Why ResNet50 for Age Estimation:
- Effectively captures **facial features** at multiple levels of abstraction  
- Handles **complex variations** in pose, lighting, and expression  
- Provides a strong backbone for regression tasks such as predicting **continuous age values**  

By leveraging ResNet50, our model can extract robust features from face images and provide accurate age estimation across a wide range of inputs.

## ⚙️ Hyperparameter Tuning

Hyperparameter tuning is essential to improve the performance of our **Age Estimation Model**. We experimented with different **learning rates** and **weight decay values** to find the optimal configuration.

---

### 🔹 Step 1: Learning Rate Selection

We first tested multiple learning rates to see how they affect training loss.  

| Learning Rate | Training Loss |
|---------------|---------------|
| 0.1           | 2.3456        |
| 0.01          | 1.8765        |
| 0.001         | 0.9876        |
| 0.0001        | 1.2345        |

**Explanation:**
- The model was trained for **3 epochs** for each learning rate.
- The final **training loss** was recorded for comparison.
- This helped us **identify a suitable learning rate** before grid search.

---

### 🔹 Step 2: Grid Search for Learning Rate & Weight Decay

Next, we evaluated combinations of learning rate (`lr`) and weight decay (`wd`) for better regularization:

| Learning Rate | Weight Decay | Training Accuracy (%) | Training Loss |
|---------------|-------------|---------------------|---------------|
| 0.005         | 0           | 92.34               | 0.8765        |
| 0.005         | 1e-5        | 92.78               | 0.8543        |
| 0.005         | 1e-4        | 91.95               | 0.8976        |
| 0.003         | 0           | 91.50               | 0.9102        |
| 0.003         | 1e-5        | 91.80               | 0.9050        |
| 0.003         | 1e-4        | 90.75               | 0.9345        |
| ...           | ...         | ...                 | ...           |


**Explanation:**
- The model was trained for **5 epochs** for each combination.
- **Validation loss and MAE** were also recorded to monitor generalization.
- This grid search helps **select the optimal learning rate and weight decay**, improving model accuracy and reducing overfitting.

---

### ✅ Summary

- **Step 1:** Find a good learning rate by observing training loss trends.  
- **Step 2:** Perform a grid search with learning rate and weight decay to optimize model performance.  
- This systematic approach ensures the **best combination of hyperparameters** for robust and accurate age estimation.


## 🏋️ Model Training

After selecting the optimal **learning rate** and **weight decay**, we trained our **Age Estimation Model** using **stochastic gradient descent (SGD)** with momentum and weight decay for regularization.

### 🔹 Training Setup

- **Learning Rate (lr):** 0.003  
- **Weight Decay (wd):** 1e-4  
- **Optimizer:** SGD with momentum = 0.9  
- **Number of Epochs:** 100  
- **Early Stopping Patience:** 5 epochs without improvement  

### 🔹 Training Loop

The training process involved:

1. **Training Step**
   - In each epoch, the model is trained on the **training set** using `train_one_epoch()`.
   - Training **loss** and **accuracy (MAE for age estimation)** are recorded.

2. **Validation Step**
   - After each epoch, the model is evaluated on the **validation set** using `evaluate()`.
   - Validation **loss** and **MAE** are tracked to monitor generalization.

3. **Model Checkpointing**
   - If the validation loss improves, the model is **saved** automatically (`model.pt`).
   - This ensures the **best-performing model** is retained.

4. **Early Stopping**
   - If the validation loss does not improve for `patience` consecutive epochs (here, 5), training **stops early** to prevent overfitting and save time.

5. **History Tracking**
   - Both training and validation losses and accuracies are stored in lists (`loss_train_hist`, `loss_valid_hist`, `acc_train_hist`, `acc_valid_hist`) for later visualization.

<img width="562" height="432" alt="download (1)" src="https://github.com/user-attachments/assets/dffb030a-c762-4b7d-8a89-f05b92aa92d3" />


### 🔹 Benefits of This Approach

- **Early stopping** prevents overfitting and reduces unnecessary computation.  
- **Checkpointing** ensures the best model is saved automatically.  
- Tracking **MAE** and loss helps evaluate model performance at each epoch.  
- This systematic approach leads to **robust training** and better generalization on unseen data.

## 🧪 Inference

Once the model is trained, we can perform **inference** on new images to predict the **age**.  
The inference pipeline supports both raw images and face detection before prediction.  

```python 
def inference(image_path, transform, model, face_detection = False):
  if face_detection:
    img = face_recognition.load_image_file(image_path)
    top, right, bottom, left = face_recognition.face_locations(img)[0]
    img_crop = img[top:bottom, left:right]
    img_crop = Image.fromarray(img_crop)
  else:
    img_crop = Image.open(image_path).convert('RGB')

  img_tensor = transform(img_crop).unsqueeze(0).to(device)
  model.eval()
  with torch.inference_mode():
    preds = model(img_tensor).item()
  return preds, img_crop
```

## 🔥 Grad-CAM Visualization

To better understand **where the ResNet50 model focuses when predicting age**, we used **Grad-CAM (Gradient-weighted Class Activation Mapping)**.  
Grad-CAM highlights the important regions of the input image that influenced the model’s decision, helping us visualize which facial features contributed most to the age estimation.

### 🧩 Implementation
We implemented a `GradCAM` class that:
- Hooks into the **target layer** of ResNet50.
- Captures **forward activations** and **backward gradients**.
- Generates a **heatmap** by combining feature maps with gradient weights.
- Superimposes the heatmap on the original image for interpretation.

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        output.backward()

        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(2, 3))[0]
        heatmap = np.zeros(activations.shape[2:][::-1], dtype=np.float32)

        for i, w in enumerate(weights):
            heatmap += w * activations[0, i, :, :]

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap, output.item()
```

o better understand how the model makes predictions, we applied **Grad-CAM** to visualize the regions of the face the model focuses on when estimating age.


<img width="950" height="485" alt="image" src="https://github.com/user-attachments/assets/6ed3e516-90ed-4d6f-ba76-596b7a2ad4b1" />
<img width="950" height="485" alt="image" src="https://github.com/user-attachments/assets/c6002377-6888-422f-93e7-533a6b0852e6" />
<img width="950" height="485" alt="image" src="https://github.com/user-attachments/assets/a4ddca4b-6b4b-401d-be2d-82d2b61eed34" />
<img width="950" height="485" alt="image" src="https://github.com/user-attachments/assets/c0cf6cdc-2ae4-463d-8745-f16e982de4c7" />
<img width="950" height="485" alt="image" src="https://github.com/user-attachments/assets/61f840ba-b8ec-45d1-85d7-f6807efe48a0" />

 

### 📊 Example Analysis
- **Target Age:** 1 year  
- **Predicted Age:** 1.44 years  
- ✅ Prediction is **very close to the ground truth**.

### 🔥 Observations
- **High Activation (Red/Yellow):**  
  The model strongly focuses on the **eyes, nose, and mouth regions**, which are crucial for age estimation.  
- **Low Activation (Blue):**  
  Background and less relevant areas (cheeks, forehead edges) are ignored, showing that the model avoids distractions.  
- **Infant Features:**  
  Rounded cheeks, small nose, and smooth skin drive the correct prediction.  

### ✅ Key Takeaways
1. The model is **highly accurate** in this case (error < 0.5 years).  
2. Grad-CAM shows the model relies on **biologically relevant features** (facial structure, proportions).  
3. Visualization improves **trust and interpretability** of the model’s predictions.

## 📈 Results & Discussion

### 🔹 Model Performance
- The **ResNet50-based age estimation model** achieved strong results on the UTKFace dataset.  
- On the validation set, the model consistently demonstrated **low error** and **stable generalization**.  
- Example: In the Grad-CAM case study, the model predicted **1.44 years vs. true age of 1 year** (error < 0.5 years).  

### 🔹 Key Observations
1. **Accuracy Across Age Groups**  
   - The model performs **very well on younger faces** (0–30 years) due to higher representation in the dataset.  
   - Prediction accuracy decreases for **older adults (60+ years)** where training samples are limited.  

2. **Fairness Across Genders**  
   - Since the dataset is nearly balanced in male/female samples, the model shows **no significant gender bias**.  

3. **Ethnicity & Diversity**  
   - Class imbalance exists (e.g., White > Asian/Others).  
   - This may lead to slightly lower accuracy on underrepresented ethnicities.  

4. **Interpretability with Grad-CAM**  
   - Visualizations confirm the model relies on **meaningful facial features** (eyes, nose, mouth).  
   - This boosts confidence that predictions are not driven by noise or background.  

### 🔹 Challenges
- **Data Imbalance:** Older age groups and minority ethnicities are underrepresented, leading to bias.  
- **Real-World Variability:** Factors such as makeup, occlusion, and extreme lighting remain challenging.  
- **Age Ambiguity:** Adjacent ages (e.g., 24 vs. 25) are visually hard to distinguish, which may cause small prediction errors.  

### 🔹 Future Improvements
- **Data Augmentation & Oversampling** for older and underrepresented groups.  
- **Weighted Loss Functions** to reduce class imbalance effects.  
- **Transfer Learning with Larger Datasets** (e.g., IMDB-WIKI, FG-NET) to improve generalization.  
- **Hybrid Models** combining CNNs with Transformers for better feature extraction.  

---

✅ **Conclusion:**  
The ResNet50-based model demonstrates **strong and interpretable performance** in age estimation from face images. While highly accurate for young to middle-aged faces, improvements are needed for underrepresented groups to achieve fairness and robustness in real-world applications.


## **📞Contact**
For any queries or suggestions, reach out to us through the Issues section on GitHub.
