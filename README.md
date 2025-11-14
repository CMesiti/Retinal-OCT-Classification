
# Retinal-OCT-Classification
Independent Research on Retinal OCT (Optical Coherence Topography) image classification, OCT is a non-invasive diagnostic technique highlighting the retina's distinctive layers. This research is following what I have learned with the OpenCV library and working with images. The goal of this project is to learn how to build a neural network with pytorch and enhance my understanding on deep learning concepts. Eventually leading to an end product that is a multi-class OCT image classification model. Manual detection from professionals is prone to errors and time-consuming. This project explores a deep learning approach as an automatic classification for OCT images. This study includes comparing two models, a fine-tuned EfficientNetB0-v2 transfer learning model and a custom-build Convolutional Neural Network inspired by VGG-16 to evaluate their effectiveness in distinguishing between normal and pathological cases


<img width="1201" height="699" alt="Predicted vs Actual" src="https://github.com/user-attachments/assets/33a84ec5-d810-4d9d-8053-8203ace54fbf" />


# Research Topics
### AI Applications in Medical Image Classification
- Medical image classification leverages AI to assist in diagnosing diseases from imaging modalities such as Optical Coherence Tomography (OCT). Deep learning-based CNNs can help detect abnormalities like Age-related Macular Degeneration (AMD) and Diabetic Retinopathy with high accuracy.

### CNN Architectures of VGG-16 and EfficientNet
- VGG-16 is a deep convolutional neural network known for its uniform 3×3 convolutional filters and deep hierarchical structure. EfficientNet, on the other hand, uses a compound scaling method to balance depth, width, and resolution for improved performance and efficiency.

### Implementation
- PyTorch is an open-source deep-learning framework widely used for research and production applications. It provides dynamic computational graphs, making it flexible for designing and modifying deep-learning models.
- The classification models were implemented using PyTorch, utilizing pre-trained architectures (EfficientNetV2) and a custom CNN model. Training involved using torchvision for dataset augmentation, torch.nn for model definition, and torch.optim for optimization.


### Custom Datasets
- A custom dataset class (OCTDataset) was created in PyTorch to load and preprocess OCT images. It applies transformations like contrast enhancement, brightness adjustments, Gaussian blur, and resizing, improving image quality before feeding them into the model.

### CNN Architecture / Transfer Learning
- Transfer Learning: Pre-trained EfficientNetV2 was fine-tuned for OCT classification. The fully connected layers were modified to match the dataset's three-class classification.
- Custom CNN: A deep CNN model was built, inspired by VGG-16, using multiple convolutional layers with batch normalization, max pooling, and fully connected layers.

### EfficientNet Architecture
EfficientNetV2 was used due to its superior performance in medical imaging tasks. It features:
- Depthwise-separable convolutions for efficiency
- Squeeze-and-Excitation blocks for better feature learning
- Adaptive scaling of depth, width, and resolution for optimized model performance
- The model was trained with CrossEntropyLoss and optimized with Adam.

### Custom CNN Architecture and Methods
The custom CNN architecture consists of:
- Multiple convolutional layers with increasing filter sizes (from 32 to 256)
- Batch normalization for stable learning
- Max pooling to downsample feature maps
- Fully connected layers with dropout for regularization
- ReLU activations and softmax at the output
- Hooks were implemented to extract intermediate feature maps for visualization.

### Models and Techniques
Primary model choice: EfficientNetV2 (IMAGENET1K_V1)
Loss function: CrossEntropyLoss
Optimizer: Adam (LR = 0.001)
Custom model: Inspired by VGG-16, tested alongside EfficientNetV2
(Work in Progress)...
Model performance comparison (EfficientNetV2 vs. Custom CNN)
Hyperparameter tuning
Experimenting with different learning rates and augmentation techniques


<img width="706" height="527" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/62a925b7-4c0a-4822-81b1-615b8a4e2c18" />



---

### What’s Next
Fine-tuning the custom CNN model for better convergence
Implementing learning rate schedulers
Exploring ensemble techniques by combining both models
Deployment considerations for real-world OCT analysis
  #### Evaluation
  Model evaluation includes:
  - Loss analysis: Monitoring training/validation loss curves
  - Accuracy metrics: Comparing classification accuracy
  - Feature visualization: Understanding model interpretability
  - ROC-AUC scores: Assessing model confidence
  Metrics Including:
  - Accuracy: Measures correct predictions
  - Precision & Recall: Evaluates performance on individual classes
  - F1-score: Balances precision and recall
  - Confusion Matrix: Analyzes misclassifications
  - AUC-ROC Curve: Checks model’s ability to distinguish between classes


> Some of this code, particularly the pytorch lesssons can be attributed to @danielBourke and his pytorch course on Youtube which I followed thoroughly https://www.youtube.com/@mrdbourke
Other Sources:
## Sources
- [GitHub CNN Architecture](https://github.com/mr7495/OCT-classification)

- [Detection and Classification Methods](https://link.springer.com/article/10.1007/s10462-024-10883-3#Sec2)

- [OCT-based deep-learning models for the identification of retinal key signs](https://www.nature.com/articles/s41598-023-41362-4)

- [Evaluating deep learning models for classifying OCT images with limited data and noisy labels](https://www.nature.com/articles/s41598-024-81127-1)

- [Optical Coherence Tomography Image Classification Using Hybrid Deep Learning and Ant Colony Optimization](https://pmc.ncbi.nlm.nih.gov/articles/PMC10422382/)

- [Pytorch Docs](https://pytorch.org/tutorials/beginner/ptcheat.html)

- [Pytorch Course](https://www.youtube.com/watch?v=V_xro1bcAuA&t=19230s)

- [Pytorch Book](https://www.learnpytorch.io/02_pytorch_classification/)

- [Pytorch Course Materials](https://github.com/mrdbourke/pytorch-deep-learning)

- [Hugging Face](https://huggingface.co/models?library=timm,pytorch&sort=trending)

- [Kaggle](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data/comments)

- [Pytorch CNN Model GitHub](https://github.com/shre-db/OCT-Retinal-Disease-Detection-CNN)

- [Article On CNN Models](https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48)

- [ML TIPS](https://www.youtube.com/watch?v=oMc9StPVzOU)

- [Machine Learning in Clinical Interpretation ]( https://www.mdpi.com/2306-5354/10/4/407)
- [detecting and classifying age](https://link.springer.com/article/10.1007/s10462-024-10883-3#Sec2)
- [GAN network Article](https://link.springer.com/article/10.1007/s00521-021-05826-w)
- [Evaluation Metrics](https://wandb.ai/claire-boetticher/mendeleev/reports/Data-Visualization-for-Image-Classification--VmlldzozMjU5NDUz)
