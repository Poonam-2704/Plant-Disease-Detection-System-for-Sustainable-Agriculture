# Plant-Disease-Detection-System-for-Sustainable-Agriculture

An advanced Convolutional Neural Network (CNN) model for detecting and classifying plant diseases from leaf images. This project aims to assist farmers and agriculturists in identifying diseases early, ensuring timely treatment and improved crop health.

🌟 About the Project
This project leverages deep learning to classify plant diseases with high accuracy. By analyzing images of leaves, the CNN model identifies the type of disease, enabling precise and data-driven agricultural decision-making.

✨ Key Features
Accurate Disease Detection: High accuracy in identifying multiple plant diseases.
Custom CNN Model: Optimized architecture tailored for image classification.
Scalable Solution: Handles large datasets and diverse plant species.
Confusion Matrix & Metrics: Detailed analysis of model performance.
Data Augmentation: Improves model generalization through random transformations.

🛠️ Technologies Used
Programming Language: Python
Deep Learning Framework: TensorFlow and Keras
Visualization: Matplotlib, Seaborn
Data Processing: NumPy, Pandas
Jupyter Notebook: For experimentation and analysis

📂 Dataset
Source: C:\Edunet\Dataset
Size: [Number of images and total size, e.g., 50,000 images and 2 GB]
Classes: Includes diseases like [list a few diseases, e.g., Powdery Mildew, Leaf Spot, Rust, etc.].
Format: JPEG/PNG images categorized by folder.

🧠 Model Architecture
The model uses a custom Convolutional Neural Network with the following key layers:
Convolutional Layers: Extract features from input images.
Pooling Layers: Downsample feature maps to reduce complexity.
Dropout Layers: Prevent overfitting by random neuron deactivation.
Fully Connected Layers: Perform final classification based on learned features.
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Evaluation Metrics: Accuracy, Precision, Recall, F1-score

📊 Results
Training Accuracy: [e.g., 95%]
Validation Accuracy: [e.g., 93%]
Confusion Matrix: [Include or link to a visualization]
Sample Predictions: Correct classifications and areas of improvement.
![Screenshot 2025-01-28 201845](https://github.com/user-attachments/assets/39a90120-fceb-430f-a179-e7b66fcafaab)
![Screenshot 2025-01-28 201959](https://github.com/user-attachments/assets/1faa4d2a-9b85-4c86-afd3-9234534aaefb)
![Screenshot 2025-01-28 201945](https://github.com/user-attachments/assets/94de9d00-ac47-4caa-9ba0-0b2e8961290a)




🚀 Installation
Clone the repository:
git clone https://github.com/yourusername/plant-disease-classification.git
cd plant-disease-classification

Install dependencies:
pip install -r requirements.txt
Download the dataset and place it in the /data folder.
Run the Jupyter Notebook or Python script to train the model.

🖥️ Usage
Train the Model: Run the following command:
python train_model.py
Evaluate the Model: Use the evaluation script to test the model:
python evaluate_model.py
Make Predictions: Use the prediction script to classify new images:
python predict.py --image_path path_to_image.jpg

🤝 Contributing
Contributions are welcome! To contribute:
Fork the repository.
Create a new branch for your feature (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature/YourFeature).
Open a Pull Request.
