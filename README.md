# Spam Email Classification README

## Overview

This project focuses on detecting spam emails using a machine learning model implemented in Python. The model employs a **TF-IDF Vectorizer** for text feature extraction and a **Neural Network** built with TensorFlow/Keras for binary classification.

### Key Files:

1. `spam_email_Classification.ipynb`: Jupyter notebook containing the full implementation of the spam classification model.
2. `spam_mail_data.csv`: Dataset containing email messages and their labels (spam/ham).
3. `spam_model_efficient.h5`: Pre-trained model file for efficient reuse.

---

## Steps in the Implementation

### 1. **Data Loading and Preprocessing**

- **Dataset**: The data is loaded from `spam_mail_data.csv`. It consists of two columns:
  - `Message`: The email content.
  - `Category`: Labels indicating spam (1) or ham (0).
- **Shuffling and Sampling**: The dataset is shuffled, and only half the data is used for training/testing.
- **Label Encoding**: Labels are converted to numerical values (spam = 1, ham = 0).
- **Train-Test Split**: The dataset is split into 80% training and 20% testing data.

### 2. **Text Vectorization**

- **TF-IDF Vectorizer**: Converts text data into numerical feature vectors by calculating term frequency-inverse document frequency (TF-IDF) values.
- Preprocessing includes:
  - Lowercasing text.
  - Removing English stop words.

### 3. **Model Architecture**

- **Model**: A Sequential Neural Network with the following structure:
  - Input Layer: 128 neurons, ReLU activation.
  - Dropout Layer: Regularization with a 50% dropout rate.
  - Hidden Layer: 64 neurons, ReLU activation.
  - Dropout Layer: Regularization with a 30% dropout rate.
  - Output Layer: 1 neuron, Sigmoid activation for binary classification.

### 4. **Training the Model**

- **Loss Function**: Binary Crossentropy.
- **Optimizer**: Adam with a learning rate of 0.001.
- **Metrics**: Accuracy.
- The model is trained for 10 epochs with a batch size of 32 and validation on the test set.

### 5. **Saving and Loading the Model**

- The trained model is saved as `spam_model_efficient.h5` for future use.
- The model can be reloaded using TensorFlow's `load_model` function.

### 6. **Model Evaluation**

- Predictions are made on the test data, and probabilities are converted into binary classifications.
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### 7. **User Input Classification**

- The user can input a message to classify as spam or not spam.
- The message is preprocessed using the TF-IDF Vectorizer and passed to the trained model for prediction.

---

## Results

- **Evaluation Metrics**:
  - Accuracy: Achieved over 90% on the test set.
  - Precision, Recall, and F1 Score are calculated to assess performance comprehensively.
- **Prediction Example**:
  - Input: “Dear Coder, Codecademy on Infosys Springboard empowers millions by equipping them with essential tech skills...”
  - Output: “NOT SPAM”

---

## Requirements

- **Python Libraries**:
  - pandas
  - scikit-learn
  - tensorflow
- **Environment**: Google Colab or any Python environment with the above libraries installed.

---

## How to Run the Project

1. Place `spam_mail_data.csv` and `spam_model_efficient.h5` in your working directory.
2. Open `spam_email_Classification.ipynb` in a Jupyter environment.
3. Run the notebook cells step by step to:
   - Load the data.
   - Train or load the model.
   - Evaluate the model.
   - Test custom email messages for spam classification.
4. To use the pre-trained model provided (`spam_model_efficient.h5`):
   - Ensure the file is in the correct directory.
   - Uncomment the line `model=load_model("/content/drive/MyDrive/spam_mail_project/spam_model_efficient.h5")` in the notebook.
   - Run the cell to load the model and proceed directly to evaluation and predictions.

---

## Future Enhancements

- Use advanced preprocessing techniques such as stemming and lemmatization.
- Train with a larger dataset for improved generalization.
- Experiment with alternative architectures like RNNs or Transformers for text classification.

---

## Credits

- **Dataset**: from Kaggle.
- **Libraries Used**: TensorFlow, scikit-learn, pandas.
### Spam Email Classification README

## Overview

This project focuses on detecting spam emails using a machine learning model implemented in Python. The model employs a **TF-IDF Vectorizer** for text feature extraction and a **Neural Network** built with TensorFlow/Keras for binary classification.

### Key Files:

1. `spam_email_Classification.ipynb`: Jupyter notebook containing the full implementation of the spam classification model.
2. `spam_mail_data.csv`: Dataset containing email messages and their labels (spam/ham).
3. `spam_model_efficient.h5`: Pre-trained model file for efficient reuse.

---

## Steps in the Implementation

### 1. **Data Loading and Preprocessing**

- **Dataset**: The data is loaded from `spam_mail_data.csv`. It consists of two columns:
  - `Message`: The email content.
  - `Category`: Labels indicating spam (1) or ham (0).
- **Shuffling and Sampling**: The dataset is shuffled, and only half the data is used for training/testing.
- **Label Encoding**: Labels are converted to numerical values (spam = 1, ham = 0).
- **Train-Test Split**: The dataset is split into 80% training and 20% testing data.

### 2. **Text Vectorization**

- **TF-IDF Vectorizer**: Converts text data into numerical feature vectors by calculating term frequency-inverse document frequency (TF-IDF) values.
- Preprocessing includes:
  - Lowercasing text.
  - Removing English stop words.

### 3. **Model Architecture**

- **Model**: A Sequential Neural Network with the following structure:
  - Input Layer: 128 neurons, ReLU activation.
  - Dropout Layer: Regularization with a 50% dropout rate.
  - Hidden Layer: 64 neurons, ReLU activation.
  - Dropout Layer: Regularization with a 30% dropout rate.
  - Output Layer: 1 neuron, Sigmoid activation for binary classification.

### 4. **Training the Model**

- **Loss Function**: Binary Crossentropy.
- **Optimizer**: Adam with a learning rate of 0.001.
- **Metrics**: Accuracy.
- The model is trained for 10 epochs with a batch size of 32 and validation on the test set.

### 5. **Saving and Loading the Model**

- The trained model is saved as `spam_model_efficient.h5` for future use.
- The model can be reloaded using TensorFlow's `load_model` function.

### 6. **Model Evaluation**

- Predictions are made on the test data, and probabilities are converted into binary classifications.
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### 7. **User Input Classification**

- The user can input a message to classify as spam or not spam.
- The message is preprocessed using the TF-IDF Vectorizer and passed to the trained model for prediction.

---

## Results

- **Evaluation Metrics**:
  - Accuracy: Achieved over 90% on the test set.
  - Precision, Recall, and F1 Score are calculated to assess performance comprehensively.
- **Prediction Example**:
  - Input: “Dear Coder, Codecademy on Infosys Springboard empowers millions by equipping them with essential tech skills...”
  - Output: “NOT SPAM”

---

## Requirements

- **Python Libraries**:
  - pandas
  - scikit-learn
  - tensorflow
- **Environment**: Google Colab or any Python environment with the above libraries installed.

---

## How to Run the Project

1. Place `spam_mail_data.csv` and `spam_model_efficient.h5` in your working directory.
2. Open `spam_email_Classification.ipynb` in a Jupyter environment.
3. Run the notebook cells step by step to:
   - Load the data.
   - Train or load the model.
   - Evaluate the model.
   - Test custom email messages for spam classification.
4. To use the pre-trained model provided (`spam_model_efficient.h5`):
   - Ensure the file is in the correct directory.
   - Uncomment the line `model=load_model("/content/drive/MyDrive/spam_mail_project/spam_model_efficient.h5")` in the notebook.
   - Run the cell to load the model and proceed directly to evaluation and predictions.

---

## Future Enhancements

- Use advanced preprocessing techniques such as stemming and lemmatization.
- Train with a larger dataset for improved generalization.
- Experiment with alternative architectures like RNNs or Transformers for text classification.

---

## Credits

- **Dataset**: Source not mentioned (ensure proper citation if publicly available).
- **Libraries Used**: TensorFlow, scikit-learn, pandas.

# Spam Email Classification README

## Overview

This project focuses on detecting spam emails using a machine learning model implemented in Python. The model employs a **TF-IDF Vectorizer** for text feature extraction and a **Neural Network** built with TensorFlow/Keras for binary classification.

### Key Files:

1. `spam_email_Classification.ipynb`: Jupyter notebook containing the full implementation of the spam classification model.
2. `spam_mail_data.csv`: Dataset containing email messages and their labels (spam/ham).
3. `spam_model_efficient.h5`: Pre-trained model file for efficient reuse.

---

## Steps in the Implementation

### 1. **Data Loading and Preprocessing**

- **Dataset**: The data is loaded from `spam_mail_data.csv`. It consists of two columns:
  - `Message`: The email content.
  - `Category`: Labels indicating spam (1) or ham (0).
- **Shuffling and Sampling**: The dataset is shuffled, and only half the data is used for training/testing.
- **Label Encoding**: Labels are converted to numerical values (spam = 1, ham = 0).
- **Train-Test Split**: The dataset is split into 80% training and 20% testing data.

### 2. **Text Vectorization**

- **TF-IDF Vectorizer**: Converts text data into numerical feature vectors by calculating term frequency-inverse document frequency (TF-IDF) values.
- Preprocessing includes:
  - Lowercasing text.
  - Removing English stop words.

### 3. **Model Architecture**

- **Model**: A Sequential Neural Network with the following structure:
  - Input Layer: 128 neurons, ReLU activation.
  - Dropout Layer: Regularization with a 50% dropout rate.
  - Hidden Layer: 64 neurons, ReLU activation.
  - Dropout Layer: Regularization with a 30% dropout rate.
  - Output Layer: 1 neuron, Sigmoid activation for binary classification.

### 4. **Training the Model**

- **Loss Function**: Binary Crossentropy.
- **Optimizer**: Adam with a learning rate of 0.001.
- **Metrics**: Accuracy.
- The model is trained for 10 epochs with a batch size of 32 and validation on the test set.

### 5. **Saving and Loading the Model**

- The trained model is saved as `spam_model_efficient.h5` for future use.
- The model can be reloaded using TensorFlow's `load_model` function.

### 6. **Model Evaluation**

- Predictions are made on the test data, and probabilities are converted into binary classifications.
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### 7. **User Input Classification**

- The user can input a message to classify as spam or not spam.
- The message is preprocessed using the TF-IDF Vectorizer and passed to the trained model for prediction.

---

## Results

- **Evaluation Metrics**:
  - Accuracy: Achieved over 90% on the test set.
  - Precision, Recall, and F1 Score are calculated to assess performance comprehensively.
- **Prediction Example**:
  - Input: “Dear Coder, Codecademy on Infosys Springboard empowers millions by equipping them with essential tech skills...”
  - Output: “NOT SPAM”

---

## Requirements

- **Python Libraries**:
  - pandas
  - scikit-learn
  - tensorflow
- **Environment**: Google Colab or any Python environment with the above libraries installed.

---

## How to Run the Project

1. Place `spam_mail_data.csv` and `spam_model_efficient.h5` in your working directory.
2. Open `spam_email_Classification.ipynb` in a Jupyter environment.
3. Run the notebook cells step by step to:
   - Load the data.
   - Train or load the model.
   - Evaluate the model.
   - Test custom email messages for spam classification.
4. To use the pre-trained model provided (`spam_model_efficient.h5`):
   - Ensure the file is in the correct directory.
   - Uncomment the line `model=load_model("/content/drive/MyDrive/spam_mail_project/spam_model_efficient.h5")` in the notebook.
   - Run the cell to load the model and proceed directly to evaluation and predictions.

---

## Future Enhancements

- Use advanced preprocessing techniques such as stemming and lemmatization.
- Train with a larger dataset for improved generalization.
- Experiment with alternative architectures like RNNs or Transformers for text classification.

---

## Credits

- **Dataset**: Source not mentioned (ensure proper citation if publicly available).
- **Libraries Used**: TensorFlow, scikit-learn, pandas.

# Spam Email Classification README

## Overview

This project focuses on detecting spam emails using a machine learning model implemented in Python. The model employs a **TF-IDF Vectorizer** for text feature extraction and a **Neural Network** built with TensorFlow/Keras for binary classification.

### Key Files:

1. `spam_email_Classification.ipynb`: Jupyter notebook containing the full implementation of the spam classification model.
2. `spam_mail_data.csv`: Dataset containing email messages and their labels (spam/ham).
3. `spam_model_efficient.h5`: Pre-trained model file for efficient reuse.

---

## Steps in the Implementation

### 1. **Data Loading and Preprocessing**

- **Dataset**: The data is loaded from `spam_mail_data.csv`. It consists of two columns:
  - `Message`: The email content.
  - `Category`: Labels indicating spam (1) or ham (0).
- **Shuffling and Sampling**: The dataset is shuffled, and only half the data is used for training/testing.
- **Label Encoding**: Labels are converted to numerical values (spam = 1, ham = 0).
- **Train-Test Split**: The dataset is split into 80% training and 20% testing data.

### 2. **Text Vectorization**

- **TF-IDF Vectorizer**: Converts text data into numerical feature vectors by calculating term frequency-inverse document frequency (TF-IDF) values.
- Preprocessing includes:
  - Lowercasing text.
  - Removing English stop words.

### 3. **Model Architecture**

- **Model**: A Sequential Neural Network with the following structure:
  - Input Layer: 128 neurons, ReLU activation.
  - Dropout Layer: Regularization with a 50% dropout rate.
  - Hidden Layer: 64 neurons, ReLU activation.
  - Dropout Layer: Regularization with a 30% dropout rate.
  - Output Layer: 1 neuron, Sigmoid activation for binary classification.

### 4. **Training the Model**

- **Loss Function**: Binary Crossentropy.
- **Optimizer**: Adam with a learning rate of 0.001.
- **Metrics**: Accuracy.
- The model is trained for 10 epochs with a batch size of 32 and validation on the test set.

### 5. **Saving and Loading the Model**

- The trained model is saved as `spam_model_efficient.h5` for future use.
- The model can be reloaded using TensorFlow's `load_model` function.

### 6. **Model Evaluation**

- Predictions are made on the test data, and probabilities are converted into binary classifications.
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### 7. **User Input Classification**

- The user can input a message to classify as spam or not spam.
- The message is preprocessed using the TF-IDF Vectorizer and passed to the trained model for prediction.

---

## Results

- **Evaluation Metrics**:
  - Accuracy: Achieved over 90% on the test set.
  - Precision, Recall, and F1 Score are calculated to assess performance comprehensively.
- **Prediction Example**:
  - Input: “Dear Coder, Codecademy on Infosys Springboard empowers millions by equipping them with essential tech skills...”
  - Output: “NOT SPAM”

---

## Requirements

- **Python Libraries**:
  - pandas
  - scikit-learn
  - tensorflow
- **Environment**: Google Colab or any Python environment with the above libraries installed.

---

## How to Run the Project

1. Place `spam_mail_data.csv` and `spam_model_efficient.h5` in your working directory.
2. Open `spam_email_Classification.ipynb` in a Jupyter environment.
3. Run the notebook cells step by step to:
   - Load the data.
   - Train or load the model.
   - Evaluate the model.
   - Test custom email messages for spam classification.
4. To use the pre-trained model provided (`spam_model_efficient.h5`):
   - Ensure the file is in the correct directory.
   - Uncomment the line `model=load_model("/content/drive/MyDrive/spam_mail_project/spam_model_efficient.h5")` in the notebook.
   - Run the cell to load the model and proceed directly to evaluation and predictions.

---

## Future Enhancements

- Use advanced preprocessing techniques such as stemming and lemmatization.
- Train with a larger dataset for improved generalization.
- Experiment with alternative architectures like RNNs or Transformers for text classification.

---

## Credits

- **Dataset**: Source not mentioned (ensure proper citation if publicly available).
- **Libraries Used**: TensorFlow, scikit-learn, pandas.

# Spam Email Classification README

## Overview

This project focuses on detecting spam emails using a machine learning model implemented in Python. The model employs a **TF-IDF Vectorizer** for text feature extraction and a **Neural Network** built with TensorFlow/Keras for binary classification.

### Key Files:

1. `spam_email_Classification.ipynb`: Jupyter notebook containing the full implementation of the spam classification model.
2. `spam_mail_data.csv`: Dataset containing email messages and their labels (spam/ham).
3. `spam_model_efficient.h5`: Pre-trained model file for efficient reuse.

---

## Steps in the Implementation

### 1. **Data Loading and Preprocessing**

- **Dataset**: The data is loaded from `spam_mail_data.csv`. It consists of two columns:
  - `Message`: The email content.
  - `Category`: Labels indicating spam (1) or ham (0).
- **Shuffling and Sampling**: The dataset is shuffled, and only half the data is used for training/testing.
- **Label Encoding**: Labels are converted to numerical values (spam = 1, ham = 0).
- **Train-Test Split**: The dataset is split into 80% training and 20% testing data.

### 2. **Text Vectorization**

- **TF-IDF Vectorizer**: Converts text data into numerical feature vectors by calculating term frequency-inverse document frequency (TF-IDF) values.
- Preprocessing includes:
  - Lowercasing text.
  - Removing English stop words.

### 3. **Model Architecture**

- **Model**: A Sequential Neural Network with the following structure:
  - Input Layer: 128 neurons, ReLU activation.
  - Dropout Layer: Regularization with a 50% dropout rate.
  - Hidden Layer: 64 neurons, ReLU activation.
  - Dropout Layer: Regularization with a 30% dropout rate.
  - Output Layer: 1 neuron, Sigmoid activation for binary classification.

### 4. **Training the Model**

- **Loss Function**: Binary Crossentropy.
- **Optimizer**: Adam with a learning rate of 0.001.
- **Metrics**: Accuracy.
- The model is trained for 10 epochs with a batch size of 32 and validation on the test set.

### 5. **Saving and Loading the Model**

- The trained model is saved as `spam_model_efficient.h5` for future use.
- The model can be reloaded using TensorFlow's `load_model` function.

### 6. **Model Evaluation**

- Predictions are made on the test data, and probabilities are converted into binary classifications.
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### 7. **User Input Classification**

- The user can input a message to classify as spam or not spam.
- The message is preprocessed using the TF-IDF Vectorizer and passed to the trained model for prediction.

---

## Results

- **Evaluation Metrics**:
  - Accuracy: Achieved over 90% on the test set.
  - Precision, Recall, and F1 Score are calculated to assess performance comprehensively.
- **Prediction Example**:
  - Input: “Dear Coder, Codecademy on Infosys Springboard empowers millions by equipping them with essential tech skills...”
  - Output: “NOT SPAM”

---

## Requirements

- **Python Libraries**:
  - pandas
  - scikit-learn
  - tensorflow
- **Environment**: Google Colab or any Python environment with the above libraries installed.

---

## How to Run the Project

1. Place `spam_mail_data.csv` and `spam_model_efficient.h5` in your working directory.
2. Open `spam_email_Classification.ipynb` in a Jupyter environment.
3. Run the notebook cells step by step to:
   - Load the data.
   - Train or load the model.
   - Evaluate the model.
   - Test custom email messages for spam classification.
4. To use the pre-trained model provided (`spam_model_efficient.h5`):
   - Ensure the file is in the correct directory.
   - Uncomment the line `model=load_model("/content/drive/MyDrive/spam_mail_project/spam_model_efficient.h5")` in the notebook.
   - Run the cell to load the model and proceed directly to evaluation and predictions.

---

## Future Enhancements

- Use advanced preprocessing techniques such as stemming and lemmatization.
- Train with a larger dataset for improved generalization.
- Experiment with alternative architectures like RNNs or Transformers for text classification.

---

## Credits

- **Dataset**: Source not mentioned (ensure proper citation if publicly available).
- **Libraries Used**: TensorFlow, scikit-learn, pandas.

##Output
![output](https://github.com/user-attachments/assets/d5ddfca7-26a6-4690-a9a7-f0b5d1720d29)


