# Deep Learning & Machine Learning Portfolio

##  Overview
This repository serves as a code portfolio demonstrating the implementation and comparative analysis of various Neural Network architectures against Classical Machine Learning baselines. The projects cover **Natural Language Processing (NLP)** and **Computer Vision (CV)** tasks, utilizing **PyTorch** for deep learning and **Scikit-Learn** for statistical modeling.

---

## Project Breakdown

### 1. Sentiment Analysis with Deep Learning (NLP)
* **File:** [`biLSTM_imdb.ipynb`](./biLSTM_imdb.ipynb)
* **Task:** Binary Sentiment Classification on the IMDB Movie Reviews dataset.
* **Architecture:** Stacked **Bidirectional LSTM (BiLSTM)** with Global Max Pooling.
* **Key Techniques:**
    * **Word Embeddings:** Manually mapped vocabulary to pre-trained **Word2Vec (Google News 300d)** vectors.
    * **Frozen Layers:** Leveraged Transfer Learning by freezing embedding weights to prevent overfitting on limited data.
    * **Custom Data Pipeline:** Implemented custom PyTorch `Dataset` and `DataLoader` classes for efficient batching and tokenization.

### 2. Classical ML Benchmark (NLP)
* **File:** [`classic_ml_imdb.ipynb`](./classic_ml_imdb.ipynb)
* **Task:** Establishing a statistical baseline to evaluate the performance gain of Deep Learning methods.
* **Models Evaluated:**
    * **Logistic Regression (SGD):**
    * **Random Forest:**
    * **AdaBoost:** 
* **Feature Engineering:**
    * **N-Grams:** Generated Unigrams and Bigrams using `CountVectorizer`.
    * **Dimensionality Reduction:** Applied **Chi-Squared ($\chi^2$)** statistical testing to select the top 10,000 discriminative features.

### 3. Image Classification with ResNet (Computer Vision)
* **File:** [`ResNet_fine-tune_FashionMNIST.ipynb`](./ResNet_fine-tune_FashionMNIST.ipynb)
* **Task:** Multi-class classification of fashion articles (FashionMNIST) achieving **93.75% Test Accuracy**.
* **Architecture:** Adapted **ResNet18** (Residual Neural Network).
* **Key Techniques:**
    * **Transfer Learning:** Fine-tuned a ResNet18 model pre-trained on ImageNet.
    * **Architecture Modification:** Modified the input convolutional layer to accept **1-channel grayscale** images (vs. standard 3-channel RGB) and replaced the final Fully Connected layer.
    * **Data Augmentation:** Utilized `torchvision.transforms` (Random Rotation, Horizontal Flip) to improve model generalization.

---

## Tech Stack
* **Deep Learning Framework:** PyTorch
* **Machine Learning:** Scikit-Learn
* **Data Processing:** Pandas, NumPy, Hugging Face `datasets`
* **Visualization:** Matplotlib
* **Embeddings:** Gensim (Word2Vec)

---
*Created by Paris Salonikios*