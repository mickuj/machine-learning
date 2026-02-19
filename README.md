# Machine Learning Labs

This repository contains my laboratory work from a **Machine Learning course (Python)**.  
It demonstrates hands-on experience with classical ML, neural networks, model evaluation, and hyperparameter tuning.

Although these are lab scripts, they cover a broad range of real ML workflows: data preprocessing, model comparison, validation, serialization, and experiment tracking.

---

## Tech Stack

- **Python**
- **scikit-learn** (classical ML, pipelines, cross-validation, tuning)
- **TensorFlow / Keras** (MLP, CNN, transfer learning, RNN, autoencoders)
- **pandas / numpy / matplotlib**
- **SciKeras**
- **KerasTuner**
- **TensorBoard**

---

## Repository Overview (Labs Summary)

### 1️⃣ Exploratory Data Analysis + Data Preparation (Housing Dataset)
- Data loading and visualization
- Correlation analysis
- Train/test split and dataset serialization  
File: `lab01.py`

---

### 2️⃣ Classical Regression
- Linear Regression baseline
- kNN Regression
- Polynomial Regression
- Evaluation using **MSE**
- Saving experiment results to `.pkl` files  
File: `lab02.py`

---

### 3️⃣ Classification – MNIST (SGD)
- Binary and multiclass classification
- Cross-validation
- Accuracy and confusion matrix
- Model evaluation and result persistence  
File: `lab03.py`

---

### 4️⃣ Support Vector Machines (SVM)
- LinearSVC with and without scaling (Pipeline)
- SVR (polynomial kernel)
- Hyperparameter tuning with **GridSearchCV**
- Performance comparison  
File: `lab04.py`

---

### 5️⃣ Decision Trees
- Classification and regression trees
- Hyperparameter selection (`max_depth`)
- Evaluation using **F1-score** and **MSE**
- Tree visualization with Graphviz  
File: `lab05.py`

---

### 6️⃣ Ensemble Methods
- VotingClassifier (hard & soft voting)
- Bagging & Pasting
- Random Forest
- AdaBoost
- Gradient Boosting
- Estimator ranking by accuracy  
File: `lab06.py`

---

### 7️⃣ Dimensionality Reduction (PCA)
- PCA with explained variance threshold
- Scaling + PCA pipeline
- Feature importance analysis from components  
File: `lab07.py`

---

### 8️⃣ Clustering (MNIST)
- KMeans with silhouette score comparison
- Mapping clusters to labels (confusion matrix analysis)
- DBSCAN with heuristic `eps` selection  
File: `lab08.py`

---

### 9️⃣ Neural Networks with Keras
- MLP for Fashion-MNIST classification
- Confidence-based single prediction
- Regression on California Housing dataset
- EarlyStopping
- TensorBoard logging
- Model saving (`.keras`)  
File: `lab09.py`

---

### 🔟 Autoencoders (Representation Learning)
- Stacked Autoencoder
- Convolutional Autoencoder
- Reconstruction visualization
- t-SNE on encoded representations
- Denoising with Dropout  
File: `lab10.py`

---

### 1️⃣1️⃣ CNN & Transfer Learning
- Custom CNN model
- Transfer learning with **Xception**
- Layer freezing and fine-tuning
- Train/validation/test accuracy comparison  
File: `lab11.py`

---

### 1️⃣2️⃣ Hyperparameter Optimization
- Custom Keras model builder
- SciKeras + RandomizedSearchCV
- KerasTuner (RandomSearch)
- TensorBoard integration
- Best parameter tracking and model persistence  
File: `lab12.py`

---

### 1️⃣3️⃣ Time Series Forecasting (Bike Sharing Dataset)
- Hourly resampling and preprocessing
- Baseline models (shifted day/week)
- Linear model vs LSTM architectures
- Deep LSTM and multivariate LSTM
- Evaluation using **MAE**
- Model serialization  
File: `lab13.py`
