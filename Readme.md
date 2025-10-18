# Breast Cancer Classification with SVM (Streamlit App)

This project is an interactive Streamlit web application that applies Support Vector Machines (SVM) to the Breast Cancer Wisconsin Dataset for classifying tumors as benign or malignant.

The application allows users to:

- Choose the SVM kernel (Linear or RBF)
- Explore the dataset
- View evaluation metrics (Accuracy, Precision, Recall, F1-score)
- Predict cancer likelihood for a new patient by entering feature values

---

## Features

- Interactive web-based UI with Streamlit
- Model training with Linear and RBF kernels
- Performance evaluation with classification metrics
- Visualization of dataset distribution and confusion matrix
- Patient prediction form for real-time classification

---

## Workflow Diagram

![SVM Breast Cancer Workflow](flowchart_of_breastCancer.png)

---

## Screenshots

Linear classification  
![Linear Classification](screenshots\linear_ui1.png)
RBF Classification  
![RBF Classification](screenshots\rbf_ui2.png)

New Data Prediction Form
![New Data Prediction Form](screenshots\prediction_newdata_rbfmodel_UI.png)

---

## Results

- Linear Kernel performs well for linearly separable data
- RBF Kernel captures non-linear decision boundaries more effectively
- Achieved high accuracy on test data
