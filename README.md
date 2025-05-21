## 🔍 Outlier Detection in Breast Cancer Data  
**Tech Stack:** Java, Spring Boot, [ELKI (Environment for Developing KDD-Applications Supported by Index-Structures)](https://elki-project.github.io/)

🧠 Project Overview
This project focuses on anomaly detection in Breast Cancer data using the Local Outlier Factor (LOF) algorithm with K-Nearest Neighbors (KNN) and High Contrast Subspaces (HICS) for effective subspace selection.

⚙️ Key Features
✅ LOF-based Anomaly Detection: Applied the LOF algorithm using KNN to detect anomalies in medical data.

✅ High Contrast Subspace Selection: Implemented HICS to identify relevant subspaces for better feature contrast and anomaly identification.

✅ Custom Feature Selection Methods:

selectHighContrastSubspaces

selectRandomFeatures

filterData

calculateContrast

✅ Outlier Reporting: Identified and printed the top 10 outliers based on LOF scores.

✅ Model Evaluation:

Evaluated results using a Confusion Matrix

Plotted the ROC curve and computed the AUC to measure detection performance.

This project demonstrates a practical application of subspace analysis and anomaly detection in medical datasets using Java and ELKI.
