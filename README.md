# 👩‍⚕️🔬 AI-Powered Skin Disease Diagnosis: An MSc Research Project 🚀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-green?style=flat&logo=keras)](https://keras.io/)
[![Flask](https://img.shields.io/badge/Flask-black?style=flat&logo=flask)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-blue?style=flat&logo=scikit-learn)](https://scikit-learn.org/stable/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)](https://jupyter.org/)
[![GitHub](https://img.shields.io/badge/GitHub-black?style=flat&logo=github)](https://github.com/)

---

## ✨ Project Overview

This repository presents my Master of Science (MSc) research project focused on leveraging Artificial Intelligence for the automated diagnosis of common skin conditions and chronic wounds. The initiative addresses critical global healthcare challenges, including the high cost of treatment, diagnostic delays, and the significant morbidity associated with skin diseases. By developing an image-analysis-driven system, this project aims to enhance diagnostic accuracy and accessibility for conditions like **Diabetic Ulcers, Pressure Ulcers, Venomous Ulcers, Psoriasis, and Eczema**.

This project showcases an end-to-end Machine Learning pipeline, from data handling and deep learning model development to its deployment within an intuitive web application. It reflects a comprehensive understanding of AI's application in real-world medical scenarios.

## 🎯 Key Features & Accomplishments

* **Deep Learning Model Development (CNN):** Architected and optimized a robust **Convolutional Neural Network (CNN)** for multi-class image classification. This model forms the core of the diagnostic system, capable of learning intricate patterns from skin lesion images.
* **High Performance & Accuracy:** Achieved impressive model performance with a **training accuracy of 85.72%** and a **validation accuracy of 73.18%**. This demonstrates the model's effectiveness and generalisation capabilities on unseen data.
* **Comprehensive Data Handling:** Managed and preprocessed a substantial dataset of approximately **16,000 skin lesion images**. This involved integrating diverse sources, including a proprietary dataset collected under contractual agreement and publicly available Kaggle data.
* **Advanced Image Preprocessing Pipeline:** Implemented critical preprocessing techniques vital for deep learning:
    * **Image Resizing:** Standardising image dimensions for consistent model input.
    * **Data Augmentation:** Strategically generating new training samples (e.g., rotations, flips, zooms, shifts) to significantly expand dataset diversity, improve model robustness, and prevent overfitting.
    * **Normalisation:** Scaling pixel values to optimise neural network training efficiency and stability.
* **Comparative Algorithm Analysis:** Conducted a rigorous empirical study comparing the CNN's performance against traditional Machine Learning algorithms such as **Naïve Bayes, Decision Tree, Random Forest, and Support Vector Machine (SVM)**. This analysis empirically justified the selection of deep learning for superior accuracy in this complex image classification task.
* **Model Optimisation & Hyperparameter Tuning:** Applied advanced techniques like **GridSearchCV** for systematic hyperparameter tuning, maximising the CNN's performance and ensuring its robustness and generalisation capabilities.
* **Full-Stack Deployment with Flask:** Developed and integrated the trained CNN model into a functional, user-friendly **Flask-based web application**. This allows patients or users to easily upload skin lesion images for instant AI-driven diagnoses and receive relevant, suggested tips and treatments. This demonstrates proficiency in bringing AI models to production.
* **Comprehensive Model Evaluation:** Performed thorough model evaluation using industry-standard metrics, including **Confusion Matrix, Precision, Recall, and F1-score**, showcasing a deep understanding of model performance assessment beyond mere accuracy.
* **Future Vision & Scalability:** Identified clear pathways for future enhancements, such as leveraging greater computational resources to potentially achieve even higher diagnostic accuracies (projected 95%) and enabling wider accessibility through mobile application integration.

## 🛠️ Technologies & Libraries Used

* **Programming Language:** Python 🐍
* **Deep Learning Frameworks:** TensorFlow, Keras
* **Machine Learning Libraries:** scikit-learn
* **Web Framework:** Flask
* **Data Manipulation:** NumPy, Pandas
* **Data Visualization:** Matplotlib, Seaborn
* **Version Control:** Git, GitHub
* **Development Environment:** Jupyter Notebook 📓

## 📂 Project Structure & Contents

* `202124284_Code/`: Contains the Jupyter Notebooks detailing the model development and training processes.
    * `202124284_Final_CNN_Model_for_Website_integration.ipynb`: The finalised CNN model used for web application integration.
    * `202124284_MSc Research Project_Skin disease diagnosis_Multi model.ipynb`: Notebook exploring the comparative analysis of multiple machine learning models.
* `MSc in Artificial Intelligence and Data Science-dissertation-202124284.pdf`: My complete Master's dissertation. This document provides an in-depth academic overview of the project's background, detailed methodology, experimental setup, comprehensive results, discussions, and future work.
* `README.md`: This document, providing an overview of the project.

---

### ⚠️ Important Note on Data & Full Application Code:

To strictly adhere to academic integrity guidelines for my Master's program and to ensure the privacy and security of specific image data collected under a contractual agreement (with Simon Hudson), the full sensitive dataset (`skin_data_simon.zip`) and the complete, runnable Flask application package (`Application_flask.zip`) are **not publicly available** in this repository.

This measure is crucial for:

* **Preventing Misuse:** Ensuring the project cannot be directly used for academic dishonesty by future students.
* **Data Privacy & Security:** Protecting any potentially identifiable information or contractual obligations related to the data provider.

The uploaded `README.md`, the detailed Jupyter Notebooks, and the comprehensive Master's Dissertation aim to provide a complete understanding of the project's scope, rigorous methodology, and impactful results, showcasing my practical skills and theoretical knowledge in AI.

---

## 🙏 Acknowledgements

* **Data Provider:** Special thanks to Simon Hudson for generously providing a critical portion of the wound/skin lesion image dataset under a contractual agreement, which was foundational to this Master's research. His contribution was invaluable for the practical application of this project.

## 📞 Connect with me!

* [LinkedIn Profile](https://www.linkedin.com/in/venu-madhuri-yerramsetti-349057aa)
* Email: venumadhuri.y@gmail.com

---
