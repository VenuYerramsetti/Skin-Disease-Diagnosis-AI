# MSc Research Project: AI-Powered Skin Disease Diagnosis (Image Analysis)

## Project Overview

This repository showcases my Master's research project focused on developing an Artificial Intelligence solution for the diagnosis of common skin conditions and chronic wounds using deep learning and image analysis. The project demonstrates an end-to-end machine learning pipeline, from data acquisition and preprocessing to model training, evaluation, and deployment within a web application.

The primary goal was to create an accessible system to aid in the early detection and management of skin-related issues, addressing challenges like high treatment costs, time-consuming diagnoses, and the need for specialized medical expertise.

## Key Features & Contributions

* **Deep Learning Model Development:** Engineered a Convolutional Neural Network (CNN) for multi-class classification of five skin diseases/wound types: Diabetic, Pressure Ulcer, Venomous Ulcer, Psoriasis, and Eczema.
* **High Performance:** Achieved a robust training accuracy of **85.72%** and a validation accuracy of **73.18%**, demonstrating the model's effectiveness in a complex medical imaging domain.
* **Comprehensive Data Handling:** Managed and processed a diverse dataset of approximately 16,000 skin lesion images, integrating both proprietary data (from a contractual agreement) and publicly available sources (Kaggle).
* **Advanced Image Preprocessing:** Implemented crucial steps including image resizing, strategic data augmentation (e.g., rotations, flips, zooms) to enhance dataset diversity and prevent overfitting, and normalization for optimal model training.
* **Comparative Analysis:** Conducted a rigorous comparison of the CNN model's performance against traditional machine learning algorithms (Naïve Bayes, Decision Tree, Random Forest, Support Vector Machine), validating the superior efficacy of deep learning for this image-based diagnostic task.
* **Model Optimization:** Utilized hyperparameter tuning techniques (e.g., GridSearchCV) to fine-tune model parameters, leading to improved accuracy and generalization.
* **Web Application Integration:** Developed a Flask-based web application that allows users to upload skin lesion images for real-time AI-driven diagnosis and receive relevant tips and treatment suggestions.
* **Full ML Lifecycle Proficiency:** Gained hands-on experience across the entire machine learning pipeline: data acquisition, cleaning, feature engineering (implicit in CNNs), model selection, training, evaluation (using Confusion Matrix, Precision, Recall, F1-score), and deployment.
* **Future Scope:** Identified clear pathways for future enhancements, including leveraging increased computational resources to potentially achieve higher diagnostic accuracies (projected 95%) and integrating the solution into mobile applications for broader accessibility.

## Skills Demonstrated

* **Artificial Intelligence & Machine Learning:** Deep Learning, Convolutional Neural Networks (CNNs), Supervised Learning, Model Training, Evaluation & Optimization.
* **Programming & Frameworks:** Python, TensorFlow, Keras, Flask (Web Framework).
* **Data Science:** Image Processing, Data Preprocessing, Data Augmentation, Dataset Management, Performance Metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
* **Tools & Methodologies:** Git, GitHub, Hyperparameter Tuning, Problem Solving, Research & Analysis.
* **Domain Expertise:** Healthcare Technology, Medical Imaging, AI in Diagnostics.

## Project Structure & Contents

This repository contains the core components of my Master's research project.

* `202124284_Code/`: Contains Jupyter Notebooks illustrating the model development and training process.
    * `202124284_Final_CNN_Model_for_Website_integration.ipynb`: Final CNN model for web application.
    * `202124284_MSc Research Project_Skin disease diagnosis_Multi model.ipynb`: Notebook exploring multiple ML models.
* `MSc in Artificial Intelligence and Data Science-dissertation-202124284.pdf`: My complete Master's dissertation providing an in-depth overview of the project, methodology, results, and discussions.
* `coversheet2020_202124284.docx`: Project coversheet.
* `README.md`: This file.

**Important Note on Data and Full Application Code:**

Due to academic integrity guidelines for my Master's program and the sensitive nature of the image data collected under a contractual agreement (with Simon Hudson), the full dataset (`skin_data_simon.zip`) and the complete, runnable Flask application package (`Application_flask.zip`) are **not** publicly available in this repository. This measure is taken to prevent potential misuse, ensure data privacy, and uphold academic fairness for future students.


## How to Explore (Without Full Data)

While the full dataset is not public, you can review:

* The **Jupyter Notebooks** in `202124284_Code/` to understand the model architecture, training steps, and evaluation process.
* My **Master's Dissertation (PDF)** for a complete and detailed breakdown of the research, data sources (publicly available vs. contractual), methodology, and comprehensive results.
* The **images of the Flask application UI** (as depicted in the dissertation and screenshots) to visualize the user experience.

## Contact

Feel free to connect with me on   * [LinkedIn Profile](https://www.linkedin.com/in/venu-madhuri-yerramsetti-349057aa)
  * Email: venumadhuri.y@gmail.com

---