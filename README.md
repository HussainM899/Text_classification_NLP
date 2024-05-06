# **Text Classification Web Application**

## **Objective**
The primary goal of this project is to develop and deploy a machine learning model that can accurately classify textual content into one of four categories: **World, Sports, Business, and Sci/Tech**. This application aims to demonstrate how natural language processing (**NLP**) can be utilized to organize and interpret large datasets of text, making information more accessible and actionable.

## **Overview**
This project encompasses the full lifecycle of a machine learning application, from data preparation and model training to deployment of a web application for user interaction. Key aspects include:

### **LIVE WEB APP LINK: https://huggingface.co/spaces/HussainM899/AG_News_V2**

- **Preprocessing** text data to remove noise and standardize the format.
- Utilizing **TF-IDF vectorization** to transform the text into a numerical representation.
- Training a **Logistic Regression** model for classification.
- Deploying the model through a web application using **FastAPI** and **Gradio**.
- Hosting the application on **Hugging Face Spaces** for public access.

## **Approach and Methodology**
### **Data Preparation and Preprocessing**
- **Dataset:** Utilized AG News labeled dataset suited for multiclass classification.
- **Cleaning:** Applied regex and **NLTK** for removing HTML tags, non-alphabetic characters, and stopwords. Implemented lemmatization to normalize words.
- **Vectorization:** Employed **TF-IDF** to convert text data into feature vectors, highlighting word importance.

### **Model Training and Evaluation**
- **Model Selection:** Compared several models (Naive Bayes, SVM, Random Forest, Logistic Regression) based on **accuracy, precision, recall,** and **F1 scores**.
- **Training:** Chose **Logistic Regression** for its efficiency and interpretability. The model was trained on the vectorized dataset.
- **Validation:** Validated the model using a held-out dataset to ensure robustness and generalizability.

## Evaluation Metrics

**Logistic Regression**
  - **Training Accuracy/F1 Score:** 93.41% / 93.40%
  - **Validation Accuracy/F1 Score:** 91.51% / 91.49%
    
The Logistic Regression model continues to demonstrate a balanced performance between training and validation sets, indicating good generalization without significant overfitting. It outperforms other models in terms of accuracy and F1 score across both sets.

### **Deployment**
- **FastAPI:** Developed **RESTful APIs** to handle classification requests.
- **Gradio:** Created a frontend interface for users to input text and view classifications.
- **Hugging Face Spaces:** Deployed the application for easy access by the broader community.

## **Dependencies**
- **Python**
- **scikit-learn**
- **nltk**
- **Gradio**
- **FastAPI**
- **joblib**

## **Results and Impact**
- The deployed model demonstrates high accuracy of **89%** and reliability in classifying text into the respective categories.
- The web application provides an accessible interface for users to leverage the model's capabilities, showcasing the practical application of NLP in organizing and interpreting text data.
- This project exemplifies the potential of machine learning in enhancing data accessibility and insights, contributing to more informed decision-making and knowledge discovery.

## **Getting Started**
To run this project locally, follow these steps:

### **Clone the Repository:**
```bash
git clone https://github.com/HussainM899/Text_classification_NLP.git
```

## **Install Dependencies**

To install the necessary dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## **Launch the Gradio interface**
To launch the Gradio interface, use the command below:
```
python app.py
```

## **Walkthrough**
For a detailed walkthrough, here's a presentation explaining the code and the work behind it: https://drive.google.com/file/d/1Iowj6iQIrnxtS1Y08LaetQ39EjU3iKJa/view?usp=drive_link

## **Contributions and Feedback**
We welcome contributions and feedback to improve this project. Please feel free to open issues or submit pull requests on GitHub.
