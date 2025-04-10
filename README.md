
# **BankChurnPredictor-ANN**

BankChurnPredictor-ANN is a deep learning project designed to predict customer churn in the banking sector. The project uses an Artificial Neural Network (ANN) to analyze customer data and predict whether a customer who has recently opened a bank account is likely to leave the bank. By identifying potential churners early, banks can take proactive steps to retain customers.

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

---

## **Overview**

Customer churn is a critical issue in the banking industry, as retaining existing customers is more cost-effective than acquiring new ones. BankChurnPredictor-ANN is designed to forecast customer churn using historical data and help banks optimize their retention strategies. The project employs a deep learning model (ANN) to capture complex patterns and predict churn probabilities with high accuracy.

---

## **Features**
- **Customer Churn Prediction**: Identifies customers likely to leave after opening an account.
- **Artificial Neural Network (ANN)**: Utilizes a deep learning model to capture non-linear relationships in customer data.
- **Customizable**: Can be tailored for different banks and datasets.
- **Performance Metrics**: Evaluates model performance using accuracy, precision, recall, and F1-score.
- **Proactive Retention**: Provides insights to help banks develop retention strategies.

---

## **Installation**

To get started with BankChurnPredictor-ANN, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/01ankon01/BankChurnPredictor-ANN.git
cd BankChurnPredictor-ANN
pip install -r requirements.txt
```

---

## **Usage**

1. **Prepare the Dataset**: Place your customer data in the `/data` directory in CSV format. Ensure that it includes relevant features such as customer demographics, transaction history, and account details.
   
2. **Train the Model**: Use the following command to train the ANN model:

    ```bash
    python train.py --data ./data/Churn_Modelling.csv --epochs 50 --batch-size 32
    ```

3. **Make Predictions**: After training, use the trained model to predict churn probabilities:

    ```bash
    python predict.py --model ./models/trained_model.h5 --data ./data/new_customers.csv
    ```

---

## **Model Architecture**

The ANN model consists of multiple fully connected layers designed to capture relationships between customer features. The architecture includes:
- **Input Layer**: Accepts customer data features.
- **Hidden Layers**: Several hidden layers with ReLU activation.
- **Output Layer**: A single node with sigmoid activation for binary churn prediction (0 = stay, 1 = churn).

---

## **Dataset**

The model is trained using customer data that includes:
- **Customer demographics**: Age, gender, income, etc.
- **Account information**: Account age, balance, transaction history.
- **Activity metrics**: Frequency of transactions, customer support interactions.

*Note: The dataset must be pre-processed before training (e.g., handling missing values, normalization).*

---

## **Performance**

The model is evaluated using various performance metrics:
- **Accuracy**: Overall correctness of the model.
- **Precision**: Ratio of true positives to predicted positives.
- **Recall**: Ratio of true positives to actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

For a detailed visualization of model performance, visit the [Tableau Dashboard](https://public.tableau.com/views/DataModeling2/Balance?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link).

---

## **Contributing**

Contributions are welcome! Please feel free to submit issues or pull requests to enhance the project.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
