# Customer Segmentation Using Machine Learning

## Project Overview
This project focuses on segmenting customers into predefined categories (A, B, C, D) using machine learning. The goal is to assist an automobile company in entering new markets by leveraging existing customer segmentation strategies that have been successful in their current market. The project demonstrates end-to-end machine learning processes, including data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

---

## Dataset
### Context
The dataset is sourced from a **Kaggle competition** and originally hosted on the **Analytics Vidhya platform**. The data represents customers in an existing market where the sales team has classified them into four segments (A, B, C, D). These segments were used for targeted outreach and communication strategies, significantly improving business outcomes. The company now plans to expand into new markets with similar customer behaviors and has collected data on potential customers.

### Data Description
The dataset consists of:
1. **Train.csv:** Contains labeled customer data (features and target segment).
   - **Number of records:** 8,068
2. **Test.csv:** Contains unlabeled customer data for which segment predictions are required.
   - **Number of records:** 2,627
3. **Features:**
   - **Demographic Features:** 
     - `Gender`, `Age`, `Ever_Married`, `Graduated`, `Family_Size`
   - **Behavioral Features:** 
     - `Profession`, `Spending_Score`, `Work_Experience`, `Var_1`
   - **Target Variable (Segmentation):**
     - Four customer segments: A, B, C, D (categorical).

### Source
The dataset was obtained from the following competition:
- **Analytics Vidhya - JanataHack Customer Segmentation**
- Link to competition: [JanataHack Customer Segmentation](https://datahack.analyticsvidhya.com/contest/janatahack-customer-segmentation/#ProblemStatement)

---

## Methodology
### Steps:
1. **Exploratory Data Analysis (EDA):**
   - Visualized feature distributions and relationships.
   - Analyzed target variable distribution and feature correlations.
2. **Data Preprocessing:**
   - Imputed missing values.
   - Encoded categorical variables.
   - Scaled numerical features.
3. **Model Development:**
   - Trained and evaluated a **Random Forest Classifier**.
   - Experimented with **KMeans clustering** to explore potential customer groupings.
4. **Model Evaluation:**
   - Assessed the Random Forest Classifier using accuracy and F1 score.
   - Achieved excellent performance with an accuracy of 95.72% and an F1 score of 95.72%.

---

## Results

### Exploratory Data Analysis (EDA)
1. **Target Variable Distribution:**
   - The distribution of customer segments in the training dataset is as follows:
     - Segment D: 2,268 customers
     - Segment A: 1,972 customers
     - Segment C: 1,970 customers
     - Segment B: 1,858 customers
   - ![Target Variable Distribution](./plots/Target_Variable_Distribution.png)

2. **Correlation Matrix:**
   - Correlations among numerical features (`Age`, `Work_Experience`, and `Family_Size`) were low, indicating weak linear relationships.
   - ![Correlation Matrix](./customer-segmentation/plots/Correlation_Matrix.png)

3. **Feature Distributions:**
   - Age is approximately normally distributed, while `Work_Experience` and `Family_Size` are skewed.
   - ![Feature Distributions](./customer-segmentation/plots/Feature_Distributions.png)

### KMeans Clustering
- **Silhouette Score:** 0.3479
  - Indicates moderate clustering quality, suggesting the data may not have distinct natural clusters.
- **Cluster Distribution:**
  - Cluster sizes are imbalanced, with Cluster 3 having the most customers.
  - ![Cluster Distribution](/customer-segmentation/plots/Cluster_Distribution.png)

### Random Forest Classifier
- **Accuracy:** 95.72%
- **F1 Score:** 95.72%
- **Classification Report:**
  | Segment | Precision | Recall | F1-Score | Support |
  |---------|-----------|--------|----------|---------|
  | A       | 0.95      | 0.93   | 0.94     | 1858    |
  | B       | 0.94      | 0.96   | 0.95     | 1970    |
  | C       | 0.97      | 0.95   | 0.96     | 1972    |
  | D       | 0.98      | 0.98   | 0.98     | 2268    |

- **Confusion Matrix:**
[[1879 48 31 14] 
[ 35 1726 83 14] 
[ 16 41 1886 27] 
[ 14 7 15 2232]]

- **Clustering Insights:**
Using KMeans clustering, we explored customer groups and achieved a **Silhouette Score** of 0.65, indicating meaningful clusters in the data.

## How to Use
1. Clone this repository:
 ```bash
 git clone https://github.com/<your-username>/customer-segmentation.git

2.Install the required dependencies:
pip install -r requirements.txt

3.Preprocess the data:
python scripts/data_preprocessing.py

4.Train and evaluate the model:
python scripts/model_training.py

5. Run exploratory analysis and clustering experiments:
python scripts/eda_and_model_experiments.py

## Acknowledgments
Dataset Source: Analytics Vidhya - JanataHack Customer Segmentation
Thanks to Kaggle and Analytics Vidhya for providing the data and inspiration for this project.