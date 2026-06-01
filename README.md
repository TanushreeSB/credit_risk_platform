# credit_risk_platform


## 1. Project Overview

The Credit Risk Platform is an AI-powered solution that predicts the probability of loan default using machine learning and provides business-friendly insights through explainable AI and Natural Language to SQL querying.

The system enables:

* Credit default prediction
* Model training and evaluation
* Explainable risk assessment
* Business rule generation
* Natural language interaction with data
* Containerized deployment using Docker

---

# 2. Architecture Overview

## High-Level Architecture

```text
                    +------------------+
                    | Home Credit Data |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Data Processing  |
                    | Preprocessing    |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | LightGBM Model   |
                    | Training         |
                    +--------+---------+
                             |
          +------------------+------------------+
          |                                     |
          v                                     v
+------------------+                +------------------+
| Risk Prediction  |                | Explainability   |
| (Probability)    |                | SHAP Analysis    |
+------------------+                +------------------+

                             |
                             v

                    +------------------+
                    | Talk-to-Data     |
                    | NL → SQL Engine  |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Query Results    |
                    +------------------+
```

---

# 3. Project Structure

```text
credit_risk_platform/
├── data/
├── documents/
├── notebooks/
├── src/
│   ├── data/
│   ├── ml/
│   ├── talk_to_data/
│   └── utils/
├── sql/
├── models/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 4. Major Design Decisions

### Why LightGBM?

* Designed for tabular financial datasets.
* Handles missing values efficiently.
* Faster training compared to many ensemble models.
* Produces strong predictive performance on imbalanced datasets.

### Why Modular Architecture?

Each component has a single responsibility:

* Data Module → Loading and preprocessing
* ML Module → Training and prediction
* Talk-to-Data Module → NL-to-SQL pipeline
* Utils Module → Shared utilities

This improves maintainability, testing, and scalability.

### Why Natural Language to SQL?

Business users may not know SQL.

The NL-to-SQL layer enables users to ask questions such as:

> "What is the average credit amount for high-risk applicants?"

and receive database results automatically.

---

# 5. Model Selection Rationale

Several algorithms were considered:

| Model               | Reason                                 |
| ------------------- | -------------------------------------- |
| Logistic Regression | Baseline model                         |
| Random Forest       | Strong ensemble baseline               |
| XGBoost             | High predictive performance            |
| LightGBM            | Faster training and excellent accuracy |

### Final Selection: LightGBM

LightGBM was selected because it provides:

* High AUC performance
* Fast training
* Native handling of missing values
* Scalability to large datasets

---

# 6. Class Imbalance Strategy

The Home Credit dataset contains significantly more non-default cases than default cases.

To address class imbalance:

### Techniques Used

* Stratified train-test split
* LightGBM class weighting
* Evaluation using ROC-AUC rather than accuracy

### Why Not Accuracy?

A model predicting all applicants as non-defaulters could still achieve high accuracy.

ROC-AUC and Recall provide better insight into default detection performance.

---

# 7. Evaluation Metrics and Results

### Metrics Used

* ROC-AUC
* Precision
* Recall
* F1 Score

### Sample Results

| Metric    | Score |
| --------- | ----- |
| ROC-AUC   | 0.78  |
| Precision | 0.69  |
| Recall    | 0.63  |
| F1 Score  | 0.66  |

*Results may vary depending on preprocessing and train-test split.*

---

# 8. Prompt Engineering Strategy

The Talk-to-Data system is schema-aware.

The prompt includes:

* Database schema
* Table names
* Column names
* User question
* SQL generation constraints

### Example

```text
Schema:
application_train

Columns:
TARGET
AMT_INCOME_TOTAL
AMT_CREDIT
CODE_GENDER

User Question:
Show average credit amount by gender.
```
## Example Questions

The Talk-to-Data module allows business users to query the dataset using natural language without writing SQL.

### Credit Risk Analysis

- How many customers defaulted?
- What is the default rate?
- Average income of defaulters?
- Which occupations have the highest default rate?
- Do males or females have a higher default rate?

### Customer & Income Analysis

- Show top 10 occupations by average income.
- What is the average credit amount by gender?
- What is the average income by occupation?
- Which customer segment has the highest average income?

### Loan Analysis

- What is the average loan amount?
- Which occupations receive the highest credit amounts?
- What is the average loan-to-income ratio?

### Sample Query Flow

**User Question**

```text
Which occupations have the highest default rate?

```
### SQL Constraints

The model is instructed to:

* Generate only SELECT queries
* Avoid DDL/DML operations
* Use valid schema columns
* Return concise SQL

---

# 9. Token Optimization Approach

To reduce token usage and improve response quality:

### Techniques

* Include only relevant tables
* Include only required columns
* Use compact schema descriptions
* Reuse prompt templates
* Limit SQL generation context

Benefits:

* Lower latency
* Reduced cost
* More accurate SQL generation

---

# 10. Rule Derivation Logic

Business rules are generated using:

* Feature importance analysis
* SHAP explanations
* Risk segmentation thresholds

### Example Rule

```text
IF Income < 150000
AND Credit Amount > 500000

THEN
High Risk Applicant
```

### Example Output

```json
{
  "customer_id": 101,
  "risk_probability": 0.82,
  "risk_category": "High",
  "top_factors": [
    "Low Income",
    "High Credit Amount",
    "Previous Payment Delays"
  ]
}
```

---

# 11. Setup Instructions

## Clone Repository

```bash
git clone <repository_url>
cd credit_risk_platform
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 12. Running the Project

## Train Model

```bash
python src/ml/train.py
```

## Evaluate Model

```bash
python src/ml/evaluate.py
```

## Generate Predictions

```bash
python src/ml/predict.py
```

---

## Run Streamlit

```bash
streamlit run src/ui/app.py
```

Access:

```text
http://localhost:8501
```

---
# 13. Docker Setup

## Build Image

```bash
docker build -t credit-risk-platform .
```

## Run Container

```bash
docker run -p 8000:8000 credit-risk-platform
```

## Docker Compose

```bash
docker-compose up --build
```

---

# 14. Sample Outputs

### Credit Risk Prediction

```text
Customer ID: 1001

Default Probability: 82%

Risk Category: High
```

### Natural Language Query

Input:

```text
Show average credit amount by occupation.
```

Generated SQL:

```sql
SELECT OCCUPATION_TYPE,
AVG(AMT_CREDIT)
FROM application_train
GROUP BY OCCUPATION_TYPE;
```

---

# 15. Known Limitations

* Limited to available dataset features.
* Performance depends on data quality.
* NL-to-SQL may generate imperfect queries for ambiguous questions.
* Business rules are heuristic and not regulatory recommendations.
* Model retraining is required for new data distributions.

---

# 16. Future Improvements

* Real-time prediction API.
* RAG-based business rule explanation.
* Multi-table NL-to-SQL support.
* Model monitoring and drift detection.
* Automated retraining pipeline.
* Advanced explainability dashboard.

---

# 17. Conclusion

This project demonstrates an end-to-end credit risk assessment platform combining machine learning, explainable AI, and natural language analytics to support data-driven lending decisions.
