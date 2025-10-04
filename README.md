# Voice Gender Recognition with XGBoost

This repository contains an implementation of **gender classification** based on voice features using the [Voice Gender Kaggle Dataset](https://www.kaggle.com/datasets/primaryobjects/voicegender).  
The goal is to classify whether a voice belongs to **male** or **female** using machine learning.

---

## 📂 Dataset
- Source: [Kaggle - Voice Gender Dataset](https://www.kaggle.com/datasets/primaryobjects/voicegender)  
- Samples: 3168 voice recordings  
- Features: Acoustic parameters such as `meanfreq`, `sd`, `median`, `Q25`, `Q75`, `IQR`, `sp.ent`, `meanfun`, etc.  
- Target: `label` (male/female)

---

## 🛠️ Libraries Used
The implementation is based on Python and several popular machine learning libraries:
- **pandas** → data manipulation and loading dataset  
- **numpy** → numerical computation  
- **scikit-learn** → preprocessing, train-test split, and evaluation metrics  
- **xgboost** → main machine learning algorithm (XGBoost Classifier)  
- **matplotlib / seaborn** → visualization (confusion matrix, feature importance)

---

### Classification Report (Test Data)

**Accuracy**: 0.98 (634 samples)

| Label   | Precision | Recall | F1-score | Support |
|---------|-----------|--------|----------|---------|
| Male    | 0.99      | 0.97   | 0.98     | 317     |
| Female  | 0.97      | 0.99   | 0.98     | 317     |
| Macro Avg     | 0.98      | 0.98   | 0.98     | 634     |
| Weighted Avg  | 0.98      | 0.98   | 0.98     | 634     |

### Key Insights
- **Male voices**: Precision 99% → highly accurate when predicting male, Recall 97% → a few male samples were misclassified.  
- **Female voices**: Recall 99% → almost all female voices were correctly identified, Precision 97% → some predictions of female were incorrect.  
- **Overall accuracy: 98%** → the model performs very well in distinguishing male and female voices.  
- **Balanced performance** → no significant class imbalance issues since macro avg and weighted avg are consistently 0.98.  
