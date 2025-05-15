# Credit Card Fraud Detection Dashboard

This is a machine learning powered dashboard to detect fraudulent credit card transactions using the Kaggle creditcard.csv dataset.

Developed with XGBoost, SMOTE for handling class imbalance, and Streamlit for interactive visualization.

---

## Features

- Trained fraud detection model using XGBoost
- SMOTE resampling to handle extreme class imbalance
- Precision-Recall evaluation
- Streamlit dashboard to explore flagged transactions
- Slider to filter frauds by amount
- Downloadable CSV of filtered results

---

## Tech Stack

- Python
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- Streamlit
- Pandas, NumPy, Matplotlib

---

## Dataset Required

This project uses the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

**To run the project, you NEED to:**

1. Download `creditcard.csv` from Kaggle.
2. Place it into the `data/` folder inside this repo (you may need to create it).
3. Run the training script to generate fraud predictions:

```bash
cd model
python train_model.py
```

This will generate `flagged_frauds.csv` automatically.

4. Start the Streamlit dashboard:

```bash
cd ../dashboard
streamlit run app.py
```

---

## Engineered by Sai Sanapala

End-to-end ML pipeline + UI built as a practical FinTech data science project.

---

## License

This project is for educational use only. Dataset is property of its original authors on Kaggle.
