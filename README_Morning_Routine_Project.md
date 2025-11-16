# 🌅 Morning Routine Productivity Analysis  
### _Machine Learning Project with From-Scratch Models_

This repository contains a complete machine learning pipeline to analyze **morning routines** (sleep, exercise, meditation, breakfast, journaling, etc.) and predict **daily productivity scores**.  
All core ML models in this project are implemented **entirely from scratch**, without using any built-in algorithms from Scikit-Learn.

---

# 📌 Project Overview  
This project explores how lifestyle habits affect productivity using a real-world dataset.  
It includes:

- ✔ Data Cleaning & Preprocessing  
- ✔ Exploratory Data Analysis (EDA)  
- ✔ ML Models Fully Implemented From Scratch:
  - **Linear Regression (Gradient Descent)**
  - **KNN with Manhattan Distance**
  - **Optimized Random Forest Regressor**
- ✔ Model Evaluation (MAE, MSE, RMSE, R²)
- ✔ Prediction Visualizations  
- ✔ Exported Models & Preprocessor (for deployment)
- ✔ Flask App (`app.py`) for frontend/API integration  
- ✔ Full Project Report (PDF)

---

# 📂 Repository Structure

```
📦 Morning_Routine_Productivity_Analysis
│
├── Morning_Routine_Productivity_Dataset.csv      # Dataset
├── ml_morning.ipynb                              # Main notebook (analysis + models)
├── app.py                                        # Flask app for deployment
├── preprocessor.pkl                              # Saved preprocessor
├── knn_model.pkl                                 # Saved KNN (scratch) model
├── rf_model.pkl                                  # Saved Random Forest (scratch) model
├── report.pdf                                    # Final project report
├── PROPOSALML.pdf                                # Initial proposal
└── README.md                                     # You are here
```

---

# 📊 Dataset Summary

The dataset includes key features like:

- **Sleep Hours**
- **Wake-up Time**
- **Screen Time**
- **Exercise (Yes/No)**
- **Breakfast Type**
- **Water Intake**
- **Mood**
- **Day Type**

🎯 **Target Variable:** Productivity Score (1–10)

---

# 🔍 Exploratory Data Analysis (Highlights)

### ✔ Sleep Hours  
Students sleeping **6.5–8.5 hours** showed the highest productivity.

### ✔ Exercise  
Regular exercisers consistently scored higher.

### ✔ Breakfast  
Healthy breakfast choices (protein/fruit) resulted in better productivity.

### ✔ Screen Time  
Productivity dropped significantly after **3+ hours** of morning screen exposure.

### ✔ Mood  
Positive morning mood strongly correlated with higher performance.

---

# 🤖 ML Models (From Scratch)

## **1️⃣ Linear Regression (Gradient Descent)**
- Custom gradient descent optimizer  
- Manual bias term  
- Mean Squared Error used as cost function  
- Works well for linear relationships  

---

## **2️⃣ KNN Regressor (Manhattan Distance)**
- Custom Manhattan distance implementation  
- No sklearn KNN used  
- Predicts using average of K nearest neighbors  
- Captures local patterns efficiently  

---

## **3️⃣ Random Forest Regressor (Optimized Scratch Version)**
- Custom Decision Trees  
- Feature subsampling (`sqrt(d)` rule)  
- Quantile-based threshold selection  
- Bootstrap sampling  
- Best model for non-linear patterns and interactions  

---

# 📈 Model Performance Summary

| Model | R² Score | Notes |
|------|----------|-------|
| **Random Forest (Scratch)** | ⭐ Highest | Best overall performance |
| **KNN (Manhattan)** | Good | Captures local trends |
| **Linear Regression (GD)** | Moderate | Limited for non-linear data |

---

# ⚙️ How to Run

## 1️⃣ Install dependencies
```
pip install numpy pandas scikit-learn flask matplotlib
```

## 2️⃣ Run Notebook
```
jupyter notebook ml_morning.ipynb
```

## 3️⃣ Run Flask App (optional)
```
python app.py
```

---

# 🧪 Example Prediction Input (for API)

```json
{
  "Sleep_Hours": 7,
  "Screen_Time": 2,
  "Exercise": "Yes",
  "Breakfast": "Healthy",
  "Mood": "Positive",
  "Water_Intake": 2
}
```

---

# 🧑‍🤝‍🧑 Team Collaboration

The team collaborated on:
- EDA & visualizations  
- ML model implementation from scratch  
- Debugging & optimization  
- Deployment & documentation  

GitHub helped manage code updates and merge workflows efficiently.

---

# 🤖 Role of AI Tools

AI tools (ChatGPT, Copilot) were used for:
- Debugging  
- Report formatting  
- Improving documentation structure  

All ML models themselves were written manually to maintain academic originality.

---

# 📄 License  
This project is open for academic and learning purposes.  
Feel free to fork, modify, or extend with attribution.

---

# ⭐ Acknowledgments  
Thanks to the course instructor and team members for support and collaboration.
