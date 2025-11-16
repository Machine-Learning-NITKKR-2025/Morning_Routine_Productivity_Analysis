# app.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import sys

# ===============================================================
# 1. CUSTOM MODEL CLASS DEFINITIONS (REQUIRED FOR PICKLE LOADING)
# ===============================================================
class DecisionTreeRegressorFromScratch:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.tree = None

    def _mean_squared_error(self, y):
        if len(y) == 0:
            return 0.0
        m = np.mean(y)
        return np.mean((y - m) ** 2)

    def _best_split(self, X, y):
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        n_samples, n_features = X.shape
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = np.where(X[:, feature] <= t)[0]
                right = np.where(X[:, feature] > t)[0]
                if len(left) == 0 or len(right) == 0:
                    continue
                y_left, y_right = y[left], y[right]
                mse_left = self._mean_squared_error(y_left)
                mse_right = self._mean_squared_error(y_right)
                weighted_mse = (len(left) / n_samples) * mse_left + (len(right) / n_samples) * mse_right
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = int(feature)
                    best_threshold = float(t)
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split or \
           len(np.unique(y)) == 1:
            return float(np.mean(y))
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return float(np.mean(y))
        left_idx = np.where(X[:, feature] <= threshold)[0]
        right_idx = np.where(X[:, feature] > threshold)[0]
        left_tree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return {"feature": feature, "threshold": threshold, "left": left_tree, "right": right_tree}

    def fit(self, X, y):
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y).flatten()
        self.tree = self._build_tree(X_arr, y_arr, 0)

    def _predict_single(self, x, node):
        if not isinstance(node, dict):
            return float(node)
        if x[int(node["feature"])] <= float(node["threshold"]):
            return self._predict_single(x, node["left"])
        else:
            return self._predict_single(x, node["right"])

    def predict(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return np.array([self._predict_single(row, self.tree) for row in X_arr])


class RandomForestRegressorFromScratch:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, random_state=42):
        self.n_estimators = int(n_estimators)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.random_state = int(random_state)
        self.trees = []

    def fit(self, X, y):
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y).flatten()
        np.random.seed(self.random_state)
        n_samples = X_arr.shape[0]
        self.trees = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X_arr[idx], y_arr[idx]
            tree = DecisionTreeRegressorFromScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        if not self.trees:
            raise RuntimeError("Model must be fitted before calling predict.")
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        preds = np.array([tree.predict(X_arr) for tree in self.trees])
        return np.mean(preds, axis=0)


class KNNRegressorFromScratch:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = int(n_neighbors)
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        if self.X_train.ndim == 1:
            self.X_train = self.X_train.reshape(-1, 1)
        self.y_train = np.asarray(y).flatten()

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        dists = np.sqrt(((X_test[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2))
        idx = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        return np.mean(self.y_train[idx], axis=1)


# ===============================================================
# 2. MODEL ARTIFACTS AND LOADING (ACTIVE LOADING)
# ===============================================================

FINAL_WEIGHTS_LR = np.array([
    5.64539169e+00,  8.00693510e-01, -1.89069677e-02,  2.01570533e-02,
    7.00932269e-02, -8.32415178e-02,  5.71759081e-02,  1.17835106e-01,
    -1.25892557e-01,  1.20968940e-01, -1.45524278e-01, -1.45164878e-01,
    -1.15787834e-01, -1.16453073e-01,  1.12788574e-01,  2.16460114e-02,
    -5.87708577e-02, -1.15682281e-01,  9.78906666e-02, -3.98902891e-02,
    1.12450702e-02, -2.13110257e-02,  2.06734107e-02,  1.02871130e-02
])

SCALING_INFO = {
    'mean': [6.9784, 14.7111, 29.7333, 5.9221, 10.0381, 243.6826],
    'scale': [1.1498, 9.9372, 20.6225, 0.9991, 1.3402, 70.8143],
    'numerical_features': ['Sleep Duration (hrs)', 'Meditation (mins)', 'Exercise (mins)', 'Wake-up Hour', 'Work Start Hour', 'Time to Start Work (mins)']
}

OHE_CATEGORIES = {
    'Breakfast Type': ['Carb-rich', 'Heavy', 'Light', 'Protein-rich', 'Skipped'],
    'Journaling (Y/N)': ['No', 'Yes'],
    'Mood': ['Happy', 'Neutral', 'Sad'],
    'Day of Week': ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
}

TRAINED_MODEL_RF = None
TRAINED_MODEL_KNN = None

try:
    with open('rf_model.pkl', 'rb') as f:
        TRAINED_MODEL_RF = pickle.load(f)
    st.sidebar.success("rf_model.pkl loaded.")
except Exception as e:
    st.sidebar.error(f"❌ RF Load Failed: {e}")

try:
    with open('knn_model.pkl', 'rb') as f:
        TRAINED_MODEL_KNN = pickle.load(f)
    st.sidebar.success("knn_model.pkl loaded.")
except Exception as e:
    st.sidebar.error(f"❌ KNN Load Failed: {e}")

# Load preprocessor saved from your notebook
PREPROCESSOR = None
try:
    with open("preprocessor.pkl", "rb") as f:
        PREPROCESSOR = pickle.load(f)
    st.sidebar.success("preprocessor.pkl loaded.")
except FileNotFoundError:
    PREPROCESSOR = None
    st.sidebar.warning("preprocessor.pkl not found — RF/KNN may not work without it.")
except Exception as e:
    PREPROCESSOR = None
    st.sidebar.error(f"Failed to load preprocessor.pkl: {e}")


# ===============================================================
# 3. PREPROCESSING AND PREDICTION FUNCTIONS
# ===============================================================

def time_to_hour(time_str):
    try:
        dt_obj = datetime.strptime(time_str, '%I:%M %p')
        return dt_obj.hour + dt_obj.minute / 60
    except Exception:
        return np.nan

def calculate_time_diff(wake_time_str, work_start_time_str):
    dt_today = datetime.now().date()
    try:
        wake_dt = datetime.strptime(wake_time_str, '%I:%M %p').replace(year=dt_today.year, month=dt_today.month, day=dt_today.day)
        work_dt = datetime.strptime(work_start_time_str, '%I:%M %p').replace(year=dt_today.year, month=dt_today.month, day=dt_today.day)
    except Exception:
        return np.nan
    time_diff = work_dt - wake_dt
    if time_diff.total_seconds() < 0:
        time_diff += pd.Timedelta(days=1)
    return time_diff.total_seconds() / 60

def scale_numerical(value, feature_name):
    try:
        idx = SCALING_INFO['numerical_features'].index(feature_name)
        mean = SCALING_INFO['mean'][idx]
        scale = SCALING_INFO['scale'][idx]
        return (value - mean) / scale
    except Exception:
        return np.nan

def encode_categorical(value, feature_name):
    if feature_name in OHE_CATEGORIES:
        categories = OHE_CATEGORIES[feature_name]
        return [1.0 if cat == value else 0.0 for cat in categories]
    return []

def preprocess_input(input_dict):
    """Transforms raw input into the final 24-feature vector (bias + 23 features)."""
    wake_hour = time_to_hour(input_dict['Wake-up Time'])
    work_start_hour = time_to_hour(input_dict['Work Start Time'])
    time_to_start_work = calculate_time_diff(input_dict['Wake-up Time'], input_dict['Work Start Time'])

    num_values = [
        input_dict['Sleep Duration (hrs)'],
        input_dict['Meditation (mins)'],
        input_dict['Exercise (mins)'],
        wake_hour,
        work_start_hour,
        time_to_start_work
    ]
    scaled_num_features = [scale_numerical(num_values[i], SCALING_INFO['numerical_features'][i]) for i in range(len(num_values))]

    encoded_cat_features = []
    encoded_cat_features.extend(encode_categorical(input_dict['Breakfast Type'], 'Breakfast Type'))
    encoded_cat_features.extend(encode_categorical(input_dict['Journaling (Y/N)'], 'Journaling (Y/N)'))
    encoded_cat_features.extend(encode_categorical(input_dict['Mood'], 'Mood'))
    encoded_cat_features.extend(encode_categorical(input_dict['Day of Week'], 'Day of Week'))

    feature_vector = [1.0] + scaled_num_features + encoded_cat_features
    return np.array(feature_vector, dtype=float)

# Updated build_raw_dataframe: include both raw time strings and numeric derived columns
def build_raw_dataframe(input_dict, debug=False):
    """
    Build a 1-row DataFrame with:
      - numeric features used in SCALING_INFO,
      - original raw time-string columns ('Wake-up Time','Work Start Time'),
      - categorical columns expected by preprocessor.
    This maximizes compatibility with preprocessors saved from different pipelines.
    """
    num_names = SCALING_INFO['numerical_features']  # e.g. ['Sleep Duration (hrs)', ..., 'Time to Start Work (mins)']

    # Derived numeric values
    wake_hour = time_to_hour(input_dict['Wake-up Time'])
    work_start_hour = time_to_hour(input_dict['Work Start Time'])
    time_to_start_work = calculate_time_diff(input_dict['Wake-up Time'], input_dict['Work Start Time'])

    raw = {}
    raw[num_names[0]] = input_dict['Sleep Duration (hrs)']
    raw[num_names[1]] = input_dict['Meditation (mins)']
    raw[num_names[2]] = input_dict['Exercise (mins)']
    raw[num_names[3]] = wake_hour
    raw[num_names[4]] = work_start_hour
    raw[num_names[5]] = time_to_start_work

    # Also include original time strings (some preprocessors expect them)
    raw['Wake-up Time'] = input_dict.get('Wake-up Time', np.nan)
    raw['Work Start Time'] = input_dict.get('Work Start Time', np.nan)

    # categorical raw values
    raw['Breakfast Type'] = input_dict['Breakfast Type']
    raw['Journaling (Y/N)'] = input_dict['Journaling (Y/N)']
    raw['Mood'] = input_dict['Mood']
    raw['Day of Week'] = input_dict['Day of Week']

    # Column order: numeric features, raw time strings, categoricals
    cols_in_order = list(num_names) + ['Wake-up Time', 'Work Start Time', 'Breakfast Type', 'Journaling (Y/N)', 'Mood', 'Day of Week']
    df = pd.DataFrame([{col: raw.get(col, np.nan) for col in cols_in_order}])

    if debug and PREPROCESSOR is not None:
        try:
            # best-effort: show what PREPROCESSOR exposes about feature names
            if hasattr(PREPROCESSOR, "feature_names_in_"):
                st.sidebar.write("PREPROCESSOR.feature_names_in_:", PREPROCESSOR.feature_names_in_)
            elif hasattr(PREPROCESSOR, "get_feature_names_out"):
                try:
                    st.sidebar.write("PREPROCESSOR.get_feature_names_out():", PREPROCESSOR.get_feature_names_out())
                except Exception:
                    st.sidebar.write("PREPROCESSOR.get_feature_names_out() not available with no input")
            else:
                st.sidebar.write("PREPROCESSOR transformers_:", getattr(PREPROCESSOR, "transformers_", None))
            st.sidebar.write("Raw df columns:", list(df.columns))
        except Exception as e:
            st.sidebar.write("Preprocessor debug failed:", e)

    return df

def predict_lr(feature_vector):
    fv = np.asarray(feature_vector).flatten()
    pred = np.dot(fv, FINAL_WEIGHTS_LR)
    return float(np.clip(pred, 1.0, 10.0))

def predict_rf(feature_vector, raw_input_dict=None):
    if TRAINED_MODEL_RF is None:
        return "Model Unavailable: Load TRAINED_MODEL_RF object."
    if PREPROCESSOR is not None and raw_input_dict is not None:
        raw_df = build_raw_dataframe(raw_input_dict)
        X_trans = PREPROCESSOR.transform(raw_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X = np.asarray(X_trans)
    else:
        fv = np.asarray(feature_vector).flatten()
        X = fv[1:].reshape(1, -1) if fv.size > 1 else fv.reshape(1, -1)
    try:
        pred = TRAINED_MODEL_RF.predict(X)
    except Exception as e:
        return f"Prediction failed (RF). Error: {e}; X.shape={getattr(X, 'shape', None)}"
    return float(np.clip(np.asarray(pred).flatten()[0], 1.0, 10.0))

def predict_knn(feature_vector, raw_input_dict=None):
    if TRAINED_MODEL_KNN is None:
        return "Model Unavailable: Load TRAINED_MODEL_KNN object."
    if PREPROCESSOR is not None and raw_input_dict is not None:
        raw_df = build_raw_dataframe(raw_input_dict)
        X_trans = PREPROCESSOR.transform(raw_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X = np.asarray(X_trans)
    else:
        fv = np.asarray(feature_vector).flatten()
        X = fv[1:].reshape(1, -1) if fv.size > 1 else fv.reshape(1, -1)
    try:
        pred = TRAINED_MODEL_KNN.predict(X)
    except Exception as e:
        return f"Prediction failed (KNN). Error: {e}; X.shape={getattr(X, 'shape', None)}"
    return float(np.clip(np.asarray(pred).flatten()[0], 1.0, 10.0))


# ===============================================================
# 4. STREAMLIT UI
# ===============================================================
st.set_page_config(page_title="Productivity Model Selector", layout="wide")

st.title("☀️ Morning Routine Productivity Predictor")
st.markdown("Select a model and enter your metrics to predict your Productivity Score (1-10).")

# --- Model Selection ---
st.sidebar.header("Model Selection")
model_options = {
    "Linear Regression (LR)": predict_lr,
    "Random Forest Regressor (RF)": predict_rf,
    "K-Nearest Neighbors (KNN)": predict_knn
}
model_name = st.sidebar.selectbox("Choose a Model:", list(model_options.keys()), index=0)
prediction_function = model_options[model_name]

# --- Input Fields ---
col1, col2 = st.columns(2)
with col1:
    st.header("Routine Timing & Duration")
    sleep_duration = st.slider("Sleep Duration (hrs)", min_value=5.0, max_value=9.0, value=7.5, step=0.1)
    meditation_mins = st.slider("Meditation (mins)", min_value=0, max_value=30, value=10, step=5)
    exercise_mins = st.slider("Exercise (mins)", min_value=0, max_value=60, value=20, step=5)
    wake_up_time = st.text_input("Wake-up Time (e.g., 6:30 AM)", "6:30 AM")
    work_start_time = st.text_input("Work Start Time (e.g., 9:00 AM)", "9:00 AM")

with col2:
    st.header("Categorical Factors")
    breakfast_type = st.selectbox("Breakfast Type", OHE_CATEGORIES['Breakfast Type'], index=1)
    journaling = st.radio("Journaling (Y/N)", OHE_CATEGORIES['Journaling (Y/N)'], index=1)
    mood = st.selectbox("Mood", OHE_CATEGORIES['Mood'], index=0)
    day_of_week = st.selectbox("Day of Week", OHE_CATEGORIES['Day of Week'], index=1)

input_data = {
    'Sleep Duration (hrs)': sleep_duration,
    'Meditation (mins)': meditation_mins,
    'Exercise (mins)': exercise_mins,
    'Wake-up Time': wake_up_time,
    'Work Start Time': work_start_time,
    'Breakfast Type': breakfast_type,
    'Journaling (Y/N)': journaling,
    'Mood': mood,
    'Day of Week': day_of_week
}

# Optional debug toggle to display preprocessor expectations in sidebar
debug_preprocessor = st.sidebar.checkbox("Show preprocessor debug info", value=False)

if st.button("PREDICT PRODUCTIVITY SCORE", type="primary"):
    try:
        feature_vector = preprocess_input(input_data)
        if model_name == "Linear Regression (LR)":
            predicted_score = prediction_function(feature_vector)
        else:
            # pass raw input so RF/KNN can use PREPROCESSOR to make exact shape
            if debug_preprocessor:
                raw_df = build_raw_dataframe(input_data, debug=True)
            predicted_score = prediction_function(feature_vector, input_data)

        st.markdown("---")
        st.subheader(f"Prediction from {model_name}:")
        if isinstance(predicted_score, str):
            st.warning(predicted_score)
        else:
            st.success(f"## {predicted_score:.2f} / 10")

    except Exception as e:
        st.error(f"An error occurred during prediction. Please check time formats. Error: {e}")
