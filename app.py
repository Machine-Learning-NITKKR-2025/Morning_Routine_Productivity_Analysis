# app.py - Robust Streamlit app with safe model loading and fallbacks
import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------
# 0) Static artifacts (Linear Regression weights, scaling, categories)
# ---------------------------------------------------------------------
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
    'numerical_features': [
        'Sleep Duration (hrs)', 'Meditation (mins)', 'Exercise (mins)',
        'Wake-up Hour', 'Work Start Hour', 'Time to Start Work (mins)'
    ]
}

OHE_CATEGORIES = {
    'Breakfast Type': ['Carb-rich', 'Heavy', 'Light', 'Protein-rich', 'Skipped'],
    'Journaling (Y/N)': ['No', 'Yes'],
    'Mood': ['Happy', 'Neutral', 'Sad'],
    'Day of Week': ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
}

# ---------------------------------------------------------------------
# 1) Minimal model classes (decision tree / RF / KNN from scratch)
#    These are present so pickles referencing these names can be unpickled.
# ---------------------------------------------------------------------
class DT:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
        self.tree = None
    def _mse(self,y): return 0.0 if len(y)==0 else np.mean((y-np.mean(y))**2)
    def _best(self,X,y):
        n,m=X.shape; best=(None,None,float('inf'))
        for j in range(m):
            for t in np.unique(X[:,j]):
                L=np.where(X[:,j]<=t)[0]; R=np.where(X[:,j]>t)[0]
                if len(L)==0 or len(R)==0: continue
                wm=(len(L)/n)*self._mse(y[L]) + (len(R)/n)*self._mse(y[R])
                if wm<best[2]: best=(j,float(t),wm)
        return best[0], best[1]
    def _build(self,X,y,depth=0):
        if (self.max_depth is not None and depth>=self.max_depth) or len(y)<self.min_samples_split or len(np.unique(y))==1:
            return float(np.mean(y))
        f,t=self._best(X,y)
        if f is None: return float(np.mean(y))
        L=np.where(X[:,f]<=t)[0]; R=np.where(X[:,f]>t)[0]
        return {"f":int(f),"t":float(t),"l":self._build(X[L],y[L],depth+1),"r":self._build(X[R],y[R],depth+1)}
    def fit(self,X,y):
        X=np.asarray(X); X=X.reshape(-1,1) if X.ndim==1 else X
        y=np.asarray(y).flatten(); self.tree=self._build(X,y,0)
    def _p1(self,x,node):
        if not isinstance(node, dict): return float(node)
        return self._p1(x,node["l"]) if x[node["f"]] <= node["t"] else self._p1(x,node["r"])
    def predict(self,X):
        X=np.asarray(X); X=X.reshape(-1,1) if X.ndim==1 else X
        return np.array([self._p1(row,self.tree) for row in X])

class RF:
    def __init__(self,n_estimators=10,max_depth=10,min_samples_split=2,random_state=42):
        self.n_estimators=int(n_estimators); self.max_depth=max_depth
        self.min_samples_split=min_samples_split; self.rs=int(random_state); self.trees=[]
    def fit(self,X,y):
        X=np.asarray(X); X=X.reshape(-1,1) if X.ndim==1 else X
        y=np.asarray(y).flatten(); n=X.shape[0]; rng=np.random.RandomState(self.rs); self.trees=[]
        for _ in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); t=DT(max_depth=self.max_depth,min_samples_split=self.min_samples_split)
            t.fit(X[idx], y[idx]); self.trees.append(t)
    def predict(self,X):
        X=np.asarray(X); X=X.reshape(-1,1) if X.ndim==1 else X
        preds=np.array([t.predict(X) for t in self.trees]); return preds.mean(axis=0)

class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = int(n_neighbors)
        self.X_train = None
        self.y_train = None
    def fit(self, X, y):
        self.X_train = np.asarray(X)
        if self.X_train.ndim == 1: self.X_train = self.X_train.reshape(-1,1)
        self.y_train = np.asarray(y).flatten()
    def predict(self, X_test):
        X_test = np.asarray(X_test)
        if X_test.ndim == 1: X_test = X_test.reshape(-1,1)
        dists = np.sqrt(((X_test[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2))
        idx = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        return np.mean(self.y_train[idx], axis=1)

# Keep aliases so pickles that reference names RF/KNN succeed
RandomForestRegressorFromScratch = RF
KNNRegressorFromScratch = KNN

# ---------------------------------------------------------------------
# 2) Robust loader: try joblib, pickle, shim (map __main__.RF -> RF), provide diagnostics
# ---------------------------------------------------------------------
from joblib import load as joblib_load, dump as joblib_dump  # joblib available in most ML envs

TRAINED_MODEL_RF = None
TRAINED_MODEL_KNN = None
PREPROCESSOR = None

def try_joblib(path):
    try:
        return joblib_load(path), "joblib"
    except Exception as e:
        return None, e

def try_pickle(path, encoding=None):
    try:
        if encoding is None:
            with open(path,'rb') as f:
                return pickle.load(f), "pickle"
        else:
            with open(path,'rb') as f:
                return pickle.load(f, encoding=encoding), "pickle-encoding"
    except Exception as e:
        return None, e

def try_shim_pickle(path):
    # map __main__.RF -> local RF class, also map 'RF' name if needed
    try:
        class ShimUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if (module == "__main__" and name == "RF") or name == "RF":
                    return RF
                if (module == "__main__" and name == "KNN") or name == "KNN":
                    return KNN
                return super().find_class(module, name)
        with open(path, 'rb') as f:
            obj = ShimUnpickler(f).load()
        return obj, "shim-pickle"
    except Exception as e:
        return None, e

# diagnostic attempt
cwd = os.getcwd()

def load_artifact(name_base):
    """
    Try joblib -> pickle -> shim-pickle for a given base name (no extension).
    Returns (obj, loader_name, error_dict)
    """
    errors = {}
    # joblib
    jb = f"{name_base}.joblib"
    pkl = f"{name_base}.pkl"
    if os.path.exists(jb):
        obj, info = try_joblib(jb)
        if obj is not None: return obj, info, errors
        errors['joblib'] = info
    # normal pickle
    if os.path.exists(pkl):
        obj, info = try_pickle(pkl)
        if obj is not None: return obj, info, errors
        errors['pickle'] = info
        # try latin1 encoding (py2 -> py3 pickles)
        obj, info = try_pickle(pkl, encoding='latin1')
        if obj is not None: return obj, info, errors
        errors['pickle-latin1'] = info
        # try shim
        obj, info = try_shim_pickle(pkl)
        if obj is not None: return obj, info, errors
        errors['shim'] = info
    # not found
    errors['found'] = f"None of {jb} or {pkl} present"
    return None, None, errors

# load RF
rf_obj, rf_loader, rf_errors = load_artifact("rf_model")
if rf_obj is not None:
    TRAINED_MODEL_RF = rf_obj

# load KNN
knn_obj, knn_loader, knn_errors = load_artifact("knn_model")
if knn_obj is not None:
    TRAINED_MODEL_KNN = knn_obj

# load preprocessor
pre_obj, pre_loader, pre_errors = load_artifact("preprocessor")
if pre_obj is not None:
    PREPROCESSOR = pre_obj

# Fallback: if RF failed to load, optionally create a tiny in-memory RF trained on random data
# (useful for demo; adjust n_features to expected fed to your model)
def make_demo_rf(n_features=23):
    demo = RF(n_estimators=7, max_depth=5, random_state=42)
    Xd = np.random.randn(60, n_features)
    yd = np.random.uniform(1.0, 10.0, size=(60,))
    demo.fit(Xd, yd)
    return demo

# If desired, enable fallback by uncommenting next lines.
# if TRAINED_MODEL_RF is None:
#     TRAINED_MODEL_RF = make_demo_rf(n_features=23)

# ---------------------------------------------------------------------
# 3) Preprocessing helpers & prediction wrappers
# ---------------------------------------------------------------------
def time_to_hour(time_str):
    try:
        dt = datetime.strptime(str(time_str).strip(), "%I:%M %p")
        return dt.hour + dt.minute / 60.0
    except Exception:
        return np.nan

def calculate_time_diff(wake_time_str, work_start_time_str):
    dt_today = datetime.now().date()
    try:
        wake_dt = datetime.strptime(str(wake_time_str).strip(), "%I:%M %p").replace(year=dt_today.year, month=dt_today.month, day=dt_today.day)
        work_dt = datetime.strptime(str(work_start_time_str).strip(), "%I:%M %p").replace(year=dt_today.year, month=dt_today.month, day=dt_today.day)
    except Exception:
        return np.nan
    diff = (work_dt - wake_dt).total_seconds() / 60.0
    if diff < 0:
        diff += 24*60
    return diff

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
        cats = OHE_CATEGORIES[feature_name]
        return [1.0 if v == value else 0.0 for v in cats]
    return []

def preprocess_input(input_dict):
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
    scaled_num = [scale_numerical(num_values[i], SCALING_INFO['numerical_features'][i]) for i in range(len(num_values))]

    cats = []
    cats += encode_categorical(input_dict['Breakfast Type'], 'Breakfast Type')
    cats += encode_categorical(input_dict['Journaling (Y/N)'], 'Journaling (Y/N)')
    cats += encode_categorical(input_dict['Mood'], 'Mood')
    cats += encode_categorical(input_dict['Day of Week'], 'Day of Week')

    feature_vector = [1.0] + scaled_num + cats
    return np.array(feature_vector, dtype=float)

def _transform_with_preprocessor(raw_df):
    """Use PREPROCESSOR to transform raw_df; attempt to reindex if feature_names_in_ exists."""
    if PREPROCESSOR is None:
        raise RuntimeError("No PREPROCESSOR loaded.")
    try:
        X_trans = PREPROCESSOR.transform(raw_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        return np.asarray(X_trans)
    except Exception as e:
        # fallback: try to reindex raw_df to preprocessor.feature_names_in_ if available
        try:
            if hasattr(PREPROCESSOR, "feature_names_in_"):
                expected = list(PREPROCESSOR.feature_names_in_)
                reidx = raw_df.reindex(columns=expected, fill_value=np.nan)
                X_trans = PREPROCESSOR.transform(reidx)
                if hasattr(X_trans, "toarray"):
                    X_trans = X_trans.toarray()
                return np.asarray(X_trans)
        except Exception:
            pass
        raise

def predict_lr(feature_vector):
    fv = np.asarray(feature_vector).flatten()
    if fv.size != FINAL_WEIGHTS_LR.size:
        # fallback to clip of mean
        return float(np.clip(np.mean(fv) if fv.size>0 else 5.0, 1.0, 10.0))
    pred = np.dot(fv, FINAL_WEIGHTS_LR)
    return float(np.clip(pred, 1.0, 10.0))

def predict_rf(feature_vector, raw_input=None):
    if TRAINED_MODEL_RF is None:
        return "RF model not loaded."
    # prepare X: use preprocessor if available and raw_input is provided
    if PREPROCESSOR is not None and raw_input is not None:
        raw_df = build_raw_dataframe(raw_input)
        X = _transform_with_preprocessor(raw_df)
    else:
        fv = np.asarray(feature_vector).flatten()
        X = fv[1:].reshape(1, -1)
    pred = TRAINED_MODEL_RF.predict(X)
    return float(np.clip(np.asarray(pred).flatten()[0], 1.0, 10.0))

def predict_knn(feature_vector, raw_input=None):
    if TRAINED_MODEL_KNN is None:
        return "KNN model not loaded."
    if PREPROCESSOR is not None and raw_input is not None:
        raw_df = build_raw_dataframe(raw_input)
        X = _transform_with_preprocessor(raw_df)
    else:
        fv = np.asarray(feature_vector).flatten()
        X = fv[1:].reshape(1, -1)
    pred = TRAINED_MODEL_KNN.predict(X)
    return float(np.clip(np.asarray(pred).flatten()[0], 1.0, 10.0))

def build_raw_dataframe(input_dict):
    num_names = SCALING_INFO['numerical_features']
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
    raw['Wake-up Time'] = input_dict.get('Wake-up Time', np.nan)
    raw['Work Start Time'] = input_dict.get('Work Start Time', np.nan)
    raw['Breakfast Type'] = input_dict['Breakfast Type']
    raw['Journaling (Y/N)'] = input_dict['Journaling (Y/N)']
    raw['Mood'] = input_dict['Mood']
    raw['Day of Week'] = input_dict['Day of Week']
    cols_in_order = list(num_names) + ['Wake-up Time', 'Work Start Time', 'Breakfast Type', 'Journaling (Y/N)', 'Mood', 'Day of Week']
    return pd.DataFrame([{col: raw.get(col, np.nan) for col in cols_in_order}])

# ---------------------------------------------------------------------
# 4) Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Productivity Model Selector", layout="wide")
st.title("☀️ Morning Routine Productivity Predictor")

# sidebar diagnostics
with st.sidebar.expander("Diagnostics", expanded=True):
    st.write("Working directory:", cwd)
    try:
        st.write("Files:", os.listdir(cwd))
    except Exception as e:
        st.write("Could not list files:", e)
    st.write("RF loaded:", TRAINED_MODEL_RF is not None, "via:", rf_loader if 'rf_loader' in locals() else None)
    if TRAINED_MODEL_RF is None:
        st.write("RF errors:", rf_errors if 'rf_errors' in locals() else None)
    st.write("KNN loaded:", TRAINED_MODEL_KNN is not None, "via:", knn_loader if 'knn_loader' in locals() else None)
    if TRAINED_MODEL_KNN is None:
        st.write("KNN errors:", knn_errors if 'knn_errors' in locals() else None)
    st.write("Preprocessor loaded:", PREPROCESSOR is not None, "via:", pre_loader if 'pre_loader' in locals() else None)
    if PREPROCESSOR is None:
        st.write("Preprocessor errors:", pre_errors if 'pre_errors' in locals() else None)

st.markdown("Enter your morning routine below and select a model. LR always works (built-in). RF/KNN will be used only if model files load successfully.")

col1, col2 = st.columns(2)
with col1:
    st.header("Routine Timing & Duration")
    sleep_duration = st.slider("Sleep Duration (hrs)", 5.0, 9.0, 7.5, 0.1)
    meditation_mins = st.slider("Meditation (mins)", 0, 30, 10, 1)
    exercise_mins = st.slider("Exercise (mins)", 0, 60, 20, 1)
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

model_choice = st.selectbox("Choose model", ["Linear Regression (LR)", "Random Forest (RF)", "K-Nearest Neighbors (KNN)"], index=0)

if st.button("Predict"):
    fv = preprocess_input(input_data)
    if model_choice == "Linear Regression (LR)":
        score = predict_lr(fv)
        st.success(f"LR Prediction: {score:.2f} / 10")
    elif model_choice == "Random Forest (RF)":
        if TRAINED_MODEL_RF is None:
            st.warning("RF not loaded — falling back to LR.")
            score = predict_lr(fv)
            st.success(f"Fallback LR Prediction: {score:.2f} / 10")
        else:
            try:
                score = predict_rf(fv, input_data)
                if isinstance(score, str):
                    st.warning(score)
                else:
                    st.success(f"RF Prediction: {score:.2f} / 10")
            except Exception as e:
                st.error(f"RF prediction error: {e}")
    else:  # KNN
        if TRAINED_MODEL_KNN is None:
            st.warning("KNN not loaded — falling back to LR.")
            score = predict_lr(fv)
            st.success(f"Fallback LR Prediction: {score:.2f} / 10")
        else:
            try:
                score = predict_knn(fv, input_data)
                if isinstance(score, str):
                    st.warning(score)
                else:
                    st.success(f"KNN Prediction: {score:.2f} / 10")
            except Exception as e:
                st.error(f"KNN prediction error: {e}")

st.markdown("---")
st.caption("If RF/KNN fail to load, re-save the trained model from your notebook using: `from joblib import dump; dump(rf, 'rf_model.joblib')` and copy into this app folder.")
