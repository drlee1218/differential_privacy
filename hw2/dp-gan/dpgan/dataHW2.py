import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras

# 데이터 전처리 함수
def preprocess_data(df):
    df = df.copy().apply(LabelEncoder().fit_transform)
    X = df.drop(columns=["income"])
    y = df["income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test

# DNN 학습 및 평가 함수
def train_dnn(X_train, X_test, y_train, y_test):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # TensorFlow 2.x에 맞게 optimizer 수정
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.002), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_test, y_test))

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    y_prob = model.predict(X_test).flatten()

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Misclassification Error": 1 - accuracy_score(y_test, y_pred)
    }


# 데이터 불러오기
columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country", "income"]
data = pd.read_csv("C:\\Users\\drlee\\Downloads\\adult\\adult.data", names=columns, sep=", ", engine="python")

# =============== K-Anonymity 처리 ===============
class MondrianKAnonymity:
    def __init__(self, k): self.k = k

    def partition(self, data, quasi_identifiers):
        if len(data) < 2 * self.k: return [data]
        attr = self._max_spread_attribute(data, quasi_identifiers)
        median = np.median(data[attr])
        left = data[data[attr] <= median]
        right = data[data[attr] > median]
        if len(left) < self.k or len(right) < self.k: return [data]
        return self.partition(left, quasi_identifiers) + self.partition(right, quasi_identifiers)

    def _max_spread_attribute(self, data, quasi_identifiers):
        return max(quasi_identifiers, key=lambda attr: data[attr].max() - data[attr].min())

    def anonymize(self, data, quasi_identifiers):
        partitions = self.partition(data, quasi_identifiers)
        result = []
        for p in partitions:
            gen_vals = {q: f"[{p[q].min()}-{p[q].max()}]" for q in quasi_identifiers}
            for _, row in p.iterrows():
                row_copy = row.copy()
                for q in quasi_identifiers:
                    row_copy[q] = gen_vals[q]
                result.append(row_copy)
        return pd.DataFrame(result)

def convert_to_numeric(df, quasi_identifiers):
    for col in quasi_identifiers:
        df[col] = df[col].apply(lambda x: np.mean([float(num) for num in x.strip("[]").split("-")]))
    return df

k = 5
quasi_identifiers = ["age", "education-num", "hours-per-week"]
k_anon = MondrianKAnonymity(k)
k_anon_data = k_anon.anonymize(data, quasi_identifiers)
k_anon_data = convert_to_numeric(k_anon_data, quasi_identifiers)

# =============== 진짜 TensorFlow 기반 DP-GAN ===============
from dpgan import DPGAN  # 이 부분은 반드시 DP-GAN 구현 파일에서 가져올 것 (M. Alzantot 코드 기반)

# DP-GAN 학습 (이 과정은 사전 학습되어 있다고 가정)
df_encoded = data.copy().apply(LabelEncoder().fit_transform)
X_data = df_encoded.drop(columns=["income"])
y_data = df_encoded["income"]

dpgan = DPGAN(input_dim=X_data.shape[1], num_classes=2, batch_size=64)
dpgan.train(X_data.values, y_data.values, epochs=500, epsilon=1.0, delta=1e-5)

synthetic_data = dpgan.sample(10000)
synthetic_df = pd.DataFrame(synthetic_data, columns=X_data.columns)
synthetic_df["income"] = np.random.choice([0, 1], size=(len(synthetic_df),))  # 샘플링된 라벨 (적절히 조절 가능)

# 결과 비교
results = {}

# 1. Original Data
X_train, X_test, y_train, y_test = preprocess_data(data)
results["Original DNN"] = train_dnn(X_train, X_test, y_train, y_test)

# 2. K-Anonymized Data
X_train_k, X_test_k, y_train_k, y_test_k = preprocess_data(k_anon_data)
results["K-Anonymity DNN"] = train_dnn(X_train_k, X_test_k, y_train_k, y_test_k)

# 3. DP-GAN Synthetic Data
synthetic_df = synthetic_df[columns]  # 순서 맞추기
X_train_dp, X_test_dp, y_train_dp, y_test_dp = preprocess_data(synthetic_df)
results["DP-GAN DNN"] = train_dnn(X_train_dp, X_test_dp, y_train_dp, y_test_dp)

# 결과 출력
for model, metrics in results.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
