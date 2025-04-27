import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ============ Data Preprocessing ============
def preprocess_data(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    X = df.drop(columns=["income"])
    y = df["income"].astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# ============ DNN Model ============
def train_dnn(X_train, X_test, y_train, y_test):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(X_test, y_test))
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

# ============ Mondrian K-Anonymity ============
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
        df[col] = df[col].apply(lambda x: np.mean([float(i) for i in x.strip("[]").split("-")]))
    return df

# ============ Differentially Private GAN ============
class DPGAN(tf.keras.Model):
    def __init__(self, input_dim, latent_dim=100, clip_norm=1.0, noise_multiplier=1.1):
        super(DPGAN, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        return tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.input_dim, activation='sigmoid')
        ])

    def build_discriminator(self):
        return tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])

    def _apply_dp(self, grads):
        clipped_grads = [tf.clip_by_norm(g, self.clip_norm) for g in grads]
        noised_grads = [g + tf.random.normal(tf.shape(g), stddev=self.noise_multiplier * self.clip_norm)
                        for g in clipped_grads]
        return noised_grads

    def train(self, real_data, epochs=50, batch_size=64):
        dataset = tf.data.Dataset.from_tensor_slices(real_data).shuffle(10000).batch(batch_size)
        optimizer_g = tf.keras.optimizers.Adam(1e-4)
        optimizer_d = tf.keras.optimizers.Adam(1e-4)

        for epoch in range(epochs):
            for real_batch in dataset:
                noise = tf.random.normal((batch_size, self.latent_dim))
                with tf.GradientTape() as tape_d:
                    fake = self.generator(noise)
                    logits_real = self.discriminator(real_batch)
                    logits_fake = self.discriminator(fake)
                    loss_d = tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
                grads_d = tape_d.gradient(loss_d, self.discriminator.trainable_variables)
                dp_grads_d = self._apply_dp(grads_d)
                optimizer_d.apply_gradients(zip(dp_grads_d, self.discriminator.trainable_variables))

                noise = tf.random.normal((batch_size, self.latent_dim))
                with tf.GradientTape() as tape_g:
                    fake = self.generator(noise)
                    logits_fake = self.discriminator(fake)
                    loss_g = -tf.reduce_mean(logits_fake)
                grads_g = tape_g.gradient(loss_g, self.generator.trainable_variables)
                dp_grads_g = self._apply_dp(grads_g)
                optimizer_g.apply_gradients(zip(dp_grads_g, self.generator.trainable_variables))

    def sample(self, num_samples):
        noise = tf.random.normal((num_samples, self.latent_dim))
        return self.generator(noise).numpy()

# ============ Main Execution ============
if __name__ == "__main__":
    # Load dataset
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
               "hours-per-week", "native-country", "income"]
    data = pd.read_csv("C:/Users/HP/Documents/adult/adult.data", names=columns, sep=", ", engine="python")
    #data = data.sample(n=5000, random_state=42)

    results = {}

    # 1. Original
    X_train, X_test, y_train, y_test = preprocess_data(data)
    results["Original DNN"] = train_dnn(X_train, X_test, y_train, y_test)

    # 2. Mondrian K-Anonymity
    k = 5
    qid = ["age", "education-num", "hours-per-week"]
    k_anon = MondrianKAnonymity(k)
    anon_data = k_anon.anonymize(data, qid)
    anon_data = convert_to_numeric(anon_data, qid)
    X_train_k, X_test_k, y_train_k, y_test_k = preprocess_data(anon_data)
    results["K-Anonymity DNN"] = train_dnn(X_train_k, X_test_k, y_train_k, y_test_k)

    # 3. DP-GAN
    df_enc = data.copy()
    for col in df_enc.columns:
        if df_enc[col].dtype == 'object':
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
    X_data = df_enc.drop(columns=["income"])
    y_data = df_enc["income"]
    dpgan = DPGAN(input_dim=X_data.shape[1])
    dpgan.train(X_data.head(3000).values)

    synthetic = dpgan.sample(5000)
    synth_df = pd.DataFrame(synthetic, columns=X_data.columns)
    synth_df["income"] = np.random.choice([0, 1], size=len(synth_df))
    synth_df = synth_df[columns]
    X_train_dp, X_test_dp, y_train_dp, y_test_dp = preprocess_data(synth_df)
    results["DP-GAN DNN"] = train_dnn(X_train_dp, X_test_dp, y_train_dp, y_test_dp)

    # Results
    for model_name, metric in results.items():
        print(f"\n{model_name}:")
        for m, v in metric.items():
            print(f"  {m}: {v:.4f}")
