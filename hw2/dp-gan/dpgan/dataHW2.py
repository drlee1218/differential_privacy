import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ============ Data Preprocessing ============
def preprocess_data(df):
    df = df.copy().apply(LabelEncoder().fit_transform)
    X = df.drop(columns=["income"])
    y = df["income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test

# ============ DNN Model ============
def train_dnn(X_train, X_test, y_train, y_test):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
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
        df[col] = df[col].apply(lambda x: np.mean([float(num) for num in x.strip("[]").split("-")]))
    return df

# ============ DP-GAN (TensorFlow 2.x) ============
class DPGAN(tf.keras.Model):
    def __init__(self, input_dim, latent_dim=100):
        super(DPGAN, self).__init__()
        self.generator = self.make_generator(latent_dim, input_dim)
        self.discriminator = self.make_discriminator(input_dim)
        self.latent_dim = latent_dim
        self.input_dim = input_dim

    def make_generator(self, latent_dim, output_dim):
        return tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(output_dim, activation='sigmoid')
        ])

    def make_discriminator(self, input_dim):
        return tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])

    def train(self, real_data, epochs=500, batch_size=64, epsilon=1.0, delta=1e-5):
        optimizer_d = tf.keras.optimizers.Adam(1e-4)
        optimizer_g = tf.keras.optimizers.Adam(1e-4)
        data = tf.data.Dataset.from_tensor_slices(real_data).shuffle(10000).batch(batch_size)

        for epoch in range(epochs):
            for real_batch in data:
                noise = tf.random.normal((batch_size, self.latent_dim))
                with tf.GradientTape() as tape_d:
                    fake = self.generator(noise, training=True)
                    logits_real = self.discriminator(real_batch, training=True)
                    logits_fake = self.discriminator(fake, training=True)
                    loss_d = tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
                grads_d = tape_d.gradient(loss_d, self.discriminator.trainable_variables)
                optimizer_d.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))

                # Update generator
                noise = tf.random.normal((batch_size, self.latent_dim))
                with tf.GradientTape() as tape_g:
                    fake = self.generator(noise, training=True)
                    logits_fake = self.discriminator(fake, training=True)
                    loss_g = -tf.reduce_mean(logits_fake)
                grads_g = tape_g.gradient(loss_g, self.generator.trainable_variables)
                optimizer_g.apply_gradients(zip(grads_g, self.generator.trainable_variables))

    def sample(self, num_samples):
        noise = tf.random.normal((num_samples, self.latent_dim))
        return self.generator(noise).numpy()

# ============ Main Execution ============
if __name__ == "__main__":
    # Load data
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
               "hours-per-week", "native-country", "income"]
    data = pd.read_csv("C:\\Users\\drlee\\Downloads\\adult\\adult.data", names=columns, sep=", ", engine="python")

    results = {}

    # Original dataset
    X_train, X_test, y_train, y_test = preprocess_data(data)
    results["Original DNN"] = train_dnn(X_train, X_test, y_train, y_test)

    # K-Anonymity
    k = 5
    quasi_identifiers = ["age", "education-num", "hours-per-week"]
    k_anon = MondrianKAnonymity(k)
    k_anon_data = k_anon.anonymize(data, quasi_identifiers)
    k_anon_data = convert_to_numeric(k_anon_data, quasi_identifiers)
    X_train_k, X_test_k, y_train_k, y_test_k = preprocess_data(k_anon_data)
    results["K-Anonymity DNN"] = train_dnn(X_train_k, X_test_k, y_train_k, y_test_k)

    # DP-GAN
    df_encoded = data.copy().apply(LabelEncoder().fit_transform)
    X_data = df_encoded.drop(columns=["income"])
    y_data = df_encoded["income"]
    dpgan = DPGAN(input_dim=X_data.shape[1])
    dpgan.train(X_data.values, epochs=200)

    synthetic_data = dpgan.sample(10000)
    synthetic_df = pd.DataFrame(synthetic_data, columns=X_data.columns)
    synthetic_df["income"] = np.random.choice([0, 1], size=(len(synthetic_df),))

    synthetic_df = synthetic_df[columns]  # Match column order
    X_train_dp, X_test_dp, y_train_dp, y_test_dp = preprocess_data(synthetic_df)
    results["DP-GAN DNN"] = train_dnn(X_train_dp, X_test_dp, y_train_dp, y_test_dp)

    # Print results
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
