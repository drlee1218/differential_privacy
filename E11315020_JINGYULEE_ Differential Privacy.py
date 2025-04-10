import tensorflow as tf
import tensorflow_privacy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

# Load the Adult dataset
def load_data():
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    df = pd.read_csv("C:/Users/drlee/Downloads/adult/adult.data", header=None, names=column_names, na_values=" ?", skipinitialspace=True)
    df.dropna(inplace=True)
    X = df.drop(columns='income')
    y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    return X, y

# Data preprocessing
def preprocess_data(X):
    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])  # set sparse_output=False

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    X_processed = preprocessor.fit_transform(X)
    return X_processed

# Define the model
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Compile the model with Differential Privacy
def compile_model_with_dp(model, noise_multiplier, l2_norm_clip, batch_size):
    optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=1,
        learning_rate=0.001
    )
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Compute epsilon using the privacy accountant
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

def compute_epsilon(noise_multiplier, n, batch_size, epochs, delta=1e-5):
    epsilon, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
        n=n,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=epochs,
        delta=delta
    )
    return epsilon

# Load and preprocess data
X, y = load_data()
X_processed = preprocess_data(X)

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Differential Privacy settings
noise_values = [0.5, 1.0, 1.5, 2.0, 3.0]  # List of noise_multiplier values for experiments
l2_norm_clip = 1.0
batch_size = 256
epochs = 10
delta = 1e-5
n = X_train.shape[0]

results = []

# Run experiments for each noise multiplier
for noise_multiplier in noise_values:
    print(f"\n▶ Training with noise_multiplier = {noise_multiplier}")

    # Build and compile model
    model = build_model(input_dim=X_train.shape[1])
    model = compile_model_with_dp(model, noise_multiplier, l2_norm_clip, batch_size)

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predict and evaluate
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Compute epsilon (ε is privacy loss — inversely proportional to noise)
    epsilon = compute_epsilon(noise_multiplier=noise_multiplier, n=n, batch_size=batch_size, epochs=epochs, delta=delta)

    # Save results
    results.append({
        'Noise Multiplier': noise_multiplier,
        'Epsilon (ε)': round(epsilon, 2),
        'Accuracy': round(accuracy, 4),
        'Precision (class 0)': round(report['0']['precision'], 4),
        'Recall (class 0)': round(report['0']['recall'], 4),
        'F1-score (class 0)': round(report['0']['f1-score'], 4),
        'Precision (class 1)': round(report['1']['precision'], 4),
        'Recall (class 1)': round(report['1']['recall'], 4),
        'F1-score (class 1)': round(report['1']['f1-score'], 4)
    })

# Print summary table
results_df = pd.DataFrame(results)
print("\nDifferential Privacy Results Summary:")
print(results_df)
