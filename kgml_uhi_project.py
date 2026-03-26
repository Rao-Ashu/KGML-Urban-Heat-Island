import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt



# 1. Loading Data and Simulating Thermodynamic LST

print("1. Loading Data and Simulating Thermodynamic LST...")
# Load your original data
# df = pd.read_csv('Data.csv')
df = pd.read_csv('Data_Massive.csv')


# Domain Knowledge (Physics/Thermodynamics):
# Buildup (Concrete) absorbs heat. Vegetation cools via evapotranspiration.
# So idea is to generate a synthetic Land Surface Temperature (LST) column using this law.
# Base Temp = 30°C. Buildup adds heat (+0.03 coef), Veg removes heat (-0.02 coef).
np.random.seed(42)
noise = np.random.normal(0, 0.5, len(df))
df['LST_True'] = 30 + (df['Buildup'] * 0.03) - (df['Vegetation'] * 0.02) + noise

# Features: [Vegetation, Barren, Water, Buildup]
X = df[['Vegetation', 'Barren', 'Water', 'Buildup']].values.astype(np.float32)
y = df['LST_True'].values.astype(np.float32).reshape(-1, 1)

# Normalize inputs for neural network stability
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)




# 2. Defining the Physics-Informed Neural Network (PINN)

print("\n2. Defining the Physics-Informed Neural Network (PINN)...")
# subclass tf.keras.Model to create a custom KGML training loop
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = Dense(16, activation='relu')
        self.dense2 = Dense(8, activation='relu')
        self.out = Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)

    # THIS IS THE KGML MAGIC: Custom Physics-Guided Loss Function
    def train_step(self, data):
        X_batch, y_batch = data
        
        with tf.GradientTape() as tape:
            y_pred = self(X_batch, training=True)
            
            # 1. Standard Data Loss (Mean Squared Error)
            mse_loss = tf.reduce_mean(tf.square(y_batch - y_pred))
            
            # 2. Physics-Guided Loss (Domain Knowledge Constraint)
            # Extract unnormalized Buildup (index 3) and Veg (index 0) to enforce physics
            # If the model predictions deviate from the expected thermodynamic heat index, penalize it.
            # (This is my sort of "Surface Energy Balance Equation")
            expected_heat_trend = 30 + (X_batch[:, 3] * 0.03) - (X_batch[:, 0] * 0.02)
            physics_loss = tf.reduce_mean(tf.square(y_pred - tf.expand_dims(expected_heat_trend, axis=1)))
            
            # Combine losses (Lambda weight for physics = 0.5)
            total_loss = mse_loss + (0.5 * physics_loss)
            
        # Backpropagation and optimization 
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"total_loss": total_loss, "mse_loss": mse_loss, "physics_penalty": physics_loss}




# 3. Training the KGML Model

print("\n3. Training the KGML Model...")
kgml_model = PINN()
kgml_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# Train the model
history = kgml_model.fit(X_norm, y, epochs=150, verbose=0)

print("Training Complete. Final Metrics:")
print(f"Total Loss: {history.history['total_loss'][-1]:.4f}")
print(f"Data Loss (MSE): {history.history['mse_loss'][-1]:.4f}")
print(f"Physics Penalty: {history.history['physics_penalty'][-1]:.4f}")




# 4. Evaluation and Proof of Concept

print("\n4. Testing the KGML predictions...")
predictions = kgml_model.predict(X_norm, verbose=0)
# Prepare a safe 'year' sequence for printing 
if 'year' in df.columns:
    year_series = df['year'].astype(str).tolist()
elif 'Date' in df.columns:
    # If a Date column exists (string), parse and extract year
    try:
        year_series = pd.to_datetime(df['Date']).dt.year.astype(str).tolist()
    except Exception:
        year_series = [str(i) for i in range(len(df))]
elif isinstance(df.index, pd.DatetimeIndex):
    year_series = df.index.year.astype(str).tolist()
else:
    # fallback to a simple sequential index
    year_series = [str(i) for i in range(len(df))]

for i in range(len(df)):
    year = year_series[i]
    actual = df['LST_True'].iloc[i]
    pred = predictions[i][0]
    print(f"Year {year}: Actual Temp: {actual:.2f}°C | KGML Predicted: {pred:.2f}°C")

print("\n KGML Concept Successfully Demonstrated!")