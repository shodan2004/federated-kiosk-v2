import os
import numpy as np
import tensorflow as tf
import flwr as fl
from tensorflow import keras
from datetime import datetime, timezone
from supabase import create_client, Client

# 📍 Load Supabase credentials from env vars
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🌐 Simulate real per-client data partition
CLIENT_ID = int(os.environ.get("CLIENT_ID", 1))
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

def get_partition(x, y, client_id, num_clients=3):
    size = len(x) // num_clients
    start = (client_id - 1) * size
    end = client_id * size
    return x[start:end], y[start:end]

x_train, y_train = get_partition(x_train, y_train, CLIENT_ID)
x_test, y_test = get_partition(x_test, y_test, CLIENT_ID)

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 🧠 Define model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# 🌼 Define Flower client
class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

        # ⏱️ Timestamp with timezone
        now = datetime.now(timezone.utc).isoformat()

        # 📤 Log metrics to Supabase
        try:
            supabase.table("training_logs").insert({
                "round": config.get("server_round", -1),
                "client_id": self.client_id,
                "loss": round(float(loss), 4),
                "accuracy": round(float(accuracy), 4),
                "val_loss": round(float(loss), 4),   # Optional, same as loss here
                "val_accuracy": round(float(accuracy), 4),
                "kiosk_id": f"Kiosk_0{self.client_id}",
                "timestamp": now,
            }).execute()
        except Exception as e:
            print("❌ Failed to push to Supabase:", e)

        return loss, len(x_test), {"accuracy": accuracy}

# 🟢 Run client
fl.client.start_client(server_address="localhost:8080", client=FLClient(CLIENT_ID).to_client())
