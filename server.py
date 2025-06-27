import flwr as fl
import os
from supabase import create_client, Client
from datetime import datetime

# Load Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("SUPABASE_URL and SUPABASE_KEY must be set")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Logging function
def log_training_to_supabase(round_num, client_id, loss, val_loss, accuracy, kiosk_id):
    data = {
        "round": round_num,
        "client_id": client_id,
        "loss": loss,
        "val_loss": val_loss,
        "accuracy": accuracy,
        "kiosk_id": kiosk_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    supabase.table("training_logs").insert(data).execute()

# Custom strategy for logging
class SaveMetricsStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        for client_idx, (client, fit_res) in enumerate(results):
            metrics = fit_res.metrics
            if metrics:
                log_training_to_supabase(
                    round_num=rnd,
                    client_id=client_idx + 1,
                    loss=metrics.get("loss", 0),
                    val_loss=metrics.get("val_loss", 0),
                    accuracy=metrics.get("accuracy", 0),
                    kiosk_id=f"Kiosk_0{client_idx + 1}"
                )
        return aggregated_result

# Start server
if __name__ == "__main__":
    strategy = SaveMetricsStrategy()

    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
