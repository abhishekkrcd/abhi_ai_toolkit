# run_training.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np # Added for clearer numpy usage

# Import our three core components
from src.data_loader import DataLoader
from src.model import SnoopyAIModel
from src.trainer import Trainer

def generate_synthetic_data(n_samples: int) -> pd.DataFrame:
    """Generates simple linear data: y = 2*x1 + 3*x2 + noise."""
    # Feature 1 (x1) and Feature 2 (x2) are our two input dimensions
    # We use numpy directly for linspace, which gives the array we want.
    feature_A = np.linspace(0, 1, n_samples)
    feature_B = np.linspace(0, 1, n_samples)

    # Target calculation (The underlying "truth" relationship)
    # FIX APPLIED: Removed the erroneous .numpy() call.
    target_Y = (feature_A * 2 + feature_B * 3)

    # Combine into a DataFrame structure
    data = {
        'feature_A': feature_A,
        'feature_B': feature_B,
        'target_Y': target_Y
    }
    return pd.DataFrame(data)

def main():
    print("--- Starting AI Model Training Pipeline ---")

    # --- 1. DATA PIPELINE (Quest 1 & 2) ---
    N_SAMPLES = 1000
    print(f"\n[STEP 1/5] Generating {N_SAMPLES} synthetic data points...")
    df = generate_synthetic_data(N_SAMPLES)

    # Setup DataLoader instance
    data_loader = DataLoader("synthetic_data_path.csv", features=['feature_A', 'feature_B'], target='target_Y')
    X_train_df, y_train_df = data_loader.preprocess(df)

    # Split into training (80%) and testing (20%) sets
    X_train_df, X_test_df, y_train_df, y_test_df = data_loader.split_data(test_size=0.2)

    print(f"   Dataset split complete: Train samples={X_train_df.shape[0]}, Test samples={X_test_df.shape[0]}")

    # Convert Pandas DataFrames to PyTorch Tensors
    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    y_train = torch.tensor(y_train_df.values, dtype=torch.float32)

    # --- 2. MODEL & OPTIMIZATION SETUP (Quest 3 & 4) ---
    INPUT_DIM = 2
    OUTPUT_DIM = 1 # Only predicting one value (the target_Y)

    # Initialize the model
    model = SnoopyAIModel(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)

    # Define Loss Function and Optimizer
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Initialize Trainer
    trainer = Trainer(model, criterion, optimizer)

    # --- 3. TRAINING LOOP (The Core) ---
    N_EPOCHS = 15
    print(f"\n[STEP 2/5] Beginning {N_EPOCHS} epochs of training...")

    history = []
    for epoch in range(N_EPOCHS):
        # Training step
        loss = trainer.train_one_epoch(X_train, y_train)

        # Track loss (simple logging)
        current_loss = loss.item()
        history.append(current_loss)

        # Report loss, formatted to 4 decimal places
        print(f"Epoch {epoch+1}/{N_EPOCHS}: Loss = {current_loss:.4f}")

    print("\n[STEP 3/5] Training complete.")

    # --- 4. EVALUATION (Testing Performance) ---
    # Put model in evaluation mode
    model.eval()
    print("\n[STEP 4/5] Starting final evaluation on the test set...")

    # Run evaluation on the test data
    with torch.no_grad():
        test_predictions = model(torch.tensor(X_test_df.values, dtype=torch.float32))
        test_loss = criterion(test_predictions, torch.tensor(y_test_df.values, dtype=torch.float32))

    print(f"\n[STEP 5/5] Evaluation Complete.")
    print(f"Final Test Loss (MSE): {test_loss.item():.6f}")

    # --- DEPLOYMENT / INFERENCE DEMO ---
    print("\n" + "="*50)
    print("🔮 DEPLOYMENT TEST: MAKING A LIVE PREDICTION")
    print("="*50)

    # 1. Prepare the input: Let's simulate a new input (e.g., a new point (0.5, 0.8))
    sample_input_data = torch.tensor([[0.5, 0.8]], dtype=torch.float32)
    print(f"➡️ Input Sample Given (Features A, B): {sample_input_data.numpy()}")

    # 2. Ensure the model is in evaluation mode
    model.eval()

    # 3. Use torch.no_grad() to disable gradient tracking for prediction
    with torch.no_grad():
        # Run the inference! Model takes the input and predicts the output.
        predictions = model(sample_input_data)

        # 4. Clean up the output for human reading
        predicted_value = predictions.item()
        print(f"\n✅ Model Predicts Output (Estimated Y): {predicted_value:.4f}")

    print("\n--- Inference Demo Complete. Model is ready for deployment! ---\n")

if __name__ == "__main__":
    main()