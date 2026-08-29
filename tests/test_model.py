# tests/test_model.py (Updated)
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
# Update the import statement
from src.model import SnoopyAIModel
from src.trainer import Trainer

class TestSnoopyAIModel(unittest.TestCase): # <--- Renamed the test class
    def test_model_initialization_and_forward_pass(self):
        # Define fake dimensions based on our structured data
        INPUT_DIM = 2
        OUTPUT_DIM = 2
        BATCH_SIZE = 10

        # 1. Setup: Instantiate the model using the new name
        model = SnoopyAIModel(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM) # <--- Use new class name

        # 2. Setup: Create a mock input tensor
        mock_input = torch.randn(BATCH_SIZE, INPUT_DIM)

        # 3. Action: Run the forward pass
        output = model(mock_input)

        # 4. Assert: Check the output shape
        self.assertEqual(output.shape, (BATCH_SIZE, OUTPUT_DIM),
                         "Model output shape did not match expected dimensions.")


# Modify tests/test_model.py (Add a new test method)

# ... (rest of the test file) ...

class TestSimpleAIModel(unittest.TestCase):
    # ... (existing test_load_and_preprocess_success) ...
    def test_train_one_epoch_success(self):
        print("\n--- Running Training Test ---")
        # Setup constants
        INPUT_DIM = 2
        OUTPUT_DIM = 2
        BATCH_SIZE = 15

        # 1. Setup Model (Snoopy)
        model = SnoopyAIModel(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)

        # 2. Setup Optimizers and Loss
        criterion = nn.CrossEntropyLoss() # Using a common classification loss
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(model, criterion, optimizer)

        # 3. Setup Mock Data (must be Torch Tensors!)
        mock_X = torch.randn(BATCH_SIZE, INPUT_DIM)
        # For CrossEntropyLoss, y_train should be class indices (long tensor)
        mock_y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,))

        # 4. --- Test Logic ---
        # Save an initial state of the model's weights (optional, but good practice)
        initial_weight = model.fc1.weight.data.clone().mean().item()

        # Run the function under test
        trainer.train_one_epoch(mock_X, mock_y)

        # 5. Assertion: Check if the model parameters have changed (proof that training occurred)
        final_weight = model.fc1.weight.data.clone().mean().item()

        # The weight should *not* be equal to the initial weight
        self.assertNotAlmostEqual(initial_weight, final_weight, places=5,
                                   msg="Model weights did not change after calling train_one_epoch.")