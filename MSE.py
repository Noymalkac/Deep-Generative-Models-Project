import torch
from torchmetrics.functional import mean_squared_error
from datasets import DataModule
from neural_boltzmann_machine import NBM
import os

# === CONFIG ===
DATA_DIR = "datasets"
OUTPUT_TYPE = "gaussian"
DATASET_NAME = "MNIST"
BATCH_SIZE = 128
NUM_CLASSES = 10
N_VIS = 28 * 28
N_HID = 64

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ARCHITECTURE ===
model = NBM(NUM_CLASSES, N_VIS, N_HID, visible_unit_type=OUTPUT_TYPE)

# === LOAD SUBMODULES ===
model.bias_net.load_state_dict(torch.load("bias_net.pth", map_location=device, weights_only=True))
model.precision_net.load_state_dict(torch.load("precision_net.pth", map_location=device, weights_only=True))
model.weights_net.load_state_dict(torch.load("weights_net.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

# === LOAD DATA ===
dm = DataModule(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    output_type=OUTPUT_TYPE,
    dataset_name=DATASET_NAME,
)
dm.prepare_data()
dm.setup(stage="test")
test_loader = dm.test_dataloader(num_workers=0)

# === EVALUATE MSE ON TEST DATA ===
all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        v = x.view(x.size(0), -1)

        bias = model.bias_net(y)
        precision = model.precision_net(y)
        weights = model.weights_net(y)

        for _ in range(50):  # Gibbs steps
            h = model._sample_hid(v, bias, precision, weights)
            v = model._sample_vis(h, bias, precision, weights)

        all_preds.append(v)
        all_targets.append(x.view(x.size(0), -1))


# Combine all batches
all_preds = torch.cat(all_preds, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Compute Mean Squared Error (MSE)
mse_original = mean_squared_error(all_preds, all_targets)

print(f"\n NBM original: \n Mean Squared Error (MSE) on test set: {mse_original.item():.4f}")

import torch
from torchmetrics.functional import mean_squared_error
from datasets import DataModule
from NBM_altered import NBM
import os

# === CONFIG ===
DATA_DIR = "datasets"
OUTPUT_TYPE = "gaussian"
DATASET_NAME = "MNIST"
BATCH_SIZE = 128
NUM_CLASSES = 10
N_VIS = 28 * 28
N_HID = 64

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ARCHITECTURE ===
model = NBM(NUM_CLASSES, N_VIS, N_HID, visible_unit_type=OUTPUT_TYPE)

# === LOAD SUBMODULES ===
model.bias_net.load_state_dict(torch.load("bias_net_altered.pth", map_location=device, weights_only=True))
model.precision_net.load_state_dict(torch.load("precision_net_altered.pth", map_location=device, weights_only=True))
model.weights_net.load_state_dict(torch.load("weights_net_altered.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

# === LOAD DATA ===
dm = DataModule(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    output_type=OUTPUT_TYPE,
    dataset_name=DATASET_NAME,
)
dm.prepare_data()
dm.setup(stage="test")
test_loader = dm.test_dataloader(num_workers=0)

# === EVALUATE MSE ON TEST DATA ===
all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        v = x.view(x.size(0), -1)

        bias = model.bias_net(y)
        precision = model.precision_net(y)
        weights = model.weights_net(y)

        for _ in range(50):  # Gibbs steps
            h = model._sample_hid(v, bias, precision, weights)
            v = model._sample_vis(h, bias, precision, weights)

        all_preds.append(v)
        all_targets.append(x.view(x.size(0), -1))


# Combine all batches
all_preds = torch.cat(all_preds, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Compute Mean Squared Error (MSE)
mse_altered = mean_squared_error(all_preds, all_targets)

print(f"\n NBM altered: \n Mean Squared Error (MSE) on test set: {mse_altered.item():.4f}")

# Calculate percentage increase in MSE
worsening_percentage = ((mse_altered - mse_original) / mse_original) * 100

print(f"\n The new model is worse by {worsening_percentage:.2f}% in terms of MSE.")

