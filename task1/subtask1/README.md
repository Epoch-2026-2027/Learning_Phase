# Report: MLP Model for Ranking a 10-Length Array

## 1) Objective
Build and train a Multi-Layer Perceptron (MLP) that predicts the rank position of each element in an input array of length 10.

- **Input (`X`)**: 10 numeric values.
- **Target (`y`)**: 10 rank values (one rank per input position).

The model learns a mapping:
$$
f: \mathbb{R}^{10} \rightarrow \mathbb{R}^{10}
$$
where output scores are converted to a ranking using double `argsort`.

---

## 2) Data Pipeline

From `ranking_dataset.csv`:
- Features: `data.iloc[:, 0:10]`
- Labels: `data.iloc[:, 10:20]`

Custom dataset:
- `RankingDataset(Dataset)` converts features/labels to `torch.float32` tensors.

Split and loaders:
- Dataset split: **60% train, 20% test, 20% validation**
- `DataLoader(..., batch_size=32, shuffle=True)` for each split.

---

## 3) Model Architecture

Implemented as `MLP(torch.nn.Module)` with `Sequential` layers:

1. `Linear(10, 64)`
2. `ReLU`
3. `Linear(64, 64)`
4. `ReLU`
5. `Linear(64, 64)`
6. `ReLU`
7. `Linear(64, 10)`

Notes:
- Hidden dimension: 64
- 3 hidden linear layers + ReLU nonlinearities
- Final layer outputs 10 continuous values used as ranking scores.

Device selection:
- CUDA if available, else CPU.

---

## 4) Training Setup

Training function: `train(model, train_dataloader, valid_dataloader, epochs=10)`

- Optimizer: **Adam** (`lr=0.001`)
- Loss: **MSELoss** between predicted score vector and target rank vector
- Tracks:
	- average training loss per epoch
	- average validation loss per epoch
- In notebook run: `epochs=100`

A plot is generated for train vs validation loss over epochs.

---

## 5) Testing and Ranking Conversion

### Test loss
Model is evaluated on test split using MSE:

- `Test Loss = mean batch MSE over test loader`

### Ranking extraction
Predicted scores are converted to rank indices with:

`torch.argsort(torch.argsort(model_output, dim=1), dim=1)`

This yields per-sample rank assignments for 10 positions.

### Accuracy metric used
For each sample:

- Compare predicted ranks to true ranks elementwise
- Compute mean match ratio over 10 positions
- Aggregate into a histogram and report average accuracy.

---

## 6) Qualitative Example Output

Notebook prints one random test sample with:

- raw input array
- predicted ranking scores
- true ranking labels
- predicted ranks after argsort processing
- input sorted by true ranking
- input sorted by predicted ranking

This helps visually verify whether the model learns ordering behavior.

---

## 7) Summary

The notebook implements a complete supervised learning pipeline for ranking 10-element arrays using an MLP. The model is simple, fast to train, and evaluated using both regression loss (MSE) and ranking-match accuracy. The approach is appropriate for small fixed-length ranking tasks and can be improved further with rank-specific losses (e.g., pairwise/listwise objectives) if higher ranking fidelity is needed.

