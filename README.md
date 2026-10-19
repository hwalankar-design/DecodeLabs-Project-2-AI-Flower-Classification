# DecodeLabs Project 2 — Data Classification Using AI

A beginner-friendly supervised machine-learning project that classifies Iris flowers using a Decision Tree classifier.

## Assignment requirements covered

- Load and inspect a dataset
- Split the data into training and testing sets
- Apply a classification algorithm
- Train and evaluate a model
- Demonstrate a prediction on a new example

## Technology

- Python 3.10 or newer
- pandas
- scikit-learn
- Matplotlib

## Dataset

The Iris dataset bundled with scikit-learn contains 150 flower records, four numeric measurements, and three species classes. The project does not require downloading a separate dataset.

## Run in VS Code

1. Install Python and the Python extension in VS Code.
2. Open this project folder in VS Code.
3. Open Terminal → New Terminal.
4. (Recommended) Create and activate a virtual environment:

   **Windows PowerShell**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```

5. Install the libraries:
   ```bash
   python -m pip install -r requirements.txt
   ```

6. Run the program:
   ```bash
   python main.py
   ```

## Outputs

After running, the `outputs/` folder contains:

- `iris_dataset.csv` — readable dataset with species labels
- `evaluation_report.txt` — accuracy, classification report, and confusion matrix
- `confusion_matrix.png` — visual comparison of actual and predicted classes
- `feature_importance.png` — relative importance of the input measurements

The actual accuracy is calculated when the model runs. It can vary if the split or model settings are changed; do not claim a score without running the program.

## How it works

1. Load the Iris dataset.
2. Separate measurements (`X`) from species labels (`y`).
3. Split records into 80% training and 20% testing data, preserving class balance.
4. Train a Decision Tree classifier on the training set.
5. Evaluate predictions on the held-out test set.
6. Save reports and charts.
7. Predict the species for one example flower.

## Important note

This is an educational demonstration, not a production-grade botanical diagnostic system. The test score estimates performance on this small dataset and does not guarantee accuracy on every real-world flower.

## GitHub upload

Create a public repository named `DecodeLabs-Project-2-AI-Flower-Classification`, then upload the project files. Do not upload `.venv/` or personal credentials.
