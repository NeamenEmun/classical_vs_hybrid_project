## experiment_runner.py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from classical_model import trainClassical
from hybrid_model import trainHybrid
from data_utils import loadCustomCSV, loadBreastCancerCSV
import csv
import os

SEED = 42  ##set seed for reproducible results
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_FILE = "results_log.csv"

def logResult(row):
    ##logs experiment results to CSV file for analysis
    fileExists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not fileExists:
            writer.writeheader()
        writer.writerow(row)

def evaluateModel(model, xTest, yTest, device='cpu'):
    """Evaluate model accuracy on test set."""
    model.eval()
    xTestT = torch.FloatTensor(xTest).to(device)
    yTestT = torch.LongTensor(yTest).to(device)
    
    with torch.no_grad():
        preds = model(xTestT)
        correct = (preds.argmax(axis=1) == yTestT).sum().item()
        accuracy = correct / len(yTest)
    
    return accuracy

def addNoise(x, noiseLevel=0.1):
    """Add Gaussian noise to features."""
    return x + np.random.normal(0, noiseLevel, x.shape)

def reduceData(x, y, fraction=0.5):
    """Reduce dataset size."""
    nSamples = int(len(x) * fraction)
    indices = np.random.choice(len(x), nSamples, replace=False)
    return x[indices], y[indices]

def prepareDataset(datasetName):
    """Load and prepare a dataset (capped at 5K samples)."""
    maxSamples = 5000
    
    if datasetName == "diabetes":
        xAll, xTestAll, yAll, yTestAll = loadCustomCSV("diabetes_dataset.csv")
        xCombined = np.vstack([xAll, xTestAll])
        yCombined = np.concatenate([yAll, yTestAll])
    elif datasetName == "breast-cancer":
        xAll, xTestAll, yAll, yTestAll = loadBreastCancerCSV("breast_cancer.csv")
        xCombined = np.vstack([xAll, xTestAll])
        yCombined = np.concatenate([yAll, yTestAll])
    else:
        raise ValueError(f"Unknown dataset: {datasetName}. Supported: 'diabetes', 'breastcancer'")
    
    if len(xCombined) > maxSamples:
        indices = np.random.choice(len(xCombined), maxSamples, replace=False)
        xCombined = xCombined[indices]
        yCombined = yCombined[indices]
    xTrain, xTest, yTrain, yTest = train_test_split(
        xCombined, yCombined, test_size=0.2, random_state=42, stratify=yCombined
    )
    
    return xTrain, xTest, yTrain, yTest

def runExperiment(datasetName, dataCondition="full", epochs=3):
    """Run a single experiment."""
    print(f"\n{'='*70}")
    print(f"Dataset: {datasetName.upper()} | Condition: {dataCondition.upper()}")
    print(f"{'='*70}")
    
    xTrain, xTest, yTrain, yTest = prepareDataset(datasetName)
    
    if dataCondition == "reduced":
        xTrain, yTrain = reduceData(xTrain, yTrain, fraction=0.3)
    elif dataCondition == "noisy":
        xTrain = addNoise(xTrain, noiseLevel=0.05)  ##noisy training data tests model robustness to data quality issues
    
    inputDim = xTrain.shape[1]
    numClasses = len(set(yTrain))
    
    print(f"Train samples: {len(xTrain)}, Test samples: {len(xTest)}")
    print(f"Features: {inputDim}, Classes: {numClasses}\n")
    
    print("Training Classical Model...")
    classicalModel, classTrainTime = trainClassical(
        xTrain, yTrain, xTest, yTest,
        inputDim=inputDim,
        numClasses=numClasses,
        epochs=epochs,
        batchSize=32,
        lr=1e-3
    )
    classAccuracy = evaluateModel(classicalModel, xTest, yTest)
    print(f"  Classical: Accuracy={classAccuracy:.4f}, Time={classTrainTime:.2f}s\n")
    
    print("Training Hybrid Quantum Model...")
    try:
        hybridModel, hybridAccuracy, hybridTrainTime, losses = trainHybrid(
            xTrain, yTrain, xTest, yTest,
            inputDim=inputDim,
            numClasses=numClasses,
            nQubits=2,
            epochs=epochs,
            batchSize=32,  ##standardized to match classical model
            lr=1e-3
        )
        print(f"  Hybrid:    Accuracy={hybridAccuracy:.4f}, Time={hybridTrainTime:.2f}s\n")
    except Exception as e:
        print(f"  Hybrid model failed: {e}\n")
        hybridAccuracy = None
    
    ##log results to CSV
    logResult({
        "dataset": datasetName,
        "condition": dataCondition,
        "classical_accuracy": classAccuracy,
        "classical_time_sec": classTrainTime,
        "hybrid_accuracy": hybridAccuracy if hybridAccuracy is not None else np.nan,
        "hybrid_time_sec": hybridTrainTime if hybridAccuracy is not None else np.nan,
        "seed": SEED,

        "test_samples": len(xTest)
    })
    
    return {
        "dataset": datasetName,
        "condition": dataCondition,
        "classicalAcc": classAccuracy,
        "classicalTime": classTrainTime,
        "hybridAcc": hybridAccuracy,
        "testSamples": len(xTest)
    }

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SCIENCE FAIR: CLASSICAL vs HYBRID QUANTUM-CLASSICAL ML")
    print("="*70)
    print("\nAvailable datasets:")
    print("  1. Diabetes (45 features, 5000 samples)")
    print("  2. Breast Cancer (30 features, 569 samples)")
    print()
    
    datasetChoice = input("Select dataset (1 or 2): ").strip()
    if datasetChoice == "1":
        selectedDataset = "diabetes"
    elif datasetChoice == "2":
        selectedDataset = "breast-cancer"
    else:
        print("Invalid choice. Defaulting to diabetes.")
        selectedDataset = "diabetes"
    
    print(f"\nRunning experiments on: {selectedDataset.upper()}")
    print("="*70)
    
    results = []
    datasets = [selectedDataset]
    conditions = ["full", "reduced", "noisy"]
    epochs = 3
    
    for dataset in datasets:
        for condition in conditions:
            try:
                result = runExperiment(dataset, condition, epochs=epochs)
                results.append(result)
            except Exception as e:
                print(f"ERROR: {dataset} + {condition}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Dataset':<12} {'Condition':<10} {'Classical':<12} {'Hybrid':<12}")
    print("-"*70)
    
    for r in results:
        classical = f"{r['classicalAcc']:.4f}"
        hybrid = f"{r['hybridAcc']:.4f}" if r['hybridAcc'] is not None else "N/A"
        print(f"{r['dataset']:<12} {r['condition']:<10} {classical:<12} {hybrid:<12}")
    
    print("="*70)
    print("All experiments completed!")
    print("="*70)
