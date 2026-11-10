
# ML Classification Threshold Optimizer

A Python project that selects the best classification threshold
based on a minimum recall requirement while maximizing precision.

## Features

- Evaluate classification thresholds
- Select the highest-precision qualifying threshold
- Enforce minimum recall requirements
- Validate input values
- Handle cases where no threshold qualifies
- Automated testing with pytest


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ml-threshold-optimizer.git
cd ml-threshold-optimizer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Running Tests

Run the automated tests using:

```bash
python -m pytest -v
```


## Usage Example

Import the optimizer:

```python
from threshold_optimizer import find_best_threshold

thresholds = [0.2, 0.5, 0.7]
precisions = [0.4, 0.8, 0.9]
recalls = [1.0, 0.85, 0.6]

result = find_best_threshold(thresholds, precisions, recalls, minimum_recall=0.8)

print(result)
```

Expected output:

```text
(0.5, 0.8)
```

The optimizer selects the threshold with the highest precision among those satisfying the minimum recall requirement.

If no threshold qualifies, the function returns `None`.