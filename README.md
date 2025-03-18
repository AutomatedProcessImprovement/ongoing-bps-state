# Process State Evaluation

This repository contains code for evaluating process state computation approaches using various real-life and synthetic event logs.

## Installation

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- log_distance_measures
- pix-framework
- ongoing_process_state
- Prosimos (custom branch for short-term simulation)

## Running Evaluation

The main evaluation script is `test/evaluate_with_existing_alog.py`. It compares different approaches for computing process state using various event logs.

### Configuration

The script contains configuration sections for different event logs:
- BPIC 2012
- BPIC 2017
- Academic Credentials
- Work Orders
- Loan Application (steady)
- Loan Application (wobbly)
- P2P (steady)
- P2P (wobbly)

To use a specific configuration:
1. Open `test/evaluate_with_existing_alog.py`
2. Find the desired log section (marked with comments like `# # # # # # # BPIC 2012 # # # # # # #`)
3. Uncomment that section and ensure other sections are commented out
4. The configuration includes:
   - Path to event log
   - Path to BPMN model
   - Path to parameters
   - Number of cases
   - Cut-off date
   - Simulation horizon

### Running

To run the evaluation:
```bash
python test/evaluate_with_existing_alog.py
```

The script will:
1. Create an output directory with a unique ID
2. Load and preprocess the event log
3. Run multiple evaluation iterations (default: 10)
4. For each iteration:
   - Evaluate process state simulation
   - Evaluate warm-up simulation (2 versions of warm-up available)
   - Save results and metrics

### Output

Results are saved in the `outputs/<run_id>` directory, including:
- Reference datasets
- Simulation results
- Process state files
- Evaluation metrics