Python application designed to process an event log and a BPMN (Business Process Model and Notation) model to compute the state of ongoing cases in a business process. It uses the process_running_state library (https://github.com/AutomatedProcessImprovement/process-running-state) to compute the state based on the last few executed activities.

To install:

1. Set python virtual environment
python -m venv .venv

2. Activate virtual environment
.venv/Scripts/activate

3. Install required packages
pip install pandas lxml networkx xmltodict poetry

4. Navigate to process_running_state directory
cd libs/process_running_state

5. Install dependencies there
poetry install

To run, use:
python main.py path/to/event_log.csv path/to/bpmn_model.bpmn path/to/parameters.json --start_time "YYYY-MM-DDTHH:MM:SS" --column_mapping '{"csv_column_name": "ExpectedColumnName", ...}'

(currently parameters not used)

Examples:
- With start_time specified

python main.py samples/dev-samples/synthetic_xor_loop_ongoing.csv samples/dev-samples/synthetic_xor_loop.bpmn '{"param1": "value1", "param2": "value2"}' --column_mapping '{"case_id": "CaseId", "Resource": "Resource", "Activity": "Activity", "__start_time": "StartTime", "end_time": "EndTime"}' --start_time "2012-03-21T10:10:00.000Z" --simulation_horizon "2012-04-25T23:10:30.000Z"

python main.py samples/dev-samples/synthetic_and_k5_ongoing.csv samples/dev-samples/synthetic_and_k5.bpmn '{"param1": "value1", "param2": "value2"}' --column_mapping '{"case_id": "CaseId", "Resource": "Resource", "Activity": "Activity", "__start_time": "StartTime", "end_time": "EndTime"}' --start_time "2012-03-21T10:10:00.000Z" --simulation_horizon "2012-04-25T23:10:30.000Z"

- Without start_time specified

python main.py samples/dev-samples/synthetic_xor_loop_ongoing.csv samples/dev-samples/synthetic_xor_loop.bpmn '{"param1": "value1", "param2": "value2"}' --column_mapping '{"case_id": "CaseId", "Resource": "Resource", "Activity": "Activity", "__start_time": "StartTime", "end_time": "EndTime"}'

python main.py samples/dev-samples/synthetic_and_k5_ongoing.csv samples/dev-samples/synthetic_and_k5.bpmn '{"param1": "value1", "param2": "value2"}' --column_mapping '{"case_id": "CaseId", "Resource": "Resource", "Activity": "Activity", "__start_time": "StartTime", "end_time": "EndTime"}'


Example in output.json is used on  python main.py samples/dev-samples/synthetic_xor_loop_ongoing.csv samples/dev-samples/synthetic_xor_loop.bpmn '{"param1": "value1", "param2": "value2"}' --column_mapping '{"case_id": "CaseId", "Resource": "Resource", "Activity": "Activity", "__start_time": "StartTime", "end_time": "EndTime"}' --start_time "2012-03-21T10:10:00.000Z" --simulation_horizon "2012-04-25T23:10:30.000Z" --output_file output.json