# ----------- CONFIG -----------

NUM_RUNS = 10

# # # # # # #
# BPIC 2012 #
# # # # # # #

# EXISTING_ALOG_PATH = "../samples/real_life/BPIC_2012_W.csv.gz"
# BPMN_MODEL = "../samples/real_life/2012_diff_extr/best_result/BPIC_2012_train.bpmn"
# BPMN_PARAMS = "../samples/real_life/2012_diff_extr/best_result/BPIC_2012_train.json"
# PROC_TOTAL_CASES = 5000
# WARMUP_TOTAL_CASES = 5000
# Split point
# SIMULATION_CUT_DATE = pd.to_datetime("2012-01-16T13:00:00.000Z", utc=True)
# Horizon:
# horizon = pd.Timedelta(days=23)  # 90% percentile of trace durations

# # # # # # #
# BPIC 2017 #
# # # # # # #

EXISTING_ALOG_PATH = "../samples/real_life/BPIC_2017_W.csv.gz"
BPMN_MODEL = "../samples/real_life/2017_short_term/best_result/BPIC_2017_W_train.bpmn"
BPMN_PARAMS = "../samples/real_life/2017_short_term/best_result/BPIC_2017_W_train.json"
PROC_TOTAL_CASES = 10000
WARMUP_TOTAL_CASES = 10000
# Split point
SIMULATION_CUT_DATE = pd.to_datetime("2016-10-10T13:00:00.000Z", utc=True)
# Horizon:
horizon = pd.Timedelta(days=26)  # 90% percentile of trace durations

# # # # # # # # # # # # #
# Academic  credentials #
# # # # # # # # # # # # #

# EXISTING_ALOG_PATH = "../samples/real_life/AcademicCredentials.csv.gz"
# BPMN_MODEL = "../samples/real_life/academic_diff_extr/best_result/AcademicCredentials_train.bpmn"
# BPMN_PARAMS = "../samples/real_life/academic_diff_extr/best_result/AcademicCredentials_train.json"
# PROC_TOTAL_CASES = 5000
# WARMUP_TOTAL_CASES = 5000
# Split point
# SIMULATION_CUT_DATE = pd.to_datetime("2016-05-02T13:00:00.000Z", utc=True)  # Monday
# Horizon:
# horizon = pd.Timedelta(days=46)  # 90% percentile of trace durations

# # # # # # # #
# Work Orders #
# # # # # # # #

# EXISTING_ALOG_PATH = "../samples/real_life/work_orders.csv.gz"
# BPMN_MODEL = "../samples/real_life/workorders_diff_extr/best_result/work_orders_train.bpmn"
# BPMN_PARAMS = "../samples/real_life/workorders_diff_extr/best_result/work_orders_train.json"
# PROC_TOTAL_CASES = 20000
# WARMUP_TOTAL_CASES = 20000
# Split point
# SIMULATION_CUT_DATE = pd.to_datetime("2022-12-19T07:00:00.000Z", utc=True)  # Monday
# Horizon:
# horizon = pd.Timedelta(days=24)  # 90% percentile of trace durations

# Compute rest of instants
WARMUP_START_DATE = SIMULATION_CUT_DATE - horizon
EVALUATION_END_DATE = SIMULATION_CUT_DATE + horizon
SIMULATION_HORIZON = SIMULATION_CUT_DATE + (2 * horizon)
