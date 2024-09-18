from pathlib import Path

from process_running_state.bpmn_model import BPMNNodeType
from process_running_state.utils import read_bpmn_model


def test_read_bpmn_model():
    bpmn_path = Path("./tests/assets/bpmn_model_test.bpmn")
    bpmn_model = read_bpmn_model(bpmn_path)
    # Assert number of elements
    assert len(bpmn_model.nodes) == 14
    assert len([node for node in bpmn_model.nodes if node.type == BPMNNodeType.TASK]) == 7
    assert len([node for node in bpmn_model.nodes if node.type == BPMNNodeType.START_EVENT]) == 1
    assert len([node for node in bpmn_model.nodes if node.type == BPMNNodeType.END_EVENT]) == 1
    assert len([node for node in bpmn_model.nodes if node.type == BPMNNodeType.INTERMEDIATE_EVENT]) == 1
    assert len([node for node in bpmn_model.nodes if node.type == BPMNNodeType.EXCLUSIVE_GATEWAY]) == 2
    assert len([node for node in bpmn_model.nodes if node.type == BPMNNodeType.PARALLEL_GATEWAY]) == 2
    assert len([node for node in bpmn_model.nodes if node.type == BPMNNodeType.INCLUSIVE_GATEWAY]) == 0
    assert len(bpmn_model.flows) == 15
    # Check specific node names
    assert bpmn_model.id_to_node["task-10"].name == "C"
    assert bpmn_model.id_to_node["task-21"].name == "E"
    assert bpmn_model.id_to_node["task-26"].name == "G"
    assert bpmn_model.id_to_node["event-0"].name == "Start"
    assert bpmn_model.id_to_node["event-19"].name == "Timer"
    assert bpmn_model.id_to_node["gateway-4"].name == "AND-split"
    assert bpmn_model.id_to_node["gateway-15"].name == "XOR-join"
    # Search for specific flows
    assert bpmn_model.id_to_flow["edge-3"].source == "task-2"
    assert bpmn_model.id_to_flow["edge-3"].target == "gateway-4"
    assert bpmn_model.id_to_flow["edge-6"].source == "gateway-4"
    assert bpmn_model.id_to_flow["edge-6"].target == "task-12"
    assert bpmn_model.id_to_flow["edge-8"].source == "gateway-7"
    assert bpmn_model.id_to_flow["edge-8"].target == "task-10"
    assert bpmn_model.id_to_flow["edge-14"].source == "task-11"
    assert bpmn_model.id_to_flow["edge-14"].target == "gateway-15"
    assert bpmn_model.id_to_flow["edge-18"].source == "gateway-15"
    assert bpmn_model.id_to_flow["edge-18"].target == "event-19"
    assert bpmn_model.id_to_flow["edge-22"].source == "task-21"
    assert bpmn_model.id_to_flow["edge-22"].target == "gateway-24"
    assert bpmn_model.id_to_flow["edge-27"].source == "task-26"
    assert bpmn_model.id_to_flow["edge-27"].target == "event-28"
    # Check replay to ensure structure
    marking = bpmn_model.get_initial_marking()
    assert marking == {"edge-1"}
    marking = bpmn_model.simulate_execution("task-2", marking)[0]
    assert marking == {"edge-3"}
    marking = bpmn_model.simulate_execution("gateway-4", marking)[0]
    assert marking == {"edge-5", "edge-6"}
    markings = bpmn_model.simulate_execution("gateway-7", marking)
    assert {"edge-6", "edge-8"} in markings
    marking = bpmn_model.simulate_execution("task-10", {"edge-6", "edge-8"})[0]
    marking = bpmn_model.simulate_execution("task-12", marking)[0]
    assert marking == {"edge-13", "edge-16"}
    marking = bpmn_model.simulate_execution("gateway-15", marking)[0]
    marking = bpmn_model.simulate_execution("task-17", marking)[0]
    marking = bpmn_model.simulate_execution("event-19", marking)[0]
    marking = bpmn_model.simulate_execution("task-21", marking)[0]
    assert marking == {"edge-22", "edge-23"}
    marking = bpmn_model.simulate_execution("gateway-24", marking)[0]
    marking = bpmn_model.simulate_execution("task-26", marking)[0]
    assert marking == {"edge-27"}
