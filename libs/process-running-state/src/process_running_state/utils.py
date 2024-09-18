from pathlib import Path

from lxml import etree

from process_running_state.bpmn_model import BPMNModel, BPMNNodeType


def read_bpmn_model(model_path: Path) -> BPMNModel:
    try:
        # Parse the XML file
        tree = etree.parse(model_path)
        root = tree.getroot()
        # Define the namespace map for BPMN
        ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        # Find all process elements using XPath with namespaces
        processes = root.xpath('.//bpmn:process', namespaces=ns)
        if len(processes) > 1:
            print("Warning! Reading BPMN file with more than one process defined. Parsing first match.")
        bpmn_model = BPMNModel()
        process = processes[0]
        # Process attributes
        # process_id = process.get('id')
        # process_name = process.get('name', 'Unnamed Process')
        # Processing tasks
        tasks = process.xpath('.//bpmn:task', namespaces=ns)
        for task in tasks:
            task_id = task.get("id")
            task_name = task.get("name", task_id)
            bpmn_model.add_task(task_id, task_name)
        # Processing start events
        start_events = process.xpath(".//bpmn:startEvent", namespaces=ns)
        for event in start_events:
            event_id = event.get("id")
            event_name = event.get("name", event_id)
            bpmn_model.add_event(event_type=BPMNNodeType.START_EVENT, event_id=event_id, event_name=event_name)
        # Processing end events
        end_events = process.xpath(".//bpmn:endEvent", namespaces=ns)
        for event in end_events:
            event_id = event.get("id")
            event_name = event.get("name", event_id)
            bpmn_model.add_event(event_type=BPMNNodeType.END_EVENT, event_id=event_id, event_name=event_name)
        # Processing intermediate events
        inter_events = process.xpath('.//bpmn:intermediateThrowEvent | .//bpmn:intermediateCatchEvent', namespaces=ns)
        for event in inter_events:
            event_id = event.get("id")
            event_name = event.get("name", event_id)
            bpmn_model.add_event(event_type=BPMNNodeType.INTERMEDIATE_EVENT, event_id=event_id, event_name=event_name)
        # Processing AND gateways
        and_gateways = process.xpath('.//bpmn:parallelGateway', namespaces=ns)
        for gateway in and_gateways:
            gateway_id = gateway.get("id")
            gateway_name = gateway.get("name", gateway_id)
            bpmn_model.add_gateway(BPMNNodeType.PARALLEL_GATEWAY, gateway_id, gateway_name)
        # Processing XOR gateways
        xor_gateways = process.xpath('.//bpmn:exclusiveGateway', namespaces=ns)
        for gateway in xor_gateways:
            gateway_id = gateway.get("id")
            gateway_name = gateway.get("name", gateway_id)
            bpmn_model.add_gateway(BPMNNodeType.EXCLUSIVE_GATEWAY, gateway_id, gateway_name)
        # Processing OR gateways
        or_gateways = process.xpath('.//bpmn:inclusiveGateway', namespaces=ns)
        for gateway in or_gateways:
            gateway_id = gateway.get("id")
            gateway_name = gateway.get("name", gateway_id)
            bpmn_model.add_gateway(BPMNNodeType.INCLUSIVE_GATEWAY, gateway_id, gateway_name)
        # Processing flows
        flows = process.xpath('.//bpmn:sequenceFlow', namespaces=ns)
        for flow in flows:
            flow_id = flow.get("id")
            flow_name = flow.get("name", flow_id)
            source_id = flow.get("sourceRef")
            target_id = flow.get("targetRef")
            bpmn_model.add_flow(flow_id, flow_name, source_id, target_id)
        # Return build BPMN model
        return bpmn_model
    except etree.XMLSyntaxError as e:
        print(f"XML Syntax Error: {e}")
