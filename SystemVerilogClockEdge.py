class SystemVerilogClockEdge:
    edge_type: str = ""

class SystemVerilogPosedge(SystemVerilogClockEdge):
    edge_type: str = "posedge"

class SystemVerilogNegedge(SystemVerilogClockEdge):
    edge_type: str = "negedge"