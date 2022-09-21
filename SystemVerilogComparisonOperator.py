class SystemVerilogComparisonOperator:
    operator: str = ""

class Equal(SystemVerilogComparisonOperator):
    operator: str = "=="

class EqualDontCare(SystemVerilogComparisonOperator):
    operator: str = "==="

class Less(SystemVerilogComparisonOperator):
    operator: str = "<"

class LessEqual(SystemVerilogComparisonOperator):
    operator: str = "<="

class Greater(SystemVerilogComparisonOperator):
    operator: str = ">"

class GreaterEqual(SystemVerilogComparisonOperator):
    operator: str = ">="
