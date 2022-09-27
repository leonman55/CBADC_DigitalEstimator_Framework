class SystemVerilogComparisonOperator:
    operator: str = ""
    plain_text: str = ""

class Equal(SystemVerilogComparisonOperator):
    operator: str = "=="
    plain_text: str = "equal to"

class EqualDontCare(SystemVerilogComparisonOperator):
    operator: str = "==="
    plain_text: str = "equal to"

class Less(SystemVerilogComparisonOperator):
    operator: str = "<"
    plain_text: str = "less than"

class LessEqual(SystemVerilogComparisonOperator):
    operator: str = "<="
    plain_text: str = "less than or equal to"

class Greater(SystemVerilogComparisonOperator):
    operator: str = ">"
    plain_text: str = "greater than"

class GreaterEqual(SystemVerilogComparisonOperator):
    operator: str = ">="
    plain_text: str = "greater than or equal to"
