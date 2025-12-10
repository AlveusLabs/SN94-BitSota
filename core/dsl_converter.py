from .algorithm_array import (
    AlgorithmArray,
    OPCODES,
    ADDR_SCALARS,
    ADDR_VECTORS,
    ADDR_MATRICES,
)
from .instruction import Instruction


class DSLConverter:
    """Convert between human-readable DSL and array format"""

    @staticmethod
    def parse_address(addr: str) -> int:
        """Convert address string to integer index"""
        if not addr:
            return -1

        addr = addr.lower().strip()

        if addr.startswith("s"):
            return ADDR_SCALARS + int(addr[1:])
        elif addr.startswith("v"):
            return ADDR_VECTORS + int(addr[1:])
        elif addr.startswith("m"):
            return ADDR_MATRICES + int(addr[1:])
        else:
            raise ValueError(f"Invalid address format: {addr}")

    @staticmethod
    def address_to_string(addr: int) -> str:
        """Convert integer index back to address string"""
        if addr < 0:
            return "none"
        elif addr < ADDR_VECTORS:
            return f"s{addr - ADDR_SCALARS}"
        elif addr < ADDR_MATRICES:
            return f"v{addr - ADDR_VECTORS}"
        else:
            return f"m{addr - ADDR_MATRICES}"

    @classmethod
    def from_algorithm(cls, algorithm) -> AlgorithmArray:
        """Convert from old Algorithm format to AlgorithmArray"""
        total_ops = len(algorithm.setup) + len(algorithm.predict) + len(algorithm.learn)
        array_algo = AlgorithmArray.create_empty(
            algorithm.input_dim, max(total_ops, 10)
        )

        # Process each phase
        for phase, instructions in [
            ("setup", algorithm.setup),
            ("predict", algorithm.predict),
            ("learn", algorithm.learn),
        ]:
            for inst in instructions:
                cls._add_instruction(array_algo, phase, inst)

        return array_algo

    @classmethod
    def _add_instruction(
        cls, array_algo: AlgorithmArray, phase: str, inst: Instruction
    ) -> None:
        """Add a single instruction to the array"""
        op_map = {
            "const": "CONST",
            "const_vec": "CONST_VEC",
            "add": "ADD",
            "sub": "SUB",
            "mul": "MUL",
            "div": "DIV",
            "abs": "ABS",
            "exp": "EXP",
            "log": "LOG",
            "sin": "SIN",
            "cos": "COS",
            "tan": "TAN",
            "heaviside": "HEAVISIDE",
            "gaussian": "GAUSSIAN",
            "uniform": "UNIFORM",
            "dot": "DOT",
            "matmul": "MATMUL",
            "outer": "OUTER",
            "norm": "NORM",
            "mean": "MEAN",
            "std": "STD",
            "copy": "COPY",
            "+": "ADD",
            "-": "SUB",
            "*": "MUL",
            "/": "DIV",
        }

        op = op_map.get(inst.op.lower(), "NOOP")

        # Parse addresses
        arg1_addr = cls.parse_address(inst.arg1) if inst.arg1 else -1
        arg2_addr = cls.parse_address(inst.arg2) if inst.arg2 else -1
        dest_addr = cls.parse_address(inst.dest) if inst.dest else -1

        # Handle special cases
        if inst.op == "const":
            array_algo.add_instruction(
                phase, "CONST", -1, -1, dest_addr, inst.const1 or 0.0, 0.0
            )
        elif inst.op == "const_vec":
            array_algo.add_instruction(
                phase,
                "CONST_VEC",
                -1,
                -1,
                dest_addr,
                inst.const1 or 0.0,
                inst.const2 or 0.0,
            )
        elif inst.op in ["gaussian", "uniform"]:
            array_algo.add_instruction(
                phase,
                op.upper(),
                -1,
                -1,
                dest_addr,
                inst.const1 or 0.0,
                inst.const2 or 1.0,
            )
        else:
            array_algo.add_instruction(
                phase,
                op.upper(),
                arg1_addr,
                arg2_addr,
                dest_addr,
                inst.const1 or 0.0,
                inst.const2 or 0.0,
            )

    @classmethod
    def to_dsl(cls, array_algo: AlgorithmArray) -> str:
        """Convert array format back to human-readable DSL"""
        lines = []

        # Helper to format instruction
        def format_instruction(
            op_code: int, arg1: int, arg2: int, dest: int, const1: float, const2: float
        ) -> str:
            op_name = next(name for name, code in OPCODES.items() if code == op_code)

            if op_name == "CONST":
                dest_str = cls.address_to_string(dest)
                return f"{dest_str} = {const1}"

            elif op_name == "CONST_VEC":
                dest_str = cls.address_to_string(dest)
                idx = int(const1)
                val = const2
                return f"{dest_str}[{idx}] = {val}"

            elif op_name in ["GAUSSIAN", "UNIFORM"]:
                dest_str = cls.address_to_string(dest)
                return f"{dest_str} = {op_name.lower()}({const1}, {const2})"

            elif op_name in ["ADD", "SUB", "MUL", "DIV"]:
                dest_str = cls.address_to_string(dest)
                arg1_str = cls.address_to_string(arg1)
                arg2_str = cls.address_to_string(arg2)
                symbol = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/"}[op_name]
                return f"{dest_str} = {arg1_str} {symbol} {arg2_str}"

            elif op_name in [
                "ABS",
                "EXP",
                "LOG",
                "SIN",
                "COS",
                "TAN",
                "HEAVISIDE",
                "NORM",
                "MEAN",
                "STD",
            ]:
                dest_str = cls.address_to_string(dest)
                arg1_str = cls.address_to_string(arg1)
                return f"{dest_str} = {op_name.lower()}({arg1_str})"

            elif op_name in ["DOT", "MATMUL", "OUTER"]:
                dest_str = cls.address_to_string(dest)
                arg1_str = cls.address_to_string(arg1)
                arg2_str = cls.address_to_string(arg2)
                return f"{dest_str} = {op_name.lower()}({arg1_str}, {arg2_str})"

            elif op_name == "COPY":
                dest_str = cls.address_to_string(dest)
                arg1_str = cls.address_to_string(arg1)
                return f"{dest_str} = {arg1_str}"

            else:
                return f"# Unknown operation: {op_name}"

        # Setup phase
        if array_algo.setup_end > 0:
            lines.append("# setup")
            ops, arg1, arg2, dest, const1, const2 = array_algo.get_phase_ops("setup")
            for i in range(len(ops)):
                if ops[i] != 0:  # Skip NOOP
                    lines.append(
                        format_instruction(
                            ops[i], arg1[i], arg2[i], dest[i], const1[i], const2[i]
                        )
                    )

        # Predict phase
        if array_algo.predict_end > array_algo.setup_end:
            lines.append("# predict")
            ops, arg1, arg2, dest, const1, const2 = array_algo.get_phase_ops("predict")
            for i in range(len(ops)):
                if ops[i] != 0:  # Skip NOOP
                    lines.append(
                        format_instruction(
                            ops[i], arg1[i], arg2[i], dest[i], const1[i], const2[i]
                        )
                    )

        # Learn phase
        if array_algo.learn_end > array_algo.predict_end:
            lines.append("# learn")
            ops, arg1, arg2, dest, const1, const2 = array_algo.get_phase_ops("learn")
            for i in range(len(ops)):
                if ops[i] != 0:  # Skip NOOP
                    lines.append(
                        format_instruction(
                            ops[i], arg1[i], arg2[i], dest[i], const1[i], const2[i]
                        )
                    )

        return "\n".join(lines)

    @classmethod
    def from_dsl(cls, dsl_str: str, input_dim: int) -> AlgorithmArray:
        """Parse DSL string directly into AlgorithmArray"""

        # Use existing parser
