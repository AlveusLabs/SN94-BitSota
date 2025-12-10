import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Instruction:
    op: str
    arg1: Optional[str] = None  # memory address like "s0", "v1", "m2"
    arg2: Optional[str] = None  # memory address
    const1: Optional[float] = None  # constant value
    const2: Optional[float] = None  # constant value
    dest: Optional[str] = None  # output address

    def to_dsl(self) -> str:
        """Convert standardized instruction to DSL string"""

        if self.op == "const":
            return f"{self.dest} = {self.const1}"

        elif self.op == "const_vec":
            return f"{self.dest}[{int(self.const1)}] = {self.const2}"

        elif self.op in ["gaussian", "uniform"]:
            return f"{self.dest} = {self.op}({self.const1}, {self.const2})"

        elif self.op in ["add", "sub", "mul", "div"]:
            op_symbol = (
                {"+": "add", "-": "sub", "*": "mul", "/": "div"}[self.op]
                if self.op in ["+", "-", "*", "/"]
                else self.op
            )
            if self.op in ["+", "-", "*", "/"]:
                return f"{self.dest} = {self.arg1} {self.op} {self.arg2}"
            else:
                op_symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[self.op]
                return f"{self.dest} = {self.arg1} {op_symbol} {self.arg2}"

        elif self.op in [
            "abs",
            "exp",
            "log",
            "sin",
            "cos",
            "tan",
            "heaviside",
            "norm",
            "mean",
            "std",
        ]:
            # unary functions
            return f"{self.dest} = {self.op}({self.arg1})"

        elif self.op in ["dot", "outer", "matmul"]:
            # binary functions
            return f"{self.dest} = {self.op}({self.arg1}, {self.arg2})"

        elif self.op == "copy":
            return f"{self.dest} = {self.arg1}"

        else:
            # fallback for complex expressions
            return f"{self.dest} = {self.op}({self.arg1}, {self.arg2})"

    @staticmethod
    def from_dsl(dsl_str: str) -> "Instruction":
        """Parse DSL string into standardized instruction"""
        dsl_str = dsl_str.strip()

        # const scalar: s0 = 0.5
        if re.match(r"[svm]\d+\s*=\s*[\d.-]+$", dsl_str):
            match = re.match(r"([svm]\d+)\s*=\s*([\d.-]+)", dsl_str)
            dest = match.group(1)
            val = float(match.group(2))
            return Instruction("const", dest=dest, const1=val)

        # const vec: v0[3] = 1.2
        elif re.match(r"[vV]\d+\[\d+\]\s*=\s*[\d.-]+$", dsl_str):
            match = re.match(r"([vV]\d+)\[(\d+)\]\s*=\s*([\d.-]+)", dsl_str)
            dest = match.group(1).lower()
            idx = float(match.group(2))
            val = float(match.group(3))
            return Instruction("const_vec", dest=dest, const1=idx, const2=val)

        # copy: s0 = s1
        elif re.match(r"[svm]\d+\s*=\s*[svm]\d+$", dsl_str):
            match = re.match(r"([svm]\d+)\s*=\s*([svm]\d+)", dsl_str)
            dest = match.group(1)
            src = match.group(2)
            return Instruction("copy", arg1=src, dest=dest)

        # arithmetic: s2 = s0 + s1
        elif re.match(r"[svm]\d+\s*=\s*[svm]\d+\s*[\+\-\*/]\s*[svm]\d+", dsl_str):
            match = re.match(
                r"([svm]\d+)\s*=\s*([svm]\d+)\s*([\+\-\*/])\s*([svm]\d+)", dsl_str
            )
            dest = match.group(1)
            arg1 = match.group(2)
            op = match.group(3)
            arg2 = match.group(4)
            return Instruction(op, arg1=arg1, arg2=arg2, dest=dest)

        # function calls: s0 = sin(s1), v1 = dot(v0, v2)
        elif re.match(r"[svm]\d+\s*=\s*\w+\([^)]+\)", dsl_str):
            match = re.match(r"([svm]\d+)\s*=\s*(\w+)\(([^)]+)\)", dsl_str)
            dest = match.group(1)
            op = match.group(2)
            args_str = match.group(3)

            # parse arguments
            args = [arg.strip() for arg in args_str.split(",")]

            # classify operation type
            if op in ["gaussian", "uniform"]:
                # uses constants
                const1 = float(args[0]) if len(args) > 0 else 0
                const2 = float(args[1]) if len(args) > 1 else 1
                return Instruction(op, dest=dest, const1=const1, const2=const2)
            else:
                # uses args
                arg1 = args[0] if len(args) > 0 else None
                arg2 = args[1] if len(args) > 1 else None
                return Instruction(op, arg1=arg1, arg2=arg2, dest=dest)

        else:
            raise ValueError(f"couldn't parse: {dsl_str}")
