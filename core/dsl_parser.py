import re
from typing import Optional

from .algorithm_array import (
    AlgorithmArray,
    OPCODES,
    ADDR_SCALARS,
    ADDR_VECTORS,
    ADDR_MATRICES,
)


class DSLParser:
    """Parse DSL strings directly into AlgorithmArray format"""

    _NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    _DEFAULT_PHASE_MAX_SIZES = {
        "setup": 30,
        "predict": 30,
        "learn": 30,
    }

    @staticmethod
    def parse_address(addr: str) -> int:
        """Convert address string to integer index"""
        if addr is None:
            return -1

        if not isinstance(addr, str):
            addr = str(addr)

        if not addr:
            return -1

        addr = addr.lower().strip()

        if addr in {"none", "null", "-1"}:
            return -1

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
    def from_dsl(cls, dsl_str: str, input_dim: int) -> AlgorithmArray:
        """Parse DSL string into AlgorithmArray"""
        raw_lines = dsl_str.splitlines()
        lines: list[str] = []
        for raw in raw_lines:
            line = raw.strip()
            if not line:
                continue
            # Support inline comments: "s3 = 0.02  # learning rate".
            # Full-line comments (including phase headers and meta) are preserved.
            if not line.startswith("#") and "#" in line:
                line = line.split("#", 1)[0].rstrip()
                if not line:
                    continue
            lines.append(line)

        meta = {}
        for line in lines:
            if not line.lower().startswith("# meta:"):
                continue
            payload = line.split(":", 1)[1].strip() if ":" in line else ""
            # Accept "k=v" tokens separated by whitespace or commas.
            for token in re.split(r"[\s,]+", payload):
                if not token or "=" not in token:
                    continue
                key, value = token.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if not key or not value:
                    continue
                meta[key] = value

        phases = ["setup", "predict", "learn"]
        phase_counts = {phase: 0 for phase in phases}
        current_phase = "predict"
        for line in lines:
            line_lower = line.lower()

            if line.startswith("#") or line.endswith(":"):
                if "setup" in line_lower:
                    current_phase = "setup"
                elif "predict" in line_lower:
                    current_phase = "predict"
                elif "learn" in line_lower:
                    current_phase = "learn"
                continue

            phase_counts[current_phase] += 1

        def _meta_int(name: str):
            raw = meta.get(name)
            if raw is None:
                return None
            try:
                return int(raw)
            except Exception:
                return None

        def _meta_phase_max(phase: str) -> Optional[int]:
            for key in (
                f"{phase}_max_ops",
                f"{phase}_max",
                f"{phase}_max_size",
                f"{phase}_max_instructions",
            ):
                value = _meta_int(key)
                if value is not None:
                    return max(0, value)
            return None

        max_sizes = {}
        for phase in phases:
            meta_max = _meta_phase_max(phase)
            default_max = cls._DEFAULT_PHASE_MAX_SIZES.get(phase, 0)
            if meta_max is None:
                max_sizes[phase] = max(default_max, phase_counts.get(phase, 0))
            else:
                max_sizes[phase] = max(meta_max, phase_counts.get(phase, 0))

        meta_vector_dim = _meta_int("vector_dim")
        if meta_vector_dim is None:
            meta_vector_dim = int(input_dim)
        else:
            meta_vector_dim = max(int(meta_vector_dim), int(input_dim))

        array_algo = AlgorithmArray.create_empty(
            input_dim,
            phases,
            max_sizes,
            scalar_count=_meta_int("scalar_count"),
            vector_count=_meta_int("vector_count"),
            matrix_count=_meta_int("matrix_count"),
            vector_dim=meta_vector_dim,
        )

        current_phase = "predict"

        for line in lines:
            line_lower = line.lower()

            if line.startswith("#") or line.endswith(":"):
                if "setup" in line_lower:
                    current_phase = "setup"
                elif "predict" in line_lower:
                    current_phase = "predict"
                elif "learn" in line_lower:
                    current_phase = "learn"
                continue

            try:
                cls._parse_line(array_algo, current_phase, line)
            except Exception as e:
                print(f"Warning: Could not parse line '{line}': {e}")

        return array_algo

    @classmethod
    def _parse_line(cls, array_algo: AlgorithmArray, phase: str, line: str) -> None:
        """Parse a single DSL line and add to array"""

        # Arrow format: CONST 0.5 -> s0 or CONST_VEC -> v1
        if "->" in line:
            parts = line.split("->")
            if len(parts) == 2:
                op_part = parts[0].strip().upper()
                dest_part = parts[1].strip()
                dest = cls.parse_address(dest_part)

                op_tokens = op_part.split()
                op_name = op_tokens[0]

                if op_name == "CONST" and len(op_tokens) == 2:
                    val = float(op_tokens[1])
                    array_algo.add_instruction(phase, "CONST", -1, -1, dest, val, 0.0)
                    return
                elif op_name == "CONST_VEC":
                    array_algo.add_instruction(phase, "CONST_VEC", -1, -1, dest, 0.0, 0.0)
                    return
                elif op_name in OPCODES:
                    args = [cls.parse_address(t) for t in op_tokens[1:] if t.startswith(('s', 'v', 'm'))]
                    arg1 = args[0] if len(args) > 0 else -1
                    arg2 = args[1] if len(args) > 1 else -1
                    array_algo.add_instruction(phase, op_name, arg1, arg2, dest, 0.0, 0.0)
                    return

        # const scalar: s0 = 0.5
        if re.match(rf"[svm]\d+\s*=\s*{cls._NUMBER_RE}$", line, flags=re.IGNORECASE):
            match = re.match(
                rf"([svm]\d+)\s*=\s*({cls._NUMBER_RE})$",
                line,
                flags=re.IGNORECASE,
            )
            dest = cls.parse_address(match.group(1))
            val = float(match.group(2))
            array_algo.add_instruction(phase, "CONST", -1, -1, dest, val, 0.0)

        # const vec: v0[3] = 1.2
        elif re.match(rf"[vV]\d+\[\d+\]\s*=\s*{cls._NUMBER_RE}$", line, flags=re.IGNORECASE):
            match = re.match(
                rf"([vV]\d+)\[(\d+)\]\s*=\s*({cls._NUMBER_RE})$",
                line,
                flags=re.IGNORECASE,
            )
            dest = cls.parse_address(match.group(1))
            idx = float(match.group(2))
            val = float(match.group(3))
            array_algo.add_instruction(phase, "CONST_VEC", -1, -1, dest, idx, val)

        # copy: s0 = s1
        elif re.match(r"[svm]\d+\s*=\s*[svm]\d+$", line):
            match = re.match(r"([svm]\d+)\s*=\s*([svm]\d+)", line)
            dest = cls.parse_address(match.group(1))
            src = cls.parse_address(match.group(2))
            array_algo.add_instruction(phase, "COPY", src, -1, dest, 0.0, 0.0)

        # arithmetic: s2 = s0 + s1
        elif re.match(r"[svm]\d+\s*=\s*[svm]\d+\s*[\+\-\*/]\s*[svm]\d+", line):
            match = re.match(
                r"([svm]\d+)\s*=\s*([svm]\d+)\s*([\+\-\*/])\s*([svm]\d+)", line
            )
            dest = cls.parse_address(match.group(1))
            arg1 = cls.parse_address(match.group(2))
            op = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}[match.group(3)]
            arg2 = cls.parse_address(match.group(4))
            array_algo.add_instruction(phase, op, arg1, arg2, dest, 0.0, 0.0)

        # function calls: s0 = sin(s1), v1 = dot(v0, v2)
        elif re.match(r"[svm]\d+\s*=\s*\w+\([^)]+\)", line):
            match = re.match(r"([svm]\d+)\s*=\s*(\w+)\(([^)]+)\)", line)
            dest = cls.parse_address(match.group(1))
            op = match.group(2).upper()
            args_str = match.group(3)

            # parse arguments
            args = [arg.strip() for arg in args_str.split(",")]

            if op in ["GAUSSIAN", "UNIFORM"]:
                # uses constants
                const1 = float(args[0]) if len(args) > 0 else 0
                const2 = float(args[1]) if len(args) > 1 else 1
                array_algo.add_instruction(phase, op, -1, -1, dest, const1, const2)
            else:
                # uses args
                arg1 = cls.parse_address(args[0]) if len(args) > 0 else -1
                arg2 = cls.parse_address(args[1]) if len(args) > 1 else -1
                array_algo.add_instruction(phase, op, arg1, arg2, dest, 0.0, 0.0)

    @classmethod
    def to_dsl(cls, array_algo: AlgorithmArray) -> str:
        """Convert AlgorithmArray back to DSL string"""
        phase_meta = []
        for phase in array_algo.get_phases():
            try:
                phase_max = array_algo.get_phase_max_size(phase)
            except Exception:
                continue
            phase_meta.append(f"{phase}_max_ops={int(phase_max)}")

        phase_meta_txt = ""
        if phase_meta:
            phase_meta_txt = " " + " ".join(phase_meta)

        lines = [
            (
                "# meta:"
                f" scalar_count={int(array_algo.scalar_count)}"
                f" vector_count={int(array_algo.vector_count)}"
                f" matrix_count={int(array_algo.matrix_count)}"
                f" vector_dim={int(array_algo.vector_dim)}"
                f"{phase_meta_txt}"
            )
        ]

        # Helper to format instruction
        def format_instruction(
            op_code: int, arg1: int, arg2: int, dest: int, const1: float, const2: float
        ) -> str:
            op_name = next(
                (name for name, code in OPCODES.items() if code == op_code), "UNKNOWN"
            )

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

        for phase in array_algo.get_phases():
            lines.append(f"# {phase}")
            ops, arg1, arg2, dest, const1, const2 = array_algo.get_phase_ops(phase)
            for i in range(len(ops)):
                if ops[i] != 0:  # Skip NOOP
                    lines.append(
                        format_instruction(
                            ops[i], arg1[i], arg2[i], dest[i], const1[i], const2[i]
                        )
                    )

        return "\n".join(lines)
