from typing import List

import numpy as np

from .instruction import Instruction
from .memory import Memory


class Algorithm:
    def __init__(
        self,
        setup: List[Instruction],
        predict: List[Instruction],
        learn: List[Instruction],
        input_dim: int,
    ):
        self.setup = setup
        self.predict = predict
        self.learn = learn
        self.input_dim = input_dim

    def execute_phase(
        self,
        phase: str,
        mem: Memory,
        inputs: np.ndarray = None,
        label: float = None,
        max_steps: int = 30,
    ):
        """Execute one phase with support for new instruction format"""
        instructions = getattr(self, phase, [])

        # load inputs/labels into conventional locations
        if phase == "predict" and inputs is not None:
            mem.vectors[0][: self.input_dim] = inputs
        if phase == "learn" and label is not None:
            mem.scalars[1] = label

        for i, inst in enumerate(instructions[:max_steps]):
            try:
                # const ops
                if inst.op == "const":
                    addr_type, addr_idx = self._parse_address(inst.dest)
                    if addr_type == "s":
                        mem.scalars[addr_idx] = inst.const1
                    elif addr_type == "v":
                        mem.vectors[addr_idx] = inst.const1
                    elif addr_type == "m":
                        mem.matrices[addr_idx] = inst.const1

                elif inst.op == "const_vec":
                    addr_type, addr_idx = self._parse_address(inst.dest)
                    if addr_type == "v":
                        elem_idx = int(inst.const1)
                        if elem_idx < mem.vectors[addr_idx].shape[0]:
                            mem.vectors[addr_idx][elem_idx] = inst.const2

                # random ops
                elif inst.op == "gaussian":
                    addr_type, addr_idx = self._parse_address(inst.dest)
                    mean, std = inst.const1 or 0, inst.const2 or 1
                    if addr_type == "s":
                        mem.scalars[addr_idx] = np.random.normal(mean, std)
                    elif addr_type == "v":
                        mem.vectors[addr_idx] = np.random.normal(
                            mean, std, mem.vectors[addr_idx].shape
                        )
                    elif addr_type == "m":
                        mem.matrices[addr_idx] = np.random.normal(
                            mean, std, mem.matrices[addr_idx].shape
                        )

                elif inst.op == "uniform":
                    addr_type, addr_idx = self._parse_address(inst.dest)
                    low, high = inst.const1 or -1, inst.const2 or 1
                    if addr_type == "s":
                        mem.scalars[addr_idx] = np.random.uniform(low, high)
                    elif addr_type == "v":
                        mem.vectors[addr_idx] = np.random.uniform(
                            low, high, mem.vectors[addr_idx].shape
                        )
                    elif addr_type == "m":
                        mem.matrices[addr_idx] = np.random.uniform(
                            low, high, mem.matrices[addr_idx].shape
                        )

                # arithmetic ops
                elif inst.op in ["+", "-", "*", "/", "add", "sub", "mul", "div"]:
                    val1 = self._get_value_new_format(inst.arg1, mem)
                    val2 = self._get_value_new_format(inst.arg2, mem)

                    if inst.op in ["+", "add"]:
                        result = val1 + val2
                    elif inst.op in ["-", "sub"]:
                        result = val1 - val2
                    elif inst.op in ["*", "mul"]:
                        result = val1 * val2
                    elif inst.op in ["/", "div"]:
                        result = val1 / (val2 + 1e-8)  # safe divide

                    self._set_value_new_format(inst.dest, result, mem)

                # unary functions
                elif inst.op in ["abs", "exp", "log", "sin", "cos", "tan", "heaviside"]:
                    val = self._get_value_new_format(inst.arg1, mem)

                    if inst.op == "abs":
                        result = np.abs(val)
                    elif inst.op == "exp":
                        result = np.exp(np.clip(val, -10, 10))
                    elif inst.op == "log":
                        result = np.log(np.abs(val) + 1e-8)
                    elif inst.op == "sin":
                        result = np.sin(val)
                    elif inst.op == "cos":
                        result = np.cos(val)
                    elif inst.op == "tan":
                        result = np.tan(val)
                    elif inst.op == "heaviside":
                        result = (val > 0).astype(np.float32)

                    self._set_value_new_format(inst.dest, result, mem)

                # copy operations
                elif inst.op == "copy":
                    val = self._get_value_new_format(inst.arg1, mem)
                    self._set_value_new_format(inst.dest, val, mem)

                # vector/matrix specific ops
                elif inst.op == "dot":
                    v1 = self._get_value_new_format(inst.arg1, mem)
                    v2 = self._get_value_new_format(inst.arg2, mem)
                    result = np.dot(v1, v2)
                    self._set_value_new_format(inst.dest, result, mem)

                elif inst.op == "matmul":
                    m1 = self._get_value_new_format(inst.arg1, mem)
                    m2 = self._get_value_new_format(inst.arg2, mem)
                    result = m1 @ m2
                    self._set_value_new_format(inst.dest, result, mem)

                elif inst.op == "outer":
                    v1 = self._get_value_new_format(inst.arg1, mem)
                    v2 = self._get_value_new_format(inst.arg2, mem)
                    result = np.outer(v1, v2)
                    self._set_value_new_format(inst.dest, result, mem)

                elif inst.op == "norm":
                    val = self._get_value_new_format(inst.arg1, mem)
                    result = np.linalg.norm(val)
                    self._set_value_new_format(inst.dest, result, mem)

                elif inst.op == "mean":
                    val = self._get_value_new_format(inst.arg1, mem)
                    result = np.mean(val)
                    self._set_value_new_format(inst.dest, result, mem)

                elif inst.op == "std":
                    val = self._get_value_new_format(inst.arg1, mem)
                    result = np.std(val)
                    self._set_value_new_format(inst.dest, result, mem)

            except Exception as e:
                # silent fail, just skip bad instructions
                pass

    def _parse_address(self, addr: str) -> tuple:
        """Parse address like 'v1' into (type, index)"""
        if not addr or not isinstance(addr, str):
            return None, None
        addr_type = addr[0]  # 'v', 's', 'm'
        addr_idx = int(addr[1:])
        return addr_type, addr_idx

    def _get_value_new_format(self, addr: str, mem: Memory):
        """Get value from memory using new address format"""
        if not addr:
            return 0.0
        addr_type, addr_idx = self._parse_address(addr)
        if addr_type == "s":
            return mem.scalars[addr_idx]
        elif addr_type == "v":
            return mem.vectors[addr_idx]
        elif addr_type == "m":
            return mem.matrices[addr_idx]
        return 0.0

    def _set_value_new_format(self, addr: str, value, mem: Memory):
        """Set value in memory using new address format"""
        if not addr:
            return
        addr_type, addr_idx = self._parse_address(addr)
        if addr_type == "s":
            mem.scalars[addr_idx] = value
        elif addr_type == "v":
            mem.vectors[addr_idx] = value
        elif addr_type == "m":
            mem.matrices[addr_idx] = value

    def to_dsl(self) -> str:
        lines = []
        if self.setup:
            lines.append("# setup")
            lines.extend([inst.to_dsl() for inst in self.setup])
        if self.predict:
            lines.append("# predict")
            lines.extend([inst.to_dsl() for inst in self.predict])
        if self.learn:
            lines.append("# learn")
            lines.extend([inst.to_dsl() for inst in self.learn])
        return "\n".join(lines)

    @staticmethod
    def from_dsl(dsl_str: str, input_dim: int, verbose: bool = False) -> "Algorithm":
        lines = [l.strip() for l in dsl_str.strip().split("\n") if l.strip()]

        setup, predict, learn = [], [], []
        current_phase = None

        for line in lines:
            if line.startswith("#"):
                if "setup" in line.lower():
                    current_phase = "setup"
                elif "predict" in line.lower():
                    current_phase = "predict"
                elif "learn" in line.lower():
                    current_phase = "learn"
                continue

            try:
                inst = Instruction.from_dsl(line)
                if current_phase == "setup":
                    setup.append(inst)
                elif current_phase == "predict":
                    predict.append(inst)
                elif current_phase == "learn":
                    learn.append(inst)
                else:
                    # default to predict if no phase specified
                    predict.append(inst)
            except Exception as e:
                if verbose:
                    print(f"    ❌ failed to parse: {line} ({e})")

        return Algorithm(setup, predict, learn, input_dim)


def create_initial_algorithm(input_dim: int) -> Algorithm:
    """create a simple random guesser as starting point"""
    setup = [
        Instruction.from_dsl("m0 = gaussian(0, 0.1)"),  # random weight matrix
        Instruction.from_dsl("s10 = 0.01"),  # might be used as learning rate
    ]

    predict = [
        # just random prediction for now
        Instruction.from_dsl("s0 = uniform(0, 1)"),
    ]

    learn = [
        # empty - let evolution discover learning
    ]

    return Algorithm(setup, predict, learn, input_dim)


def create_ops_summary(task_type: str) -> str:
    """Create operations summary based on task type"""
    base_ops = """Available ops:
Arithmetic: s2=s0+s1, s2=s0-s1, s2=s0*s1, s2=s0/s1 (also works for vectors/matrices)
Functions: abs, exp, log, sin, cos, tan, heaviside
Random: uniform(low,high), gaussian(mean,std)
Constants: s0=0.5, v0[3]=1.2, m0[1,2]=-0.3
LinAlg: dot(v1,v2), matmul(m1,v2), outer(v1,v2), norm(v1)
Stats: mean(v1), std(v1)
Memory: s0-s19 (scalars), v0-v9 (vectors), m0-m4 (matrices)
Convention: v0=input, s0=output, s1=label"""

    if task_type == "classification":
        return base_ops + "\nNote: For classification, output s0 should be in [0,1]"
    else:
        return base_ops + "\nNote: For regression, output s0 can be any real value"
