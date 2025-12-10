from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

# Opcode definitions
OPCODES = {
    "NOOP": 0,
    "CONST": 1,
    "CONST_VEC": 2,
    "ADD": 3,
    "SUB": 4,
    "MUL": 5,
    "DIV": 6,
    "ABS": 7,
    "EXP": 8,
    "LOG": 9,
    "SIN": 10,
    "COS": 11,
    "TAN": 12,
    "HEAVISIDE": 13,
    "GAUSSIAN": 14,
    "UNIFORM": 15,
    "DOT": 16,
    "MATMUL": 17,
    "OUTER": 18,
    "NORM": 19,
    "MEAN": 20,
    "STD": 21,
    "COPY": 22,
}

# Address space encoding
ADDR_SCALARS = 0  # s0-s19 → 0-19
ADDR_VECTORS = 20  # v0-v9 → 20-29
ADDR_MATRICES = 30  # m0-m4 → 30-34
ADDR_INPUT = 20  # v0 is input vector
ADDR_OUTPUT = 0  # s0 is output scalar

OPCODE_METADATA = {
    "NOOP": {"arg1": "none", "arg2": "none", "dest": "none"},
    "CONST": {"arg1": "none", "arg2": "none", "dest": "s"},
    "CONST_VEC": {"arg1": "none", "arg2": "none", "dest": "v"},
    "ADD": {"arg1": "s", "arg2": "s", "dest": "s"},
    "SUB": {"arg1": "s", "arg2": "s", "dest": "s"},
    "MUL": {"arg1": "s", "arg2": "s", "dest": "s"},
    "DIV": {"arg1": "s", "arg2": "s", "dest": "s"},
    "ABS": {"arg1": "s", "arg2": "none", "dest": "s"},
    "EXP": {"arg1": "s", "arg2": "none", "dest": "s"},
    "LOG": {"arg1": "s", "arg2": "none", "dest": "s"},
    "SIN": {"arg1": "s", "arg2": "none", "dest": "s"},
    "COS": {"arg1": "s", "arg2": "none", "dest": "s"},
    "TAN": {"arg1": "s", "arg2": "none", "dest": "s"},
    "HEAVISIDE": {"arg1": "s", "arg2": "none", "dest": "s"},
    "GAUSSIAN": {"arg1": "none", "arg2": "none", "dest": "s"},
    "UNIFORM": {"arg1": "none", "arg2": "none", "dest": "s"},
    "DOT": {"arg1": "v", "arg2": "v", "dest": "s"},
    "MATMUL": {"arg1": "m", "arg2": "v", "dest": "v"},
    "OUTER": {"arg1": "v", "arg2": "v", "dest": "m"},
    "NORM": {"arg1": "v", "arg2": "none", "dest": "s"},
    "MEAN": {"arg1": "v", "arg2": "none", "dest": "s"},
    "STD": {"arg1": "v", "arg2": "none", "dest": "s"},
    "COPY": {"arg1": "s", "arg2": "none", "dest": "s"},
}


@dataclass
class AlgorithmArray:
    """Dynamic phase-based array representation for genetic programs"""

    # Required fields (no defaults)
    input_dim: int

    # Dynamic storage for all phases
    phase_arrays: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    phase_sizes: Dict[str, int] = field(default_factory=dict)
    phase_max_sizes: Dict[str, int] = field(default_factory=dict)

    # Memory layout with defaults
    scalar_count: int = 20
    vector_count: int = 10
    matrix_count: int = 5
    vector_dim: int = None  # derived from input_dim

    def __post_init__(self):
        if self.vector_dim is None:
            self.vector_dim = self.input_dim + 10  # Extra space for derived features

    @classmethod
    def create_empty(
        cls,
        input_dim: int,
        phases: List[str],
        max_sizes: Dict[str, int],
        *,
        scalar_count: Optional[int] = None,
        vector_count: Optional[int] = None,
        matrix_count: Optional[int] = None,
        vector_dim: Optional[int] = None,
    ) -> "AlgorithmArray":
        """Create empty algorithm with dynamic phases"""
        instance = cls(
            input_dim=input_dim,
            phase_arrays={},
            phase_sizes={},
            phase_max_sizes={},
            scalar_count=scalar_count if scalar_count is not None else cls.scalar_count,
            vector_count=vector_count if vector_count is not None else cls.vector_count,
            matrix_count=matrix_count if matrix_count is not None else cls.matrix_count,
            vector_dim=vector_dim,
        )

        # Initialize arrays for each phase
        for phase in phases:
            max_size = max_sizes.get(phase, 50)  # Default 50 if not specified
            instance.phase_arrays[phase] = {
                "ops": np.zeros(max_size, dtype=np.int8),
                "arg1": np.zeros(max_size, dtype=np.int16),
                "arg2": np.zeros(max_size, dtype=np.int16),
                "dest": np.zeros(max_size, dtype=np.int16),
                "const1": np.zeros(max_size, dtype=np.float32),
                "const2": np.zeros(max_size, dtype=np.float32),
            }
            instance.phase_sizes[phase] = 0
            instance.phase_max_sizes[phase] = max_size

        return instance

    @classmethod
    def from_algorithm(cls, algorithm) -> "AlgorithmArray":
        """Convert from old Algorithm format to array format"""
        # This will be implemented after we create the array-based system
        pass

    def add_instruction(
        self,
        phase: str,
        op: str,
        arg1: int = 0,
        arg2: int = 0,
        dest: int = 0,
        const1: float = 0.0,
        const2: float = 0.0,
    ) -> None:
        """Add instruction to specified phase"""
        if phase not in self.phase_arrays:
            raise ValueError(
                f"Phase '{phase}' not found. Available phases: {list(self.phase_arrays.keys())}"
            )

        op_code = OPCODES[op.upper()]

        if self.phase_sizes[phase] >= self.phase_max_sizes[phase]:
            raise ValueError(
                f"Phase '{phase}' is full (max {self.phase_max_sizes[phase]} instructions)"
            )

        arrays = self.phase_arrays[phase]
        idx = self.phase_sizes[phase]
        self.phase_sizes[phase] += 1

        arrays["ops"][idx] = op_code
        arrays["arg1"][idx] = arg1
        arrays["arg2"][idx] = arg2
        arrays["dest"][idx] = dest
        arrays["const1"][idx] = const1
        arrays["const2"][idx] = const2

    def get_phase_ops(
        self, phase: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get instruction arrays for a specific phase"""
        if phase not in self.phase_arrays:
            raise ValueError(
                f"Phase '{phase}' not found. Available phases: {list(self.phase_arrays.keys())}"
            )

        arrays = self.phase_arrays[phase]
        size = self.phase_sizes[phase]

        return (
            arrays["ops"][:size],
            arrays["arg1"][:size],
            arrays["arg2"][:size],
            arrays["dest"][:size],
            arrays["const1"][:size],
            arrays["const2"][:size],
        )

    def get_phases(self) -> List[str]:
        """Get list of available phases"""
        return list(self.phase_arrays.keys())

    def get_phase_size(self, phase: str) -> int:
        """Get current size of a phase"""
        if phase not in self.phase_arrays:
            raise ValueError(f"Phase '{phase}' not found")
        return self.phase_sizes[phase]

    def get_phase_max_size(self, phase: str) -> int:
        """Get maximum size of a phase"""
        if phase not in self.phase_arrays:
            raise ValueError(f"Phase '{phase}' not found")
        return self.phase_max_sizes[phase]

    def validate_addresses(self) -> List[str]:
        """Validate all addresses are within bounds"""
        errors = []
        max_addr = ADDR_MATRICES + self.matrix_count - 1

        for phase in self.phase_arrays:
            arrays = self.phase_arrays[phase]
            size = self.phase_sizes[phase]

            for i in range(size):
                op = arrays["ops"][i]
                dest = arrays["dest"][i]
                arg1 = arrays["arg1"][i]
                arg2 = arrays["arg2"][i]

                # Check destination
                if dest < 0 or dest > max_addr:
                    errors.append(f"{phase} instruction {i}: dest {dest} out of bounds")

                # Check arguments (skip for NOOP/CONST)
                if op not in [OPCODES["NOOP"], OPCODES["CONST"], OPCODES["CONST_VEC"]]:
                    if arg1 < 0 or arg1 > max_addr:
                        errors.append(
                            f"{phase} instruction {i}: arg1 {arg1} out of bounds"
                        )
                    if (
                        arg2 < 0
                        or arg2 > max_addr
                        and op not in [OPCODES["ABS"], OPCODES["EXP"], OPCODES["LOG"]]
                    ):
                        errors.append(
                            f"{phase} instruction {i}: arg2 {arg2} out of bounds"
                        )

        return errors

    def validate_semantics(self) -> List[str]:
        """Validate type semantics of all instructions."""
        errors = []
        opcode_map = {v: k for k, v in OPCODES.items()}

        for phase in self.get_phases():
            ops, arg1s, arg2s, dests, _, _ = self.get_phase_ops(phase)
            for i in range(len(ops)):
                op_name = opcode_map.get(ops[i])
                if not op_name or op_name not in OPCODE_METADATA:
                    continue

                meta = OPCODE_METADATA[op_name]

                def get_type(addr):
                    if addr < ADDR_VECTORS:
                        return "s"
                    if addr < ADDR_MATRICES:
                        return "v"
                    return "m"

                # Check arg1
                if meta["arg1"] != "none" and meta["arg1"] != get_type(arg1s[i]):
                    errors.append(
                        f"Phase {phase}, op {i} ({op_name}): arg1 type mismatch"
                    )
                # Check arg2
                if meta["arg2"] != "none" and meta["arg2"] != get_type(arg2s[i]):
                    errors.append(
                        f"Phase {phase}, op {i} ({op_name}): arg2 type mismatch"
                    )
                # Check dest
                if meta["dest"] != "none" and meta["dest"] != get_type(dests[i]):
                    errors.append(
                        f"Phase {phase}, op {i} ({op_name}): dest type mismatch"
                    )
        return errors

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        result = {"input_dim": self.input_dim, "phases": {}}

        for phase in self.phase_arrays:
            size = self.phase_sizes[phase]
            result["phases"][phase] = {
                "ops": self.phase_arrays[phase]["ops"][:size].tolist(),
                "arg1": self.phase_arrays[phase]["arg1"][:size].tolist(),
                "arg2": self.phase_arrays[phase]["arg2"][:size].tolist(),
                "dest": self.phase_arrays[phase]["dest"][:size].tolist(),
                "const1": self.phase_arrays[phase]["const1"][:size].tolist(),
                "const2": self.phase_arrays[phase]["const2"][:size].tolist(),
                "size": size,
                "max_size": self.phase_max_sizes[phase],
            }

        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "AlgorithmArray":
        """Deserialize from dictionary"""
        phases = list(data["phases"].keys())
        max_sizes = {phase: data["phases"][phase]["max_size"] for phase in phases}

        instance = cls(
            input_dim=data["input_dim"],
            phase_arrays={},
            phase_sizes={},
            phase_max_sizes={},
        )

        for phase in phases:
            max_size = max_sizes[phase]
            phase_data = data["phases"][phase]

            instance.phase_arrays[phase] = {
                "ops": np.zeros(max_size, dtype=np.int8),
                "arg1": np.zeros(max_size, dtype=np.int16),
                "arg2": np.zeros(max_size, dtype=np.int16),
                "dest": np.zeros(max_size, dtype=np.int16),
                "const1": np.zeros(max_size, dtype=np.float32),
                "const2": np.zeros(max_size, dtype=np.float32),
            }
            instance.phase_max_sizes[phase] = max_size

            # Load data
            size = len(phase_data["ops"])
            instance.phase_arrays[phase]["ops"][:size] = phase_data["ops"]
            instance.phase_arrays[phase]["arg1"][:size] = phase_data["arg1"]
            instance.phase_arrays[phase]["arg2"][:size] = phase_data["arg2"]
            instance.phase_arrays[phase]["dest"][:size] = phase_data["dest"]
            instance.phase_arrays[phase]["const1"][:size] = phase_data["const1"]
            instance.phase_arrays[phase]["const2"][:size] = phase_data["const2"]
            instance.phase_sizes[phase] = size

        return instance

    def fingerprint(self) -> str:
        """Compute a stable fingerprint for caching identical programs."""
        import hashlib
        hasher = hashlib.sha1()
        hasher.update(int(self.input_dim).to_bytes(4, "little", signed=True))
        hasher.update(int(self.scalar_count).to_bytes(2, "little", signed=True))
        hasher.update(int(self.vector_count).to_bytes(2, "little", signed=True))
        hasher.update(int(self.matrix_count).to_bytes(2, "little", signed=True))
        hasher.update(int(self.vector_dim).to_bytes(4, "little", signed=True))

        for phase in sorted(self.phase_arrays.keys()):
            arrays = self.phase_arrays[phase]
            size = self.phase_sizes[phase]
            hasher.update(phase.encode("utf-8"))
            hasher.update(arrays["ops"][:size].tobytes())
            hasher.update(arrays["arg1"][:size].tobytes())
            hasher.update(arrays["arg2"][:size].tobytes())
            hasher.update(arrays["dest"][:size].tobytes())
            hasher.update(arrays["const1"][:size].tobytes())
            hasher.update(arrays["const2"][:size].tobytes())

        return hasher.hexdigest()
