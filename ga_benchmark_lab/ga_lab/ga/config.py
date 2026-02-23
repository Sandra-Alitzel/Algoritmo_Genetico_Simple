from dataclasses import dataclass

@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 80
    generations: int = 300
    pc: float = 0.9
    elitism: int = 1
    seed: int = 7

    pm_mode: str = "1/L"   # "1/L" or "manual"
    pm_value: float = 0.01 # used only if pm_mode == "manual"

    decimals: int = 3      # precision >= 0.001