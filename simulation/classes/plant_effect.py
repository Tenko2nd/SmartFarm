from enum import Enum, auto

class PlantEffect(Enum):
    NONE = auto()
    # Visuals
    BLUE_GRAY_LEAVES = auto() # Drought
    WILTING = auto()          # Drought / Heat
    YELLOWING = auto()        # Asphyxiation / N-Deficiency
    PURPLING = auto()         # P-Deficiency
    TIP_BURN = auto()         # K-Deficiency / Salinity / Heat
    BROWNING = auto()         # Necrosis (Severe stress)
    SPINDLY_GROWTH = auto()   # Etiolation
    PALE_LEAVES = auto()      # Etiolation / N-Deficiency
    MUSH_TEXTURE = auto()     # Frost / Asphyxiation (Root rot)
    # Logic
    GROWTH_STUNTED = auto()

class EffectTrigger:
    def __init__(self, threshold, effect):
        self.threshold = threshold  # 0.0 to 1.0 (StressBuffer.value)
        self.effect = effect        # PlantEffect Enum