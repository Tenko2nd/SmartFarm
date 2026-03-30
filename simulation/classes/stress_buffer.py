import random
import simulation.utils.constants as C

class StressBuffer:
    def __init__(self, name, days_to_effect, recovery_days, base_resilience = 0.1, threshold_visible_effect = 0.5, is_fatal=False, effects=None):
        self.name = name
        self.value = 0.0  # 0.0 to 1.0
        self.threshold_effect = max(0.0, min(1.0, threshold_visible_effect))

        if isinstance(days_to_effect, tuple):
            chosen_days = random.uniform(days_to_effect[0], days_to_effect[1])
        else:
            chosen_days = days_to_effect
        self.ticks_to_fill = chosen_days * C.TICKS_DAY
        self.ticks_to_recover = recovery_days * C.TICKS_DAY

        self.resilience = base_resilience + (random.uniform(-0.05, 0.05))

        self.recovery_rate = 1.0 / self.ticks_to_recover if recovery_days > 0 else 0
        self.is_fatal = is_fatal  # If self.value = 1.0, does the plant die immediately?

    def update(self, stress_intensity):
        # Calculate how much the environment exceeds the plant's natural tolerance
        net_stress = stress_intensity - self.resilience

        if net_stress > 0:
            # The plant is actively taking damage.
            increment = net_stress / self.ticks_to_fill
            self.value = min(1.0, self.value + increment)
        else:
            # The environment is within safe bounds.
            # Recovery speed could also scale based on how 'perfect' the environment is
            recovery_modifier = abs(net_stress) / self.resilience if self.resilience > 0 else 1.0
            self.value = max(0.0, self.value - (self.recovery_rate * recovery_modifier))

    @property
    def damage_factor(self):
        """Returns the actual impact (0 to 1) only after a threshold (e.g., 0.5)"""
        if self.value > self.threshold_effect and 1.0 > self.threshold_effect > 0.0:
            return (self.value - self.threshold_effect) * (1/(1-self.threshold_effect))  # Scaled 0 to 1
        return 0.0