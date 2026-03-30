class StressBuffer:
    # TODO: days to effect can be int or tuple, when tuple, use a random for variation of effect.
    def __init__(self, name, days_to_effect, recovery_days, threshold_visible_effect = 0.5, is_fatal=False):
        self.name = name
        self.value = 0.0  # 0.0 to 1.0
        self.threshold_effect = max(0.0, min(1.0, threshold_visible_effect))

        self.ticks_to_fill = days_to_effect * C.TICKS_DAY
        self.ticks_to_recover = recovery_days * C.TICKS_DAY

        self.recovery_rate = 1.0 / self.ticks_to_recover if recovery_days > 0 else 0
        self.is_fatal = is_fatal  # If self.value = 1.0, does the plant die immediately?

    def update(self, stress_intensity):
        """
        stress_intensity: 0.0 (perfect) to 1.0 (deadly environment)
        """
        if stress_intensity > 0.1: # 10 % margin
            # We add stress based on how bad the environment is
            increment = stress_intensity / self.ticks_to_fill
            self.value = min(1.0, self.value + increment)
        else: # Natural recovery
            self.value = max(0.0, self.value - self.recovery_rate)

    @property
    def damage_factor(self):
        """Returns the actual impact (0 to 1) only after a threshold (e.g., 0.5)"""
        if self.value > self.threshold_effect and 1.0 > self.threshold_effect > 0.0:
            return (self.value - self.threshold_effect) * (1/(1-self.threshold_effect))  # Scaled 0 to 1
        return 0.0