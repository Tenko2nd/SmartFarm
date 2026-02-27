import random
import Simulation.utils.Constants as C

class Plant(object):
    def __init__(self, plant_id, coordinate):
        # Identification
        self.id = plant_id
        self.coordinate = coordinate
        # Soil mesure
        self.N = int(random.gauss(83, 10))
        self.P = int(random.gauss(43, 5))
        self.K = int(random.gauss(133, 15))
        self.moisture = random.randint(50,100)
        # Biological variable
        self.green_pixels = 0
        self.necrotic_pixels = 0
        self.GLI = 0
        self.growth_stage = "E"
        self.stage_pct = 0
        self.health_score = None
        self.is_dead = False
        self.damage_buffer = {}

        self.last_observation = {}
        self.record_probe()

    def grow(self, env_conditions):
        """
        Updates the simulation for a period of C.DATA_UPDATE_MIN.
        """
        if self.is_dead:
            return

        step_hours = C.DATA_UPDATE_MIN / 60.0

        # Update soil moisture
        self._update_soil_moisture(et0=env_conditions["et0"])
        self._calculate_health_score(env_conditions=env_conditions)

        # Calculate hourly progress adjusted by plant health
        days_in_state = C.PLANT_STAGE_TIME[self.growth_stage]
        hours_in_state = days_in_state * 24
        progress = (step_hours / hours_in_state) * 100 * min(self.health_score + 0.3,
                                                             1)  # For margin (0.7 is good for growth and it doesn't stop completetly)

        # Consume NPK based on the progression before modifying the stage
        self._consume_nutrients(progress)

        self.stage_pct += progress
        if self.growth_stage != "M":
            if self.stage_pct >= 100:
                next_stage = C.STAGE_SEQUENCE[C.STAGE_SEQUENCE.index(self.growth_stage) + 1]
                self.growth_stage = next_stage
                self.stage_pct = 0

        self._update_pixels(light_intensity=env_conditions["LightIntensity"])
        self._update_gli()

    # TODO: Make critic index values have consequences later
    #  (eg. A plant has critic vpd, 5 hours later it appears visible problems if not fixed)
    # TODO: Change smooth curve for mors realistic curve? research realistic data to see how it react to each variable
    def _calculate_health_score(self, env_conditions):
        """
        The pourcentage of optimal condition respected for the plant to grow perfectly.
        100% the plant grow perfectly, 0% it's dying
        :return: The optimal condition index in %
        """
        idx_vpd = self._optimal_vpd(env_conditions['vpd'])
        idx_moisture = self._optimal_soil(self.moisture, env_conditions['vpd'])
        idx_npk = self._optimal_npk(self.N, self.P, self.K)
        co2, light = env_conditions["c02"], env_conditions["light"]
        idx_co2_light = self._optimal_co2_light_synergy(co2, light)

        avg_score = (idx_co2_light + idx_moisture + idx_vpd + idx_npk) / 4
        limiting_factor = min(idx_co2_light, idx_moisture, idx_vpd)

        health_score = avg_score * (limiting_factor ** 0.5)
        # NOTE: For data to be more kind (real data are too harsh for simulated data, it will only penalize the model (it will have more data w/o it later)
        self.health_score = min(health_score + 0.2, 1)

    @staticmethod
    def _optimal_vpd(vpd):
        """
        Calcule l'optimalité du climat pour le blé dur.
        """

        min_ideal, max_ideal = C.IDEAL_VPD_RANGE
        min_crit, max_crit = C.CRITICAL_VPD_RANGE

        # Critic limits
        if vpd <= min_crit or vpd >= max_crit:
            return 0.0

        # Optimal state
        if min_ideal <= vpd <= max_ideal:
            return 1.0

        # Interpolation
        if vpd < min_ideal:
            t = (vpd - min_crit) / (min_ideal - min_crit)
        else:
            t = (max_crit - vpd) / (max_crit - max_ideal)

        # Quadratic smoothing
        score = (3 * t ** 2 - 2 * t ** 3)
        return round(score, 3)

    def _optimal_soil(self, current_moisture, vpd):
        """
        Calculates soil moisture optimality (0-100%) dynamically based on VPD.

        Args:
            current_moisture (float): Actual soil moisture percentage (0-100).
        """
        vpd_low, vpd_high = C.CRITICAL_VPD_RANGE
        moisture_low, moisture_high = C.SOIL_MOISTURE_RANGE  # Minimum and Maximum ideal targets
        min_crit, max_crit = C.CRITIC_SOIL_MOISTURE_RANGE

        # 2. CALCULATE DYNAMIC TARGET
        # We map the VPD to the Moisture target range
        # If VPD is low (0.5), target is 60%. If VPD is high (3.14), target is 90%.
        clamped_vpd = max(vpd_low, min(vpd_high, vpd))
        vpd_ratio = (clamped_vpd - vpd_low) / (vpd_high - vpd_low)

        # This is the "Perfect" moisture point for the current weather
        dynamic_target = moisture_low + (vpd_ratio * (moisture_high - moisture_low))

        # 3. SCORING LOGIC
        # Case A: Below Critical Minimum (Death zone)
        # TODO: yellow leaves
        if current_moisture <= min_crit:
            self.moisture = dynamic_target + random.gauss(0, 3)
            return 0.0

        # Case B: Within a small buffer around the dynamic target (Perfect zone)
        # We allow a +/- 5% tolerance for 100% score
        if (dynamic_target - 5) <= current_moisture <= (dynamic_target + 5):
            return 1.0

        # Case C: Between Critical Min and Target (Drought Stress)
        if current_moisture < dynamic_target:
            t = (current_moisture - min_crit) / ((dynamic_target - 5) - min_crit)
            score = (3 * t ** 2 - 2 * t ** 3)  # Smoothstep
            return round(score, 3)

        # Case D: Above Target (Saturation / Over-watering)
        # Higher is better than lower: the score only drops to 60% health at saturation
        if current_moisture > dynamic_target:
            # Distance from target to 100% moisture
            t = (max_crit - current_moisture) / (max_crit - (dynamic_target + 5))
            smooth = (3 * t ** 2 - 2 * t ** 3)
            # We remap the score so it goes from 100% down to 50% (instead of 0%)
            score = 0.5 + (0.5 * smooth)
            return round(score, 3)

        return 0.0

    @staticmethod
    def _score_single_nutrient(current_ppm, ideal_range, deficient_val):
        """
        Calculates a 0-100% score for a single nutrient.
        High/Over-abundance stays at 100% (Luxury consumption).
        """
        min_ideal, max_ideal = ideal_range

        # 1. Below Deficiency (0%)
        if current_ppm <= deficient_val:
            return 0.0

        # 2. Within or Above Optimal Range (100%)
        # PDF mentions 'Luxury Consumption' (page 1), so excess is usually not
        # harmful for cereal yields in the soil.
        if current_ppm >= min_ideal:
            return 1.0

        # 3. Transition from Deficient to Ideal (Smoothstep curve)
        # Ratio between 0 and 1
        t = (current_ppm - deficient_val) / (min_ideal - deficient_val)
        score = (3 * t ** 2 - 2 * t ** 3)

        return round(score, 3)

    def _optimal_npk(self, n_ppm: int, p_ppm: int, k_ppm: int):
        """
        Calculates the global NPK optimality index based on Alberta Agriculture data.
        """
        # Configuration based on your provided values (ppm)

        # Calculate individual scores
        scores = {
            "N": self._score_single_nutrient(n_ppm, C.NPK_SOIL_RATIO_RANGE["N"], C.DEFICIENT_NPK_SOIL_RATIO["N"]),
            "P": self._score_single_nutrient(p_ppm, C.NPK_SOIL_RATIO_RANGE["P"], C.DEFICIENT_NPK_SOIL_RATIO["P"]),
            "K": self._score_single_nutrient(k_ppm, C.NPK_SOIL_RATIO_RANGE["K"], C.DEFICIENT_NPK_SOIL_RATIO["K"])
        }

        # The Global Index follows the 'Law of the Minimum'
        # Your plant is only as healthy as its most deficient nutrient.
        global_index = min(scores.values())

        return global_index

    @staticmethod
    def _optimal_co2_light_synergy(co2_ppm, ppfd):
        """
        Calculates an adequacy score (0-100%) for the combination of CO2 and Light.
        Based on:
        - Bugbee & Salisbury (1988) https://doi.org/10.1104/pp.88.3.869: CO2 saturation at 1200 ppm.
        - Li et al. (2025) https://doi.org/10.1002/pei3.70085: Light optimum for wheat at 700-900 PPFD.
        - Standard C3 Physiological Heuristic: CO2(ppm) / PPFD(umol) ratio.
        """

        # 1. NIGHT MODE
        if ppfd < 30:
            # At night, CO2 doesn't "combine" with light.
            # Score returns 100% health-wise unless CO2 is toxic (>2000).
            return 1.0 if co2_ppm < 2000 else 0.0

        # 2. INTENSITY SAFETY SCORE (Is the light too strong for the species?)
        _, max_ideal_light = C.IDEAL_LIGHT_INTENSITY
        light_score = 1.0
        if ppfd > 1000:
            t_light = max(0, (1800 - ppfd) / (1800 - 1000))
            light_score = (3 * t_light ** 2 - 2 * t_light ** 3)

        # 3. ADEQUACY SCORE (The Balance)
        # We calculate the "Target CO2" for the current light intensity.
        target_co2 = ppfd * 1.1

        # Clamp target between ambient (400) and max useful (1200)
        min_co2, max_co2 = C.CO2_TARGET_LIMITS
        target_co2 = max(min_co2, min(max_co2, target_co2))

        adequacy_ratio = co2_ppm / target_co2

        # 4. Scoring the Ratio
        if 0.9 <= adequacy_ratio <= 1.1:
            # Perfect balance zone (+/- 10% deviation)
            adequacy_score = 1.0
        elif adequacy_ratio < 0.9:
            # Case: Starvation (CO2 is too low for this light)
            t = (adequacy_ratio - 0.5) / (0.9 - 0.5)
            t = max(0, min(1, t))
            adequacy_score = (3 * t ** 2 - 2 * t ** 3)
        else:
            # Case: Waste (CO2 is too high for this light)
            t = (2.0 - adequacy_ratio) / (2.0 - 1.1)
            t = max(0, min(1, t))
            adequacy_score = 0.7 + (0.3 * (3 * t ** 2 - 2 * t ** 3))

        # 4. FINAL SCORE
        # The final index is the combination of having enough CO2 for the light
        # and not having so much light that it kills the plant.
        final_score = adequacy_score * light_score

        return round(max(0, final_score), 3)

    def _consume_nutrients(self, stage_pct_improve):
        """
        Reduces N, P, K in the soil based on the growth progress of the current state.
        :param stage_pct_improve: the percentage of progression in the stage since last update
        """
        if self.growth_stage not in C.UPTAKE_MAPPING or self.growth_stage == "M":
            return

        # Get the distribution ratio for the current state
        ratios = C.UPTAKE_MAPPING[self.growth_stage]

        # Calculation:
        # (Total PPM needed for life) * (% allocated to this state) * (% of state completed now)
        consumption_n = C.TOTAL_NPK_PPM[0] * (ratios[0] / 100) * (stage_pct_improve / 100)
        consumption_p = C.TOTAL_NPK_PPM[1] * (ratios[1] / 100) * (stage_pct_improve / 100)
        consumption_k = C.TOTAL_NPK_PPM[2] * (ratios[2] / 100) * (stage_pct_improve / 100)

        # Reduce soil values (ensuring they don't go below 0)
        self.N = max(0, round(self.N - consumption_n, 5))
        self.P = max(0, round(self.P - consumption_p, 5))
        self.K = max(0, round(self.K - consumption_k, 5))

    def _update_soil_moisture(self, et0):
        """
        Estimates the loss of soil moisture % based on ET0.
        :param et0: Reference ET0 in mm/day
        """
        step_hours = C.DATA_UPDATE_MIN / 60.0
        max_water_mm = C.MAX_WATER_DEPTH_MM * C.POT_DEPTH

        # Get Crop Coefficient (Kc) with linear interpolation
        next_stage = C.STAGE_SEQUENCE[C.STAGE_SEQUENCE.index(self.growth_stage) + 1] \
            if self.growth_stage != "M" else self.growth_stage
        kc_stage = C.KC_MAPPING.get(self.growth_stage, 0.5)
        kc_linear = C.KC_MAPPING.get(next_stage, 0.5) - C.KC_MAPPING.get(self.growth_stage, 0.5) * self.stage_pct/100
        kc = kc_stage + kc_linear

        # Calculate Actual Evapotranspiration (ETc)
        etc = et0 * kc

        # Convert daily loss to hourly loss for the specific time step
        etc_step = (etc / 24.0) * step_hours

        # Convert mm loss to percentage points loss
        # Calculation: (mm_loss / total_mm_capacity) * 100
        pct_loss = (etc_step / max_water_mm) * 100

        # Apply loss to soil moisture
        #TODO: add some noise? create a function global
        self.moisture = round(max(0, self.moisture - pct_loss), 5)

    def _update_pixels(self, light_intensity):

        growth_speed = 0.01
        growth_factor = 0
        recovery_rate = 0.1
        step_hours = C.DATA_UPDATE_MIN / 60.0

        match self.growth_stage:
            case "M":
                growth_factor = self.green_pixels * self.stage_pct / 100
            case "A":
                growth_factor = 1.0
            case "H":
                growth_factor = self.green_pixels  * growth_speed
            case "FN":
                growth_factor = self.green_pixels  * growth_speed * 2
            case "TS":
                growth_factor = self.green_pixels  * growth_speed * 5
            case "FI":
                growth_factor = self.green_pixels  * growth_speed * 4
            case "E":
                growth_factor = self.stage_pct
            case _:
                growth_factor = 0.0

        growth_factor = growth_factor * 0.2 if light_intensity <= 15 else growth_factor  # growth slower at night

        if self.health_score > 0.6:
            # --- HEALTHY GROWTH ---
            potential_new_px = growth_factor * (self.health_score - 0.5) * step_hours
            # Limit growth to pot size
            if (self.green_pixels + self.necrotic_pixels + int(potential_new_px)) < C.PLANT_POT_SIZE_PX:
                self.green_pixels += int(potential_new_px)
            else:
                self.green_pixels = C.PLANT_POT_SIZE_PX - self.necrotic_pixels

            if self.necrotic_pixels > 0:
                healed = min(self.necrotic_pixels, potential_new_px * recovery_rate)
                self.necrotic_pixels -= int(healed)

        elif self.health_score < 0.4:
            # --- STRESS & NECROSIS ---
            loss_factor = (0.5 - self.health_score) * 0.03
            damage = self.green_pixels * loss_factor * step_hours

            self.green_pixels = max(0, self.green_pixels - int(damage))
            self.necrotic_pixels += int(damage)

    def _update_gli(self):
        total_visible_px = self.green_pixels + self.necrotic_pixels

        if total_visible_px == 0:
            self.GLI = 0.0
            return

        greenness_ratio = self.green_pixels / total_visible_px
        density_ratio = total_visible_px / C.PLANT_POT_SIZE_PX

        # Final GLI = (Quality of tissue) * (Quantity of tissue) * Stage
        raw_gli = greenness_ratio * density_ratio * 0.8

        # Adjust for senescence (Maturity stage 'M' naturally loses GLI)
        if self.growth_stage == "M":
            raw_gli = self.health_score * (1.0 - self.stage_pct / 1000 * 4) * 0.8

        self.GLI = round(max(0.0, min(0.8, raw_gli)), 3)

    def record_probe(self):
        """Called only when IsProbed == 1. Updates the memory."""
        self.last_observation = {
            "N": self.N,
            "P": self.P,
            "K": self.K,
            "moisture": self.moisture,
            "green_pixels": self.green_pixels,
            "necrotic_pixels": self.necrotic_pixels,
            "gli": self.GLI,
            "growth_stage": self.growth_stage
        }



