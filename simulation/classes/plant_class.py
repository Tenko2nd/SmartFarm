import math
import random

from warnings import deprecated

import simulation.utils.constants as C

class Plant(object):
    def __init__(self, plant_id, coordinate):
        # Identification
        self.id = plant_id
        self.coordinate = coordinate
        # Soil mesure
        self.N = int(random.gauss(83, 10))
        self.P = int(random.gauss(43, 5))
        self.K = int(random.gauss(133, 15))
        self.moisture_mm = random.randint(50,100)
        self.sensor_moisture = random.randint(50,100)

        # Health
        self.vitality = 1.0 # (Permanent)
        self.health = 1.0   # (Reversible)
        self.senescence = False
        self.is_dead = False
        self.stress_buffers = {}
        self.damages = {}
        self.penalties = {}
        self.death_timestamp = None
        self.severity_env = {}

        # Biological variable
        self.green_area = 0
        self.necrotic_area = 0.0 # Permanent tissue death (research: Does the plant get rid of it at one point)
        self.height = 0
        self.root_depth = 0
        self.gai = 0
        self.growth_stage = C.STAGE_SEQUENCE[0]
        self.next_stage = C.STAGE_SEQUENCE[C.STAGE_SEQUENCE.index(self.growth_stage) + 1] \
            if self.growth_stage != C.STAGE_SEQUENCE[-1] else self.growth_stage
        self.stage_pct = 0

        # Calculation variable
        self.de_prev = 0.0

        self.current_time = None
        self.last_observation = {}
        self.record_probe()

    def grow(self, env_conditions):
        """
        Updates the simulation for a period of C.DATA_UPDATE_MIN.
        """
        if self.is_dead:
            return

        step_hours = C.DATA_UPDATE_MIN / 60.0

        self.severity_env["vpd"] = 1 - self._optimal_vpd(env_conditions['vpd'])
        self.severity_env["moisture"] = 1 - self._optimal_soil(self.sensor_moisture, env_conditions['vpd'])
        self.severity_env["NPK"] = 1 - self._optimal_npk(self.N, self.P, self.K)
        co2, light = env_conditions["CO2"], env_conditions["LightIntensity"]
        self.severity_env["CO2"] = 1 - self._optimal_co2_light_synergy(co2, light)
        self._update_damages()

        # Update soil moisture
        self._update_soil_moisture(et0=env_conditions["et0"], humidity=env_conditions["Humidity"])

        # Calculate hourly progress adjusted by plant health
        days_in_state = C.PLANT_STAGE_TIME[self.growth_stage]
        hours_in_state = days_in_state * 24
        progress = (step_hours / hours_in_state) * 100 * min(1-min(sum(self.damages), 1) + 0.2, 1)

        # Consume NPK based on the progression before modifying the stage
        self._consume_nutrients(progress)

        self.stage_pct += progress
        if self.growth_stage != "M":
            if self.stage_pct >= 100:
                next_stage = C.STAGE_SEQUENCE[C.STAGE_SEQUENCE.index(self.growth_stage) + 1]
                self.growth_stage = next_stage
                self.stage_pct = 0

        self._update_height()
        self._update_gai()
        self._has_it_died()

    # TODO: Make critic index values have consequences later
    #  (eg. A plant has critic vpd, 5 hours later it appears visible problems if not fixed)
    # TODO: Change smooth curve for mors realistic curve? research realistic data to see how it react to each variable

    @deprecated("use the damage_buffers with the add_stress method for monitoring health and damage instead") # DEPRECATED
    def _calculate_health_score(self, env_conditions):
        """
        The pourcentage of optimal condition respected for the plant to grow perfectly.
        100% the plant grow perfectly, 0% it's dying
        :return: The optimal condition index in %
        """
        idx_vpd = self._optimal_vpd(env_conditions['vpd'])
        idx_moisture = self._optimal_soil(self.moisture, env_conditions['vpd'])
        idx_npk = self._optimal_npk(self.N, self.P, self.K)
        co2, light = env_conditions["CO2"], env_conditions["LightIntensity"]
        idx_co2_light = self._optimal_co2_light_synergy(co2, light)

        avg_score = (idx_co2_light + idx_moisture + idx_vpd + idx_npk) / 4
        limiting_factor = min(idx_co2_light, idx_moisture, idx_vpd)

        health_score = avg_score * (limiting_factor ** 0.5)
        # NOTE: For data to be more kind (real data are too harsh for simulated data, it will only penalize the model (it will have more data w/o it later)
        _ = min(health_score + 0.2, 1)


    def _optimal_vpd(self, vpd):
        """
        Calcule l'optimalité du climat pour le blé dur.
        """

        min_ideal, max_ideal = C.IDEAL_VPD_RANGE
        min_crit, max_crit = C.CRITICAL_VPD_RANGE

        score = 0.0
        # Critic limits
        if vpd <= min_crit or vpd >= max_crit:
            score = 0.0

        # Optimal state
        if min_ideal <= vpd <= max_ideal:
            score = 1.0

        # Interpolation
        if vpd < min_ideal:
            t = (vpd - min_crit) / (min_ideal - min_crit)
        else:
            t = (max_crit - vpd) / (max_crit - max_ideal)

        # Quadratic smoothing
        score = round((3 * t ** 2 - 2 * t ** 3), 3)

        self._apply_environmental_stress("vpd", score, 0.5, recover_ratio=4) #RESEARCH

        return score

    def _optimal_soil(self, current_moisture, vpd):
        """
        #FIXME: We use the moisture mm now and the stress from the function update moisture can help!!
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

        score = 0.0
        # 3. SCORING LOGIC
        # Case A: Below Critical Minimum (Death zone)
        if current_moisture <= min_crit:
            score = 0.0

        # Case B: Within a small buffer around the dynamic target (Perfect zone)
        # We allow a +/- 10% tolerance for 100% score
        elif (dynamic_target - 10) <= current_moisture <= (dynamic_target + 10):
            score = 1.0

        # Case C: Between Critical Min and Target (Drought Stress)
        elif current_moisture < dynamic_target:
            t = (current_moisture - min_crit) / ((dynamic_target - 5) - min_crit)
            score = round((3 * t ** 2 - 2 * t ** 3), 3)  # Smoothstep
            self.moisture = dynamic_target + abs(random.gauss(0, 5))

        # Case D: Above Target (Saturation / Over-watering)
        # Higher is better than lower: the score only drops to 60% health at saturation
        elif current_moisture > dynamic_target:
            # Distance from target to 100% moisture
            t = (max_crit - current_moisture) / (max_crit - (dynamic_target + 5))
            smooth = (3 * t ** 2 - 2 * t ** 3)
            # We remap the score so it goes from 100% down to 50% (instead of 0%)
            score = round(0.5 + (0.5 * smooth), 3)

        self._apply_environmental_stress("moisture", score, 1, recover_ratio=0.5) #RESEARCH

        return score

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
        # PDF mentions 'Luxury Consumption' (page 1), so excess is usually not harmful for cereal yields in the soil.
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

        self._apply_environmental_stress("NPK", global_index, 2, recover_ratio=0.5) #RESEARCH

        return global_index


    def _optimal_co2_light_synergy(self, co2_ppm, ppfd):
        """
        Calculates an adequacy score (0-100%) for the combination of CO2 and Light.
        Based on:
        - Bugbee & Salisbury (1988) https://doi.org/10.1104/pp.88.3.869: CO2 saturation at 1200 ppm.
        - Li et al. (2025) https://doi.org/10.1002/pei3.70085: Light optimum for wheat at 700-900 PPFD.
        - Standard C3 Physiological Heuristic: CO2(ppm) / PPFD(umol) ratio.
        """

        # INTENSITY SAFETY SCORE (Is the light too strong for the species?)
        max_ideal_light = C.IDEAL_LIGHT_INTENSITY
        light_score = 1.0
        if ppfd > max_ideal_light:
            t_light = max(0, (2500 - ppfd) / (2500 - 1000))
            light_score = (3 * t_light ** 2 - 2 * t_light ** 3)
            self._apply_environmental_stress("highLight", light_score, 1.2, recover_ratio=2) #RESEARCH
            self._apply_environmental_stress("lowLight", 1, 0, recover_ratio=2) # recover
        elif ppfd < 200:
            t_light = max(0, (ppfd - 50) / (300 - 50))
            light_score = (3 * t_light ** 2 - 2 * t_light ** 3)
            self._apply_environmental_stress("lowLight", light_score, 2, recover_ratio=2) #RESEARCH
            self._apply_environmental_stress("highLight", 1, 0, recover_ratio=2)
        else: # Recover in case everything is fine
            self._apply_environmental_stress("lowLight", 1, 0, recover_ratio=2)
            self._apply_environmental_stress("highLight", 1, 0, recover_ratio=2)


        # ADEQUACY SCORE (The Balance)
        # We calculate the "Target CO2" for the current light intensity.
        target_co2 = ppfd * 1.1

        # Clamp target between ambient (400) and max useful (1200)
        min_co2, max_co2 = C.CO2_TARGET_LIMITS
        target_co2 = max(min_co2, min(max_co2, target_co2))

        adequacy_ratio = co2_ppm / target_co2

        # Scoring the Ratio
        if 0.9 <= adequacy_ratio <= 1.1:
            # Perfect balance zone (+/- 10% deviation)
            adequacy_score = 1.0
        elif adequacy_ratio < 0.9:
            # Case: Starvation (CO2 is too low for this light)
            t = (adequacy_ratio - 0.5) / (0.9 - 0.5)
            t = max(0, min(1, t))
            adequacy_score = (3 * t ** 2 - 2 * t ** 3)
        else:
            # Case: Waste (CO2 is too high for this light) stress is minimum
            t = (2.0 - adequacy_ratio) / (2.0 - 1.1)
            t = max(0, min(1, t))
            adequacy_score = 0.9 + (0.1 * (3 * t ** 2 - 2 * t ** 3))

        self._apply_environmental_stress("CO2", adequacy_score, 1.5, recover_ratio=5) #RESEARCH

        # FINAL SCORE
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

    def _update_soil_moisture(self, et0, humidity):
        """
        Estimates the loss of soil moisture % based on ET0.
        FAO-56 Dual Crop Coefficient Method for Estimating Evaporation from Soil and Application Extensions. https://doi.org/10.1061/(ASCE)0733-9437(2005)131:1(2)
        :param et0: Reference ET0 in mm/day
        NOTE: The result is based on TAW (plant view) of the moisture used for realistic growth, not the sensor (TEW)
        """
        step_hours = C.DATA_UPDATE_MIN / 60.0
        old_taw = 1000 * C.SOIL_WATER_CAPACITY * self.root_depth
        self.root_depth = self.height * C.HEIGH_TO_ROOT_RATIO
        taw = 1000 * C.SOIL_WATER_CAPACITY * self.root_depth
        raw = C.P_RAW * taw

        if taw > old_taw:
            self.moisture_mm += (taw - old_taw) * (self.moisture_mm / old_taw) # Keep the same ratio

        # Get Crop Coefficient (Kc) with linear interpolation
        next_stage = C.STAGE_SEQUENCE[C.STAGE_SEQUENCE.index(self.growth_stage) + 1] \
            if self.growth_stage != C.STAGE_SEQUENCE[-1] else self.growth_stage
        kcb_stage = C.KC_MAPPING.get(self.growth_stage, 0.5)
        kcb_base = kcb_stage + (C.KC_MAPPING.get(next_stage, 0.5) - kcb_stage) * (self.stage_pct / 100.0)
        u2 = 0 # No wind in a green house
        # FIXME: They used the mean daily minimum relative humidity for the season (maybe simplification, daily min should do enough)
        climate_adjustment = (0.04*(u2-2) - 0.004*(humidity-45))*math.pow((self.height/3),0.3)
        kcb = kcb_base + climate_adjustment
        kc_max = max(1.2  + climate_adjustment, kcb + 0.05)

        fc = math.pow((kcb - C.KC_MIN) / (kc_max - C.KC_MIN), 1 + 0.5 * self.height)
        fc = max(0.0, min(0.99, fc))

        few = min(1 - fc, C.FW)

        if self.de_prev <= C.REW:
            kr = 1.0  # Stage 1: Soil is wet, evaporating at max rate
        else:
            # Stage 2: Soil is drying, evaporation is slowing down
            kr = (C.TEW - self.de_prev) / (C.TEW - C.REW)
            kr = max(0.0, min(1.0, kr))

        ke = min(kr * (kc_max - kcb), few * kc_max)

        depletion_mm = taw - self.moisture_mm
        if depletion_mm > raw:
            ks = (taw - depletion_mm) / (taw - raw)
        else:
            ks = 1.0
        ks = max(0.0, ks) #TODO: Use this to apply stress

        # Transpiration (from plants)
        t_daily_mm = (kcb * ks) * et0
        # Evaporation (from soil surface)
        e_daily_mm = ke * et0

        # Scale to step_hours
        t_step_mm = (t_daily_mm / 24.0) * step_hours
        e_step_mm = (e_daily_mm / 24.0) * step_hours
        etc_step_mm = t_step_mm + e_step_mm

        #NOTE: when I want to waterize use this: net_wetting = irrigation_step / fw # remember to update de_prev also
        # Also need to be change for self.moisture_mm
        # --- RECEPTION DE L'EAU (Pluie ou Irrigation) ---
        # eau_entree_mm = pluie_mm + irrigation_mm
        #
        # # 1. Mise à jour du GROS bucket (Root Zone - TAW)
        # # On réduit la déplétion du gros bucket (Dr)
        # dr_prev = (1 - (self.moisture / 100)) * TAW
        # dr_new = dr_prev - eau_entree_mm + etc_step_mm
        # dr_new = max(0, dr_new)  # On ne peut pas avoir une déplétion négative
        # self.moisture = (1 - (dr_new / TAW)) * 100
        #
        # # 2. Mise à jour du PETIT bucket (Surface - TEW)
        # # fw est la fraction mouillée (ex: 1.0 pour la pluie)
        # # On divise l'eau par fw car elle se concentre sur la zone mouillée (Eq. 12)
        # self.de_prev = self.de_prev - (eau_entree_mm / fw) + (e_step_mm / few)
        # self.de_prev = max(0, min(TEW, self.de_prev))

        # the roots in the 10cm are also taking water from it. The power 0.6 is because it is denser than lower roots.
        proportion_roots_in_surface = math.pow(C.ZE / self.root_depth, 0.6) if self.root_depth > 0 else 0
        proportion_roots_in_surface = min(1.0, proportion_roots_in_surface)

        # Water drained from 'surfacique roots'
        tei_step_mm = t_step_mm * proportion_roots_in_surface

        # Update surface depletion
        de_loss = (e_step_mm / few) if few > 0 else 0
        self.de_prev = self.de_prev + de_loss + tei_step_mm
        self.de_prev = max(0.0, min(C.TEW, self.de_prev))

        # Apply to moisture
        self.moisture_mm -= (t_step_mm + e_step_mm)
        self.moisture_mm = max(0.0, min(taw, self.moisture_mm))
        self.sensor_moisture = ((C.TEW - self.de_prev) / C.TEW) * 100

    def _update_height(self):
        next_stage = C.STAGE_SEQUENCE[C.STAGE_SEQUENCE.index(self.growth_stage) + 1] \
            if self.growth_stage != C.STAGE_SEQUENCE[-1] else self.growth_stage
        height_stage = C.HEIGHT_MAPPING.get(self.growth_stage, 0.5)
        self.height = height_stage + (C.HEIGHT_MAPPING.get(next_stage, 0.5) - height_stage) * (self.stage_pct / 100.0)

        health_status = 1-min(sum(self.damages),1) # min((1-self.damages)*1.3, 1)  #For generosity
        # TODO: Depending on health status, inflict size penalty no removable (research which variable influence size delay)

    def _update_gai(self):
        next_stage = C.STAGE_SEQUENCE[C.STAGE_SEQUENCE.index(self.growth_stage) + 1] \
            if self.growth_stage != C.STAGE_SEQUENCE[-1] else self.growth_stage
        gai_stage = C.GAI_MAPPING.get(self.growth_stage, 0.5)
        self.gai = gai_stage + (C.GAI_MAPPING.get(next_stage, 0.5) - gai_stage) * (self.stage_pct / 100.0)

        health_status = 1 - min(sum(self.damages), 1)  # min((1-self.damages)*1.3, 1)  #For generosity
        # TODO: Depending on health status, inflict gai penalty (research which variable influence gai, yellowing, ...)
        # TODO: add recovery rate if penalty and health status good

    def record_probe(self):
        """Called only when IsProbed == 1. Updates the memory."""
        self.last_observation = {
            "N": self.N,
            "P": self.P,
            "K": self.K,
            "moisture": self.sensor_moisture,
            "gai": self.gai,
            "height": self.height,
            "growth_stage": self.growth_stage,
            "is_alive": 0 if self.is_dead else 1,
        }

    def _has_it_died(self):
        if self.green_pixels < (self.green_pixels+self.necrotic_pixels)*0.15:
            self.is_dead = True
            self.death_timestamp = self.current_time

    def _apply_environmental_stress(self, stress_name, score, days_to_effect, recover_ratio=2.0):
        """
        score: 0.0 (bad) to 1.0 (perfect)
        days_to_effect: how many days before visible effect
        """
        # Create the buffer if it doesn't exist yet
        if stress_name not in self.stress_buffers:
            if score < 1.0:
                self.stress_buffers[stress_name] = StressBuffer(stress_name, recover_ratio, days_to_effect)
            else:
                return

        buffer = self.stress_buffers[stress_name]

        #  Update logic: If score is low, add stress. If score is high, recover.
        if score < 1.0:
            increment = (1.0 - score)
            buffer.add_stress(increment)
        else:
            buffer.recover()

    def _update_damages(self):
        self.damages = []
        for it, buf in self.stress_buffers.items():
            if buf.takes_damages and it in self.severity_env:
                self.damages.append(self.severity_env[it])
            elif buf.takes_damages and it not in self.severity_env:
                self.damages.append(1.0)

        finished = [name for name, buf in self.stress_buffers.items() if buf.is_empty()]
        for name in finished:
            del self.stress_buffers[name]

