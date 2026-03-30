import math
import random
import simulation.utils.constants as C

from simulation.Class.plant_class import Plant


class Farm(object):
    def __init__(self, farm_id, nb_plants):
        # Identification and PLants
        self.id = farm_id
        self.plantList = self.plants = [
            Plant(f"P_{farm_id}_{i}", [i // 2, i % 2])
            for i in range(nb_plants)
        ]
        self.current_time = None
        # Internal characteristics
        #TODO: For scenarii, change the deviation drastically over short time to stress the plant
        self.temperature_deviation = 0 #random.uniform(-10, 5)
        self.humidity_deviation = +10 #random.uniform(-20, 10)
        self.light_intensity_deviation = 0 #random.gauss(0, 100)
        self.pressure_deviation = 0 #random.gauss(50, 150)
        # Environment values
        self.temp_extreme = {"min": 0, "max": 0}  # Temperature minimal and maximal of the day
        self.temperature = None
        self.humidity = None
        self.light_intensity = None
        self.pressure = None
        self.vpd = None
        self.et0 = None
        self.co2 = 900
        # Dict for data
        self.current_condition = {}
        self.death_grace_period_days = 7

    def get_farm_environment(self):
        env = {"Temperature": self.temperature, "Humidity": self.humidity, "LightIntensity": self.light_intensity,
               "CO2": self.co2, "vpd": self.vpd, "et0": self.et0}
        return env

    def update_extreme_temp(self, temp_extreme):
        """
        Once a day is called to update the minimal and maximal temperature of the day
        :param temp_extreme: the temperature minimal and maximal for the day
        """
        self.temp_extreme = {k: v + self.temperature_deviation for k, v in temp_extreme.items()}

    def update_farm_environment(self, env):
        self.temperature = env["Temperature"] + self.temperature_deviation
        self.humidity = env["Humidity"] + self.humidity_deviation
        self.light_intensity = env["LightIntensity"] + self.light_intensity_deviation
        self.pressure = env["Pressure"] + self.pressure_deviation
        self._calculate_vpd()
        self._calculate_room_co2_drawdown()
        self._calculate_fao56_et0(timestamp=self.current_time)
        self._update_env_dict()
        for plant in self.plants:
            plant.current_time = self.current_time
            if plant.is_dead:
                days_since_death = (self.current_time - plant.death_timestamp).days
                if days_since_death > self.death_grace_period_days:
                    self.plantList.remove(plant)
                    continue
            plant.grow(env_conditions=self.get_farm_environment())

    def _calculate_room_co2_drawdown(self):
        """
        Calculates the remaining CO2 in a room after plant absorption.
        Based on NASA TM 102788 (Wheeler & Sager) : 'Carbon Dioxide And Water Exchange Rates By A Wheat Crop In NASA'S
        Biomass Production Chamber: Results From An 86-Day Study (January To April 1989)',
        and Gruda et al. (2025) : 'Environmental conditions and nutritional quality of vegetables in protected cultivation'
        """

        # 1. CONSTANTS & GAS PHYSICS
        R = 0.08206  # Ideal Gas Constant (L*atm / K*mol)
        temp_k = self.temperature + 273.15
        # Calculate Molar Volume of air at current temp (L/mol)
        pressure_atm = self.pressure / 1013.25
        molar_volume = R * temp_k / pressure_atm

        # 2. CALCULATE STAND-LEVEL UPTAKE POTENTIAL (umol/m2/s)
        # The NASA equation y = 0.054784x - 9.6297 is for a FULL CANOPY (Stand).
        base_respiration = 9.6297

        # Adjust respiration for temperature (Article 2: 75% increase from 16C to 24C)
        # This roughly equates to a 9% change per degree Celsius from a 20C baseline
        # FIXME: Research as it cannot be just a straight line
        temp_factor = 1 + (self.temperature - 20) * 0.09
        adjusted_respiration = base_respiration * temp_factor

        gross_photosynthesis = 0.054784 * self.light_intensity

        # 3. AREA & LIGHT INTERCEPTION
        total_delta_mol = 0
        for plant in self.plantList:
            # Beer-Lambert Law: Fraction of light intercepted by the green leaf layers
            # If GAI is low (0.1), f_ipar is low. If GAI is high (3.0+), f_ipar is near 1.0
            f_ipar = 1 - math.exp(-C.K_EXTINCTION * plant.gai)

            # Uptake for the plant
            net_rate = (gross_photosynthesis * f_ipar) - (adjusted_respiration * f_ipar)

            total_seconds = C.DATA_UPDATE_MIN * 60
            plant_mol = (net_rate * C.PLANT_POT_SIZE_M2 * total_seconds) / 1_000_000
            total_delta_mol += plant_mol

        # 3. ADJUST FOR CO2 CONCENTRATION LIMITATION
        # Article 2 shows rate is stable from 800-2200ppm but drops below 800.
        co2_efficiency = 1.0
        if self.co2 < 800:
            # Linear scaling factor: at 800ppm = 1.0, at 190ppm (compensation point) = 0.0
            co2_efficiency  = max(0.0, (self.co2 - 190) / (800 - 190))
        elif self.co2 > 2200:
            # Article 2 notes slight decrease/saturation above 2200
            co2_efficiency *= 0.9

        total_mol_absorbed = total_delta_mol * co2_efficiency  # convert umol to mol

        # 5. CONVERT ABSORBED MICROMOLES TO PPM CHANGE IN ROOM
        # ppm = (micromoles_of_gas / total_moles_of_air)
        total_moles_air_in_room = C.ROOM_VOLUME_LITERS / molar_volume
        delta_ppm = total_mol_absorbed / total_moles_air_in_room

        self.co2 = round(self.co2 - delta_ppm, 2) if self.co2 > 850 else self.co2 + random.randint(75, 125)

    def _calculate_vpd(self):
        """
        Calculate VPD (Vapor Pressure Deficit) in kPa.
        """
        # Saturation Vapor Pressure (es)
        es = 0.6108 * math.exp((17.27 * self.temperature) / (self.temperature + 237.3))

        # Actual Vapor Pressure (ea)
        ea = es * (self.humidity / 100.0)

        # VPD
        self.vpd = es - ea

        # Slope of Vapor Pressure Curve (delta)
        delta = (4098 * es) / math.pow((self.temperature + 237.3), 2)

        return delta

    def _estimate_solar_radiation(self, jour_annee):
        """
        Estime le rayonnement solaire (Rs) en MJ/m2/jour
        Méthode Hargreaves-Samani (recommandée par la FAO-56).
        Args:
            jour_annee (int): Jour de l'année.
        """
        # 1. Calcul de la radiation extra-terrestre (Ra)
        # Latitude en radians
        lat_rad = (math.pi / 180) * math.radians(C.LATITUDE_BANGKOK)

        # Déclinaison solaire
        dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * jour_annee)
        delta = 0.409 * math.sin(2 * math.pi / 365 * jour_annee - 1.39)

        # Angle horaire au coucher du soleil
        ws = math.acos(-math.tan(lat_rad) * math.tan(delta))

        # Constante solaire Gsc = 0.0820 MJ/m2/min
        ra = (24 * 60 / math.pi) * 0.0820 * dr * (
                ws * math.sin(lat_rad) * math.sin(delta) +
                math.cos(lat_rad) * math.cos(delta) * math.sin(ws)
        )

        # 2. Ajustement selon l'écart de température (Hargreaves)
        # k_rs est un coefficient d'ajustement (0.16 pour l'intérieur des terres)
        k_rs = 0.16
        rs = k_rs * math.sqrt(self.temp_extreme["max"] - self.temp_extreme["min"]) * ra
        return rs

    def _calculate_fao56_et0(self, timestamp, solar_rad=None):
        """
        Calculates the Reference Evapotranspiration (ET0) based on the FAO-56
        Penman-Monteith method.
        Lu, Y., Ma, D., Chen, X., & Zhang, J. (2018). A Simple Method for Estimating Field Crop Evapotranspiration from Pot Experiments. Water, 10(12), 1823. https://doi.org/10.3390/w10121823


        Args:
            timestamp (date): Date of the day.
            solar_rad (float): Net radiation (MJ/m2/day). If None, it estimates based on temp.

        Returns:
            float: Estimated ET0 in mm/day.
        """
        delta = self._calculate_vpd()

        wind_speed = 0  # We are in a green house

        # 5. Psychrometric Constant (gamma)
        # Standard value at sea level (kPa/C)
        gamma = 0.067

        # 6. Net Radiation (Rn)
        if solar_rad is None:
            rn = self._estimate_solar_radiation(int(timestamp.strftime("%j")))
        else:
            rn = solar_rad

        # 7. FAO-56 Penman-Monteith Equation
        g = 0  # Soil heat flux is usually small on daily scale

        numerator = (0.408 * delta * (rn - g)) + (gamma * (900 / (self.temperature + 273)) * wind_speed * self.vpd)
        denominator = delta + (gamma * (1 + 0.34 * wind_speed))

        self.et0 = round(numerator / denominator,5)

    def _update_env_dict(self):
        self.current_condition = {"Temperature": self.temperature, "Humidity": round(self.humidity),
                                  "LightIntensity": self.light_intensity, "CO2": self.co2,}
