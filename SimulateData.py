import random
from datetime import timedelta
from pprint import pprint

import pandas as pd
import numpy as np
import math

#TODO: At the end if this script is robust, use it to train the model in reinforcement learning.

# Maybe use other growth stages? https://en.wikipedia.org/wiki/Cereal_growth_staging_scales

# Useful links:
#   -> farming article library: https://open.alberta.ca/interact/ropin-the-web
#   -> summary quick summary (seem reliable): https://icl-growingsolutions.com/agriculture/crops/wheat/

# ==========================================
# 1. PARAMETERS & CONFIGURATION
# ==========================================
# TODO: Create a plant object for facilitation of manipulation
plants_values = {"PLANT_01":
                     {"COORDINATES": [0, 1], "N": 200, "P": 75, "K": 200, "green_px": 500, "GLI": 0.0,
                      "necrotic_spot": 0, "soil_moisture": 95, "state": "FI", "state_pct" : 0},
                 "PLANT_02":
                     {"COORDINATES": [0, 2], "N": 100, "P": 50, "K": 200, "green_px": 430, "GLI": 0.0,
                      "necrotic_spot": 0, "soil_moisture": 70, "state": "FI", "state_pct" : 47},
                 "PLANT_03":
                     {"COORDINATES": [1, 1], "N": 200, "P": 50, "K": 100, "green_px": 723, "GLI": 0.0,
                      "necrotic_spot": 0, "soil_moisture": 56, "state": "H", "state_pct" : 12}}

# ----- GLI: Green Leaf Index -----
GLI_THRESHOLD = {"Dead" : 0, "Critic" : 0.1, "Low" : 0.2, "Good" : 0.3}

# ----- NPK -----
# Based on the supply at BU-CROCCS
NPK_POWDER_RATIO = [[21,21,21], [28,6,5], [6,32,25], [15,10,35]]
# Fertilizer Requirements of Irrigated Grain and Oilseed Crops, alberta
# Minimal NPK concentration (1 ppm = 2 lb/ac) for soil. If below, add to get at least those values + margin
NPK_SOIL_RATIO_RANGE = {"N": [75,100],"P": [35, 50],"K": [125, 150]} # ppm
DEFICIENT_NPK_SOIL_RATIO = {"N": 20,"P": 10,"K": 25} # ppm
# Time of wheat developpment at different stage https://www.fao.org/4/y4011e/y4011e06.htm#bm06
STATE_SEQUENCE = ["E", "FI", "TS", "FN", "H", "A", "M"]
PLANT_STATE_TIME = {"E": 20, "FI": 25, "TS": 15, "FN": 30, "H": 10, "A": 40, "M": 10}
PLANT_STATE_DIC = {"E": "Emergence", "FI": "Floral initiation", "TS": "Terminal spikelet", "FN": "First node",
                   "H": "Heading", "A": "Anthesis", "M": "Maturity"}
# Effect of NPK uptake at different growth stages of wheat (Triticum aestivum L.) for yield maximization. AN ASIAN JOURNAL OF SOIL SCIENCE, 9(2) https://doi.org/10.15740/HAS/AJSS/9.2/265-270
NPK_UPTAKE_RATIO = {"E-FI": [20,16,19], "FI-FN": [37, 32, 30], "FN-H": [16, 39, 36], "H-A": [23, 8, 12], "A-M": [4, 5, 3]}
NPK_REQUIRED_RATIO = [83,12.27,113.75] # in kg/ha [40, 6, 54] with 125 kg/ha of seeds
# Convert kg/ha to ppm (Ratio 2:1)
TOTAL_NPK_PPM = [val/2 for val in NPK_REQUIRED_RATIO]

# Uptake ratios mapped to your STATE_SEQUENCE
# We split the FI-FN [37, 32, 30] into two parts for TS
UPTAKE_MAPPING = {
    "E":  [20, 16, 19],    # Progressing from Emergence to FI
    "FI": [18.5, 16, 15],  # Progressing from FI to TS (half of FI-FN)
    "TS": [18.5, 16, 15],  # Progressing from TS to FN (half of FI-FN)
    "FN": [16, 39, 36],    # Progressing from FN to H
    "H":  [23, 8, 12],     # Progressing from H to A
    "A":  [4, 5, 3],       # Progressing from A to M
    "M":  [0, 0, 0]        # No more uptake at maturity
}

#TODO: Based on Plant growth stage, refill with water put ratio NPK
#TODO: Plant will consume NPK with proportion (if more N then more N consume)

# ----- CO2 & light-----
# The optimal atmospheric CO2 concentration for the growth of winter wheat (Triticum aestivum). Journal of Plant Physiology, 184, 89-97. https://doi.org/10.1016/j.jplph.2015.07.003
TARGET_CO2_RANGE = [890,910] # Wheat CO2 for optimal growth (ppm) EDIT: Realy depends on the light intensity
BASE_CO2 = 900
# Li, J., Zhang, Y., Cheng, R., & Li, T. (2025). Light Spectrum, Intensity, and Photoperiod Are Key for Production as Well as Speed Breeding of Spring Wheat in Indoor Farming. Plant-Environment Interactions, 6(5), e70085. https://doi.org/10.1002/pei3.70085
# Bugbee, B. G., & Salisbury, F. B. (1988). Exploring the Limits of Crop Productivity : I. Photosynthetic Efficiency of Wheat in High Irradiance Environments. Plant Physiology, 88(3), 869‑878. https://doi.org/10.1104/pp.88.3.869
IDEAL_LIGHT_INTENSITY = [700,1000] # PPFD
# IBID
CO2_TARGET_LIMITS = [400, 1200] # ppm

# Maybe useless ↓
# CO2 flux in a wheat-soybean succession in subtropical Brazil: A carbon sink. Journal of Environmental Quality, 51, 899–915. https://doi.org/10.1002/jeq2.20362
CO2_CONSUMPTION = 5.31 # Wheat CO2 Consumption (g CO₂ m⁻² day⁻¹)

# Let's assume 100,000 px = 1 m²
PX_TO_SQRT_METER = 100_000
PLANT_POT_SIZE_PX = round(0.12 * 0.15 * PX_TO_SQRT_METER) # 12*15 cm
ROOM_VOLUME_LITERS = 0.4*0.6*0.4 + (0.4*0.6*0.15)/2

# ----- Moisture & ET -----
# Fertilizer Requirements of Irrigated Grain and Oilseed Crops, alberta
SOIL_MOISTURE_RANGE = [60, 90] # The higher the vpd, the higher the soil moisture to avoid stress
CRITIC_SOIL_MOISTURE_RANGE = [40, 100]
# The Plant-Transpiration Response to Vapor Pressure Deficit (VPD) in Durum Wheat Is Associated With Differential Yield Performance and Specific Expression of Genes Involved in Primary Metabolism and Water Transport. Frontiers in Plant Science, 9, 1994. https://doi.org/10.3389/fpls.2018.01994
IDEAL_VPD_RANGE = [0.8, 1.2] # kPa
# Future heatwave conditions inhibit CO2-induced stomatal closure in wheat. The New phytologist, 249(3), 1234–1252. https://doi.org/10.1111/nph.70722
CRITICAL_VPD_RANGE = [0.5, 3.14] # kPa

LATITUDE_BANGKOK_RADIAN = 0.240

# ----- Other -----
# https://eos.com/blog/growing-wheat/#ref-anchor-1
IDEAL_TEMPERATURE_RANGE = [21, 24] # °C
CRITICAL_TEMPERATURE_RANGE = [4, 35] # °C

# https://eos.com/crop-management-guide/wheat-growth-stages/
SOIL_PH_RANGE = [6,7]


# ----- Timing Constants -----
DATA_HEARTBEAT = 15  # 15 Minute intervals
ROBOT_IDLE_HOURS = 2  # Robot probes every 2 hours

#TODO: From moisture measure ration NPK asume concentration then predict output of sensor

CSV_FILE = r"WeatherJanv2026/Final/WeatherJanv2026.csv"

# ==========================================
# 2. BIOLOGICAL & ENVIRONMENTAL FUNCTIONS
# ==========================================

def estimate_solar_radiation(latitude, jour_annee, t_max, t_min):
    """
    Estime le rayonnement solaire (Rs) en MJ/m2/jour
    Méthode Hargreaves-Samani (recommandée par la FAO-56).
    Args:
        latitude (float): latitude en radian
        jour_annee (int): Jour de l'année.
        t_max (float): Température maximale de la journée
        t_min (float): Température minimale de le journée
    """
    # 1. Calcul de la radiation extra-terrestre (Ra)
    # Latitude en radians
    lat_rad = (math.pi / 180) * latitude

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
    rs = k_rs * math.sqrt(t_max - t_min) * ra

    return rs

# TODO: Use it only once a day for performance
def get_day_temp_extremes(target_date):
    """
    target_date should be a string in 'YYYY-MM-DD' format
    """
    # Load the data
    df = pd.read_csv(CSV_FILE)

    # Convert 'time' column to datetime objects
    df['time'] = pd.to_datetime(df['time'])

    # Filter for the specific day
    day_data = df[df['time'].dt.strftime('%Y-%m-%d') == target_date]

    if day_data.empty:
        return None, None

    max_temp = day_data['temp'].max()
    min_temp = day_data['temp'].min()

    return max_temp, min_temp

def calculate_vpd(temp_c, humidite_relative):
    """
    Calculate VPD (Vapor Pressure Deficit) in kPa.
    """
    # Saturation Vapor Pressure (es)
    es = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))

    # Actual Vapor Pressure (ea)
    ea = es * (humidite_relative / 100.0)

    # VPD
    vpd = es - ea

    # Slope of Vapor Pressure Curve (delta)
    delta = (4098 * es) / math.pow((temp_c + 237.3), 2)

    return vpd, delta

def calculate_fao56_et0(temp_c, relative_humidity, wind_speed, jour_annee, solar_rad=None):
    """
    Calculates the Reference Evapotranspiration (ET0) based on the FAO-56
    Penman-Monteith method.
    Lu, Y., Ma, D., Chen, X., & Zhang, J. (2018). A Simple Method for Estimating Field Crop Evapotranspiration from Pot Experiments. Water, 10(12), 1823. https://doi.org/10.3390/w10121823


    Args:
        temp_c (float): Air temperature in degrees Celsius.
        relative_humidity (float): Relative humidity as a percentage (0-100).
        wind_speed (float): Wind speed at 2m height (m/s).
        solar_rad (float): Net radiation (MJ/m2/day). If None, it estimates based on temp.
        jour_annee (date): Date of the day, format 'YYYY-MM-DD'

    Returns:
        float: Estimated ET0 in mm/day.
    """
    vpd, delta = calculate_vpd(temp_c, relative_humidity)

    # 5. Psychrometric Constant (gamma)
    # Standard value at sea level (kPa/C)
    gamma = 0.067

    # 6. Net Radiation (Rn)
    if solar_rad is None:
        tmax, tmin = get_day_temp_extremes(jour_annee)
        rn = estimate_solar_radiation(LATITUDE_BANGKOK_RADIAN, jour_annee, tmax, tmin)
    else:
        rn = solar_rad

    # 7. FAO-56 Penman-Monteith Equation
    g = 0  # Soil heat flux is usually small on daily scale

    numerator = (0.408 * delta * (rn - g)) + (gamma * (900 / (temp_c + 273)) * wind_speed * vpd)
    denominator = delta + (gamma * (1 + 0.34 * wind_speed))

    et0 = numerator / denominator

    return round(et0, 2)

def calculate_sunlight(hour):
    """Simulates PPFD (umol/m2/s) based on a 24h cycle."""
    # Bangkok january estimation
    if 6.5 <= hour <= 18:
        # Sine curve for natural light progression
        intensity = 950 * np.sin(np.pi * (hour - 6.5) / 11.5)
        return round(max(0, intensity), 2)
    return 0.0

def add_realistic_noise(value, noise_level=0.01):
    """Adds Gaussian noise to sensor readings."""
    return max(0,value + np.random.normal(0, noise_level))

def calculate_room_co2_drawdown(plant_size_m2, light_intensity_ppf, current_co2_ppm, interval_minutes, temp_c,
        room_pressure_hpa):
    """
    Calculates the remaining CO2 in a room after plant absorption.
    Based on NASA TM 102788 (Wheeler & Sager) : 'Carbon Dioxide And Water Exchange Rates By A Wheat Crop In NASA'S
    Biomass Production Chamber: Results From An 86-Day Study (January To April 1989)',
    and Gruda et al. (2025) : 'Environmental conditions and nutritional quality of vegetables in protected cultivation'

    Args:
        plant_size_m2 (float): Total vegetative area
        light_intensity_ppf (int): Photosynthetic Photon Flux (umol/m2/s)
        current_co2_ppm (int): Starting concentration
        interval_minutes (int): Elapsed time
        temp_c (float): Room temperature in Celsius
        room_pressure_hpa (float): Room pressure in hPa
    """

    # 1. CONSTANTS & GAS PHYSICS
    R = 0.08206  # Ideal Gas Constant (L*atm / K*mol)
    temp_k = temp_c + 273.15
    # Calculate Molar Volume of air at current temp (L/mol)
    pressure_atm = room_pressure_hpa / 1013.25
    molar_volume = R * temp_k / pressure_atm

    # 2. CALCULATE NET UPTAKE RATE (umol/m2/s)
    # Equation from NASA study (Fig 7): y = 0.054784x - 9.6297
    # -9.6297 represents the respiration at 20°C.
    base_respiration = 9.6297

    # Adjust respiration for temperature (Article 2: 75% increase from 16C to 24C)
    # This roughly equates to a 8% change per degree Celsius from a 20C baseline
    # FIXME: Research as it annot be just a straight line
    temp_factor = 1 + (temp_c - 20) * 0.08
    adjusted_respiration = base_respiration * temp_factor

    gross_photosynthesis = 0.054784 * light_intensity_ppf
    net_uptake_rate = gross_photosynthesis - adjusted_respiration

    # 3. ADJUST FOR CO2 CONCENTRATION LIMITATION
    # Article 2 shows rate is stable from 800-2200ppm but drops below 800.
    if current_co2_ppm < 800:
        # Linear scaling factor: at 800ppm = 1.0, at 190ppm (compensation point) = 0.0
        co2_factor = max(0, (current_co2_ppm - 190) / (800 - 190))
        net_uptake_rate *= co2_factor
    elif current_co2_ppm > 2200:
        # Article 2 notes slight decrease/saturation above 2200
        net_uptake_rate *= 0.9

        # 4. CALCULATE TOTAL QUANTITY ABSORBED
    total_seconds = interval_minutes * 60
    total_mol_absorbed = (net_uptake_rate * plant_size_m2 * total_seconds) / 1_000_000 # convert umol to mol

    # 5. CONVERT ABSORBED MICROMOLES TO PPM CHANGE IN ROOM
    # ppm = (micromoles_of_gas / total_moles_of_air)
    total_moles_air_in_room = ROOM_VOLUME_LITERS / molar_volume
    delta_ppm = total_mol_absorbed / total_moles_air_in_room

    final_co2_ppm = current_co2_ppm - delta_ppm

    return round(final_co2_ppm, 2)

def calculate_plant_size_farm_m2():
    total_size = sum(plant.get("green_px", 0) for plant in plants_values.values()) / PX_TO_SQRT_METER
    return total_size

# TODO: Make critic index values have consequences later
#  (eg. A plant has critic vpd, 5 hours later it appears visible problems if not fixed)
#TODO: Change smooth curve for mors realistic curve? research realistic data to see how it react to each variable
def optimal_condition_index(plant_id, co2, light, temperature, humidity):
    """
    The pourcentage of optimal condition respected for the plant to grow perfectly.
    100% the plant grow perfectly, 0% it's dying
    :return: The optimal condition index in %
    """
    idx_vpd = optimal_vpd(plant_id, temperature, humidity)
    moisture = plants_values[plant_id]["soil_moisture"]
    idx_moisture = optimal_soil(moisture, temperature, humidity)
    data = {x: plants_values[plant_id][x] for x in ["N", "P", "K"]}
    idx_npk = optimal_npk(data["N"], data["P"], data["K"])
    idx_co2_light = optimal_co2_light_synergy(co2, light)

    # Define Importance Coefficients
    weights = {
        "co2_light": 1.0,
        "moisture": 1.0,
        "vpd": 0.8,
        "npk": 0.5 # TODO: Not vital for 1 or 2 days but if not fixed becomes vital
    }

    # Calculate Weighted Average
    scores = {
        "co2_light": idx_co2_light,
        "moisture": idx_moisture,
        "vpd": idx_vpd,
        "npk": idx_npk
    }

    weighted_sum = sum(scores[key] * weights[key] for key in weights)
    total_weights = sum(weights.values())

    average_index = weighted_sum / total_weights

    #  Apply Liebig's Law (The "Limiting Factor" Penalty)
    # We take the minimum of the "Vitals" (Water, Climate, Light)
    vitals_min = min(idx_co2_light, idx_moisture, idx_vpd)

    # We blend the average and the minimum.
    # If vitals_min is 0, the final index will be heavily penalized.
    final_index = (average_index * 0.6) + (vitals_min * 0.4)

    return round(final_index / 100, 3)


def optimal_vpd(plant_id, temp_c, relative_humidity):
    """
    Calcule l'optimalité du climat pour le blé dur.
    """

    vpd, _ = calculate_vpd(temp_c, relative_humidity)

    min_ideal, max_ideal = IDEAL_VPD_RANGE
    min_crit, max_crit = CRITICAL_VPD_RANGE

    # Critic limits
    # TODO: yellow leaves
    if vpd <= min_crit  or vpd >= max_crit:
        return 0.0

    # Optimal state
    if min_ideal <= vpd <= max_ideal:
        return 100.0


    # Interpolation
    if vpd < min_ideal:
        t = (vpd - min_crit) / (min_ideal - min_crit)
    else:
        t = (max_crit - vpd) / (max_crit - max_ideal)

    # Quadratic smoothing
    score = (3 * t ** 2 - 2 * t ** 3)
    return round(score * 100, 2)


def optimal_soil(current_moisture, temperature, humidity):
    """
    Calculates soil moisture optimality (0-100%) dynamically based on VPD.

    Args:
        current_moisture (float): Actual soil moisture percentage (0-100).
        temperature (float): Actual temperature (Celsius).
        humidity (int): Actual humidity (0-100%).
    """
    vpd, _ = calculate_vpd(temperature, humidity)
    vpd_low, vpd_high = CRITICAL_VPD_RANGE
    moisture_low, moisture_high = SOIL_MOISTURE_RANGE  # Minimum and Maximum ideal targets
    min_crit, max_crit = CRITIC_SOIL_MOISTURE_RANGE

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
        return 0.0

    # Case B: Within a small buffer around the dynamic target (Perfect zone)
    # We allow a +/- 5% tolerance for 100% score
    if (dynamic_target - 5) <= current_moisture <= (dynamic_target + 5):
        return 100.0

    # Case C: Between Critical Min and Target (Drought Stress)
    if current_moisture < dynamic_target:
        t = (current_moisture - min_crit) / ((dynamic_target - 5) - min_crit)
        score = (3 * t ** 2 - 2 * t ** 3)  # Smoothstep
        return round(score * 100, 2)

    # Case D: Above Target (Saturation / Over-watering)
    # Higher is better than lower: the score only drops to 60% health at saturation
    if current_moisture > dynamic_target:
        # Distance from target to 100% moisture
        t = (max_crit - current_moisture) / (max_crit - (dynamic_target + 5))
        smooth = (3 * t ** 2 - 2 * t ** 3)
        # We remap the score so it goes from 100% down to 50% (instead of 0%)
        score = 0.5 + (0.5 * smooth)
        return round(score * 100, 2)

    return 0.0


def score_single_nutrient(current_ppm, ideal_range, deficient_val):
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
        return 100.0

    # 3. Transition from Deficient to Ideal (Smoothstep curve)
    # Ratio between 0 and 1
    t = (current_ppm - deficient_val) / (min_ideal - deficient_val)
    score = (3 * t ** 2 - 2 * t ** 3)

    return round(score * 100, 2)


def optimal_npk(n_ppm: int, p_ppm: int, k_ppm: int):
    """
    Calculates the global NPK optimality index based on Alberta Agriculture data.
    """
    # Configuration based on your provided values (ppm)

    # Calculate individual scores
    scores = {
        "N": score_single_nutrient(n_ppm, NPK_SOIL_RATIO_RANGE["N"], DEFICIENT_NPK_SOIL_RATIO["N"]),
        "P": score_single_nutrient(p_ppm, NPK_SOIL_RATIO_RANGE["P"], DEFICIENT_NPK_SOIL_RATIO["P"]),
        "K": score_single_nutrient(k_ppm, NPK_SOIL_RATIO_RANGE["K"], DEFICIENT_NPK_SOIL_RATIO["K"])
    }

    # The Global Index follows the 'Law of the Minimum'
    # Your plant is only as healthy as its most deficient nutrient.
    global_index = min(scores.values())

    return global_index


def optimal_co2_light_synergy(co2_ppm, ppfd):
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
        return 100.0 if co2_ppm < 2000 else 0.0

    # 2. INTENSITY SAFETY SCORE (Is the light too strong for the species?)
    _, max_ideal_light = IDEAL_LIGHT_INTENSITY
    light_score = 1.0
    if ppfd > 1000:
        t_light = max(0, (1800 - ppfd) / (1800 - 1000))
        light_score = (3 * t_light ** 2 - 2 * t_light ** 3)

    # 3. ADEQUACY SCORE (The Balance)
    # We calculate the "Target CO2" for the current light intensity.
    target_co2 = ppfd * 1.1

    # Clamp target between ambient (400) and max useful (1200)
    min_co2, max_co2 = CO2_TARGET_LIMITS
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

    return round(max(0, final_score) * 100, 2)


def grow_plants_step(p_id, idx_optimal):
    """
    Updates the simulation for a period of x hours.
    """
    data = plants_values[p_id]
    step_hours = ROBOT_IDLE_HOURS

    # Update State Percentage
    current_state = data["state"]
    days_in_state = PLANT_STATE_TIME[current_state]
    hours_in_state = days_in_state * 24 * 3 # NOTE: Speed up for simulation

    # Calculate hourly progress adjusted by plant health
    progress = (step_hours / hours_in_state) * min(idx_optimal+0.3, 1) # For margin (0.7 is good for growth and it doesn't stop completetly)

    data = consume_nutrients(data, progress)

    data["state_pct"] += progress

    if current_state != "M":
        if data["state_pct"] >= 100:
            idx_in_seq = STATE_SEQUENCE.index(current_state)
            next_state = STATE_SEQUENCE[idx_in_seq + 1]
            data["state"] = next_state
            data["state_pct"] = 0

    # Green Pixels & Necrosis Logic
    growth_speed = 0.005

    # GLI is a proxy for 'green density'.
    pct_val = data["state_pct"] / 1000
    stage_factor = 0
    growth_factor = 0

    match current_state:
        case "M":
            stage_factor = 1.0 - pct_val * 4 # green decrease from 1 to 0.6
            growth_factor = -data["green_px"] * data["state_pct"] / 100
        case "A":
            stage_factor = 1.0 # green is steady at 1
            growth_factor = 1.0
        case "H":
            stage_factor = 0.8 + pct_val * 2 # green increase from 0.8 to 1
            growth_factor = data["green_px"] * growth_speed
        case "FN":
            stage_factor = 0.5 + pct_val * 3
            growth_factor = data["green_px"] * growth_speed * 2
        case "TS":
            stage_factor = 0.3 + pct_val * 2
            growth_factor = data["green_px"] * growth_speed * 5
        case "FI":
            stage_factor = 0.1 + pct_val * 2
            growth_factor = data["green_px"] * growth_speed * 4
        case "E":
            stage_factor = pct_val
            growth_factor = data["state_pct"]
        case _:
            stage_factor = 0.0
            growth_factor = 0.0

    # Base GLI = Current health * max possible GLI * stage maturity
    # Max GLI 0.8 for realism
    calculated_gli = idx_optimal * stage_factor * 0.8

    data["GLI"] = round(max(0.0, min(0.6, calculated_gli)), 3)

    if idx_optimal > 0.6:  # Healthy growth
        new_px = (idx_optimal - 0.5) * growth_factor * step_hours
        data["green_px"] += int(new_px)
        # TODO: Necrotic spot are more complexe! to improve (delayed if thirst, etc...) + take in consideration night time
        if data["necrotic_spot"]>0:
            data["necrotic_spot"] -= int(new_px / (24/step_hours))
            data["green_px"] += int(new_px / (24/step_hours))
    elif idx_optimal < 0.4:  # Stress leads to necrosis
        damage = (0.5 - idx_optimal) * growth_factor * step_hours
        data["green_px"] -= int(damage)
        data["necrotic_spot"] += int(damage)

    plants_values[p_id] = data
    if p_id == "PLANT_01":
        print(idx_optimal)
        pprint(data)

    return


def consume_nutrients(plant_data, progress_pct):
    """
    Reduces N, P, K in the soil based on the growth progress of the current state.
    :param plant_data: The dictionary of the specific plant
    :param progress_pct: The percentage of state progress made in this step (float)
    """
    state = plant_data["state"]

    if state not in UPTAKE_MAPPING or state == "M":
        return plant_data

    # Get the distribution ratio for the current state
    ratios = UPTAKE_MAPPING[state]

    # Calculation:
    # (Total PPM needed for life) * (% allocated to this state) * (% of state completed now)
    consumption_n = TOTAL_NPK_PPM[0] * (ratios[0] / 100) * (progress_pct / 100)
    consumption_p = TOTAL_NPK_PPM[1] * (ratios[1] / 100) * (progress_pct / 100)
    consumption_k = TOTAL_NPK_PPM[2] * (ratios[2] / 100) * (progress_pct / 100)

    # Reduce soil values (ensuring they don't go below 0)
    plant_data["N"] = max(0, round(plant_data["N"] - consumption_n, 4))
    plant_data["P"] = max(0, round(plant_data["P"] - consumption_p, 4))
    plant_data["K"] = max(0, round(plant_data["K"] - consumption_k, 4))

    return plant_data



# ==========================================
# 3. DATA PROCESSING ENGINE
# ==========================================

def create_smart_farm_db(input_csv_path):
    # A. Load and Resample Hourly Weather Data
    # Expecting: time,temp,rhum,prcp,pres
    df_weather = pd.read_csv(input_csv_path, parse_dates=['time'])
    df_weather = df_weather.set_index('time')

    # Upsample from 1 hour to 15 minutes
    df_resampled = df_weather.resample(f'{DATA_HEARTBEAT}min').interpolate(method='cubicspline')

    final_rows = []

    co2_in_farm = BASE_CO2


    # B. Generate data for each timestamp
    for ts, row in df_resampled.iterrows():
        air_temp = add_realistic_noise(row['temp'], 0.1)
        humidity = add_realistic_noise(row['rhum'], 0.1)
        pressure = add_realistic_noise(row['pres'], 0.1)
        light = add_realistic_noise(calculate_sunlight(ts.hour),2)
        co2 = add_realistic_noise(co2_in_farm, 1)

        # C. Broadcast ambient data to all 3 plants
        # Check if robot is probing (Every 2 hours on the dot)
        is_probed = 1 if (ts.hour % ROBOT_IDLE_HOURS == 0 and ts.minute == 0) else 0

        for i, (p_id, values) in enumerate(plants_values.items()):

            # Add slight delta for robot time to take measure
            # maybe model will be better with timestamp of initial command (so ambiant values changes a bit?)
            if is_probed:
                ts += timedelta(seconds=random.randint(30,45))

            p_data = {
                "TimeStamp": ts,
                "Plant_ID": p_id,
                "Coordinates": values["COORDINATES"],
                "Air_temp": air_temp,
                "Humidity": humidity,
                "Light_intensity": light,
                "CO2": co2,
                "IsProbed": is_probed
            }


            # Robot Specific Data (Only 'Measured' if IsProbed == 1)
            if is_probed:
                optimal_idx = optimal_condition_index(p_id, co2, light, air_temp, humidity)
                grow_plants_step(p_id, optimal_idx)
                p_data["N"] = values["N"]
                p_data["P"] = values["P"]
                p_data["K"] = values["K"]
                p_data["green_px"] = values["green_px"]
                # p_data["Size"] = values["size"]
                # p_data["Leaf_temp"] = p_data["Air_temp"] - 1.8
                # p_data["Soil_Moisture"] = values["soil_moisture"]
            else:
                p_data["NPK"] = np.nan
                p_data["RGB_Metrics"] = np.nan
                p_data["Leaf_temp"] = np.nan
                p_data["Soil_Moisture"] = np.nan

            final_rows.append(p_data)
            final_rows.append(p_data)


        co2_in_farm = calculate_room_co2_drawdown(
                plant_size_m2=calculate_plant_size_farm_m2(),
                light_intensity_ppf=light,
                current_co2_ppm=co2_in_farm,
                interval_minutes=DATA_HEARTBEAT,
                temp_c=air_temp,
                room_pressure_hpa=pressure,
            )

    # D. Final Assembly and Forward Fill
    master_df = pd.DataFrame(final_rows)

    # Group by Plant_ID so ffill doesn't leak data between different plants
    master_df = master_df.groupby("Plant_ID", group_keys=False).apply(lambda x: x.ffill())

    return master_df


# ==========================================
# 4. EXECUTION
# ==========================================

# Run Engine
smart_farm_data = create_smart_farm_db(CSV_FILE)

# Display result
smart_farm_data.to_csv("test.csv", index=False)