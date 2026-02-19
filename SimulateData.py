import random
from datetime import timedelta
import pandas as pd
import numpy as np
import math

#TODO: At the end if this script is robust, use it to train the model in reinforcment learning.

# Maybe use other growth stages? https://en.wikipedia.org/wiki/Cereal_growth_staging_scales

# Useful links:
#   -> farming article library: https://open.alberta.ca/interact/ropin-the-web
#   -> summary quick summary (seem reliable): https://icl-growingsolutions.com/agriculture/crops/wheat/

# ==========================================
# 1. PARAMETERS & CONFIGURATION
# ==========================================
# TODO: Create a plant object for facilitation of manipulation
plants_values = {"PLANT_01":
                     {"COORDINATES": [0, 1], "N": 200, "P": 75, "K": 200, "green_px": 500, "GLI": 0,
                      "necrotic_spot": 0, "soil_moisture": 40},
                 "PLANT_02":
                     {"COORDINATES": [0, 2], "N": 100, "P": 50, "K": 200, "green_px": 430, "GLI": 0,
                      "necrotic_spot": 0, "soil_moisture": 40},
                 "PLANT_03":
                     {"COORDINATES": [1, 1], "N": 200, "P": 50, "K": 100, "green_px": 723, "GLI": 0,
                      "necrotic_spot": 0, "soil_moisture": 40}}

# ----- GLI: Green Leaf Index -----
GLI_THRESHOLD = {"Dead" : 0, "Critic" : 0.1, "Low" : 0.2, "Good" : 0.3}

# ----- NPK -----
# Based on the supply at BU-CROCCS
NPK_POWDER_RATIO = [[21,21,21], [28,6,5], [6,32,25], [15,10,35]]
# Fertilizer Requirements of Irrigated Grain and Oilseed Crops, alberta
# Minimal NPK concentration (ppm) for soil. If below, add to get at least those values + margin
NPK_SOIL_RATIO = {25, 45, 125}
# Maybe not usefull ↓
# Accumulation of Nutrients (NPK) at Different Growth Stages of Machine Transplanted Rice (Oryza sativa L.) Under Different Levels of Nitrogen and Split Schedules
# T-AT = transplantation to active tillering, PI = Pinnacle Initiation, F = Flowering, M = Maturation
NPK_UPTAKE_RATIO = {"T-AT": [20.9,20.6,19.4], "AT-PI": [33.7,43.4,36.5], "PI-F": [39.1,26.7,31.1], "F-M": [6.3,9.6,13]}
# Estimation of NPK requirements for rice production in diverse Chinese environments under optimal fertilization rates
# In this study, the estimated N, P, and K required to produce 1 Mg of rice grain were 21.0, 4.4, and 22.1kg in southern China
NPK_REQUIRED_RATIO = [21,4.4,22.1]

#TODO: Based on PLant growth stage, refill with water put ratio NPK
#TODO: PLant will consume NPK with proportion (if more N then more N consume)

# ----- CO2 -----
# The optimal atmospheric CO2 concentration for the growth of winter wheat (Triticum aestivum). Journal of Plant Physiology, 184, 89-97. https://doi.org/10.1016/j.jplph.2015.07.003
TARGET_CO2_RANGE = [890,910] # Wheat CO2 for optimal growth (ppm)
# CO2 flux in a wheat-soybean succession in subtropical Brazil: A carbon sink. Journal of Environmental Quality, 51, 899–915. https://doi.org/10.1002/jeq2.20362
CO2_CONSUMPTION = 5.31 # Wheat CO2 Consumption (g CO₂ m⁻² day⁻¹)
BASE_CO2 = 900

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

    Parameters:
    plant_size_m2: Total vegetative area
    light_intensity_ppf: Photosynthetic Photon Flux (umol/m2/s)
    current_co2_ppm: Starting concentration
    interval_minutes: Elapsed time
    temp_c: Room temperature in Celsius
    room_pressure_hpa: Room pressure in hPa
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
    total_umol_absorbed = net_uptake_rate * plant_size_m2 * total_seconds

    # 5. CONVERT ABSORBED MICROMOLES TO PPM CHANGE IN ROOM
    # ppm = (micromoles_of_gas / total_moles_of_air)
    total_moles_air_in_room = ROOM_VOLUME_LITERS / molar_volume
    delta_ppm = total_umol_absorbed / total_moles_air_in_room

    final_co2_ppm = current_co2_ppm - delta_ppm

    return round(final_co2_ppm, 2)

def calculate_plant_size_farm_m2():
    total_size = sum(plant.get("green_px", 0) for plant in plants_values.values()) / PX_TO_SQRT_METER
    return total_size

# TODO: Make critic index values have consequences later
#  (eg. A plant has critic vpd, 5 hours later it appears visible problems if not fixed)
def optimal_condition_index(plant_id, co2, light, temperature, humidity):
    """
    The pourcentage of optimal condition respected for the plant to grow perfectly.
    100% the plant grow perfectly, 0% it's dying
    :return: The optimal condition index in %
    """
    index_vpd = optimal_vpd(plant_id, temperature, humidity)
    moisture = plants_values[plant_id]["soil_moisture"]
    index_moisture = optimal_soil(moisture, temperature, humidity)

    return 0


def optimal_vpd(plant_id, temp_c, relative_humidity):
    """
    Calcule l'optimalité du climat pour le blé dur.
    """

    vpd, _ = calculate_vpd(temp_c, relative_humidity)

    min_ideal, max_ideal = IDEAL_VPD_RANGE
    min_crit, max_crit = CRITICAL_VPD_RANGE

    # Critic limits
    # TODO: yellow leaves
    if vpd <= min_crit or vpd >= max_crit:
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
                p_data["N"] = values["N"]
                p_data["P"] = values["P"]
                p_data["K"] = values["K"]
                p_data["green_px"] = values["green_px"]
                p_data["Size"] = values["size"]
                p_data["Leaf_temp"] = p_data["Air_temp"] - 1.8
                p_data["Soil_Moisture"] = values["soil_moisture"]
            else:
                p_data["NPK"] = np.nan
                p_data["RGB_Metrics"] = np.nan
                p_data["Leaf_temp"] = np.nan
                p_data["Soil_Moisture"] = np.nan

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
print(smart_farm_data.head(15))
smart_farm_data.to_csv("final_smart_farm_dataset2.csv", index=False)