import pandas as pd
import numpy as np
import math

#TODO: At the end if this script is robust, use it to train the model in reinforcment learning.

# ==========================================
# 1. PARAMETERS & CONFIGURATION
# ==========================================
plants_values = {"PLANT_01" : {"COORDINATES":[0,1], "N":200, "P":75, "K":200, "pixels": 500, "size": 70},
                 "PLANT_02": {"COORDINATES": [0, 2], "N": 100, "P": 50, "K": 200, "pixels": 430, "size": 53},
                 "PLANT_03": {"COORDINATES": [1, 1], "N": 200, "P": 50, "K": 100, "pixels": 723, "size": 85},
                 }

# ----- NPK -----
# Based on the supply at BU-CROCCS
NPK_POWDER_RATIO = [[21,21,21], [28,6,5], [6,32,25], [15,10,35]]
# Accumulation of Nutrients (NPK) at Different Growth Stages of Machine Transplanted Rice (Oryza sativa L.) Under Different Levels of Nitrogen and Split Schedules
# T-AT = transplantation to active tillering, PI = Pinnacle Initiation, F = Flowering, M = Maturation
NPK_UPTAKE_RATIO = {"T-AT": [20.9,20.6,19.4], "AT-PI": [33.7,43.4,36.5], "PI-F": [39.1,26.7,31.1], "F-M": [6.3,9.6,13]}
# Estimation of NPK requirements for rice production in diverse Chinese environments under optimal fertilization rates
# In this study, the estimated N, P, and K required to produce 1 Mg of rice grain were 21.0, 4.4, and 22.1kg in southern China
NPK_REQUIRED_RATIO = [21,4.4,22.1]
# to get the required in the soil for each stage
NPK_SOIL_RATIO = {
    key: [(val / 100) * NPK_REQUIRED_RATIO[i] for i, val in enumerate(values)]
    for key, values in NPK_UPTAKE_RATIO.items()
}

#TODO: Based on PLant growth stage, refill with water put ratio
#TODO: PLant will consume NPK with proportion (if more N then more N consume)
#TODO : RESEARCH → CO2 Absorption for wheat based on pixels

# ----- CO2 -----
# The optimal atmospheric CO2 concentration for the growth of winter wheat (Triticum aestivum). Journal of Plant Physiology, 184, 89-97. https://doi.org/10.1016/j.jplph.2015.07.003
TARGET_CO2_RANGE = {890,910} # Wheat CO2 for optimal growth (ppm)
# CO2 flux in a wheat-soybean succession in subtropical Brazil: A carbon sink. Journal of Environmental Quality, 51, 899–915. https://doi.org/10.1002/jeq2.20362
CO2_CONSUMPTION = 5.31 # Wheat CO2 Consumption (g CO₂ m⁻² day⁻¹)

BASE_SOIL_MOISTURE = 45.0  # %
LATITUDE_BANGKOK_RADIAN = 0.240


# Timing Constants
DATA_HEARTBEAT = '15min'  # 15 Minute intervals
ROBOT_IDLE_HOURS = 2  # Robot probes every 2 hours

#TODO: From moisture measure ration NPK asume concentration then predict output of sensor


# ==========================================
# 2. BIOLOGICAL & ENVIRONMENTAL FUNCTIONS
# ==========================================

def estimer_rayonnement_solaire(latitude, jour_annee, t_max, t_min):
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

def get_temperature_jour(jour_annee):
    #TODO: get the max and min of the temperature of the day.
    return 0, 0

def calculate_fao56_et0(temp_c, relative_humidity, wind_speed=2.0, solar_rad=None, jour_annee=0):
    """
    Calculates the Reference Evapotranspiration (ET0) based on the FAO-56
    Penman-Monteith method.
    Lu, Y., Ma, D., Chen, X., & Zhang, J. (2018). A Simple Method for Estimating Field Crop Evapotranspiration from Pot Experiments. Water, 10(12), 1823. https://doi.org/10.3390/w10121823


    Args:
        temp_c (float): Air temperature in degrees Celsius.
        relative_humidity (float): Relative humidity as a percentage (0-100).
        wind_speed (float): Wind speed at 2m height (m/s). Default is 2.0 (FAO standard).
        solar_rad (float): Net radiation (MJ/m2/day). If None, it estimates based on temp.

    Returns:
        float: Estimated ET0 in mm/day.
    """

    # 1. Saturation Vapor Pressure (es)
    # Formula based on FAO-56 Annex
    es = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))

    # 2. Actual Vapor Pressure (ea)
    ea = es * (relative_humidity / 100.0)

    # 3. Vapor Pressure Deficit (VPD)
    vpd = es - ea

    # 4. Slope of Vapor Pressure Curve (delta)
    delta = (4098 * es) / math.pow((temp_c + 237.3), 2)

    # 5. Psychrometric Constant (gamma)
    # Standard value at sea level (kPa/C)
    gamma = 0.067

    # 6. Net Radiation (Rn)
    # If not provided, we use a simplified estimation for a sunny day
    # typical of the "monsoon climate" described on Page 2 of the PDF.
    if solar_rad is None:
        tmax, tmin = get_temperature_jour(jour_annee)
        rn = estimer_rayonnement_solaire(LATITUDE_BANGKOK_RADIAN, jour_annee, tmax, tmin)
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
    # Peak light at 13:00 (1:00 PM), zero light before 6am and after 7pm
    if 6 <= hour <= 19:
        # Sine curve for natural light progression
        intensity = 500 * np.sin(np.pi * (hour - 6) / 13)
        return round(max(0, intensity), 2)
    return 0.0


def add_realistic_noise(value, noise_level=0.01):
    """Adds Gaussian noise to sensor readings."""
    return max(0,value + np.random.normal(0, noise_level))


def simulate_co2(hour, light_intensity):
    #TODO: use the plants to determine CO2 absorption and add refill when threshold crossed
    """CO2 levels drop during peak light due to photosynthesis."""
    return 1


# ==========================================
# 3. DATA PROCESSING ENGINE
# ==========================================

def create_smart_farm_db(input_csv_path):
    # A. Load and Resample Hourly Weather Data
    # Expecting: time,temp,rhum,prcp,pres
    df_weather = pd.read_csv(input_csv_path, parse_dates=['time'])
    df_weather = df_weather.set_index('time')

    # Upsample from 1 hour to 15 minutes
    df_resampled = df_weather.resample(DATA_HEARTBEAT).interpolate(method='linear')

    final_rows = []

    # B. Generate data for each timestamp
    for ts, row in df_resampled.iterrows():
        air_temp = add_realistic_noise(row['temp'], 0.1)
        humidity = add_realistic_noise(row['rhum'], 0.1)
        light = add_realistic_noise(calculate_sunlight(ts.hour),2)
        co2 = add_realistic_noise(simulate_co2(ts.hour, light), 0.01)


        # C. Broadcast ambient data to all 3 plants
        for p_id, values in plants_values.items():
            # Check if robot is probing (Every 2 hours on the dot)
            is_probed = 1 if (ts.hour % ROBOT_IDLE_HOURS == 0 and ts.minute == 0) else 0

            # Simulate Plant Metrics
            # NPK slightly drifts/decreases unless fertilized
            # RGB Metrics: [nb_green_pixels, size_index]
            # Leaf temp usually Air Temp - 1.5C (if transpiring well)

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
                p_data["Green_Pixels"] = values["pixels"]
                p_data["Size"] = values["size"]
                p_data["Leaf_temp"] = p_data["Air_temp"] - 1.8
                p_data["Soil_Moisture"] = add_realistic_noise(BASE_SOIL_MOISTURE, 0.05)
            else:
                p_data["NPK"] = np.nan
                p_data["RGB_Metrics"] = np.nan
                p_data["Leaf_temp"] = np.nan
                p_data["Soil_Moisture"] = np.nan

            final_rows.append(p_data)

    # D. Final Assembly and Forward Fill
    master_df = pd.DataFrame(final_rows)

    # Group by Plant_ID so ffill doesn't leak data between different plants
    master_df = master_df.groupby("Plant_ID", group_keys=False).apply(lambda x: x.ffill())

    return master_df


# ==========================================
# 4. EXECUTION
# ==========================================

# Run Engine
smart_farm_data = create_smart_farm_db('WeatherJanv2026/Final/WeatherJanv2026.csv')

# Display result
print(smart_farm_data.head(15))
smart_farm_data.to_csv("final_smart_farm_dataset2.csv", index=False)