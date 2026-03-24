# Useful links:
#   -> farming article library: https://open.alberta.ca/interact/ropin-the-web
#   -> summary quick summary (seem reliable): https://icl-growingsolutions.com/agriculture/crops/wheat/

# ----- GLI: Green Leaf Index -----
GLI_THRESHOLD = {"Dead" : 0, "Critic" : 0.1, "Low" : 0.2, "Good" : 0.3}

# ----- NPK -----
# Based on the supply at BU-CROCCS
NPK_POWDER_RATIO = [[21,21,21], [28,6,5], [6,32,25], [15,10,35]]
# Fertilizer Requirements of Irrigated Grain and Oilseed Crops, alberta
# Minimal NPK concentration (1 ppm = 2 lb/ac) for soil. If below, add to get at least those values + margin
NPK_SOIL_RATIO_RANGE = {"N": [75,100],"P": [35, 50],"K": [125, 150]} # ppm
DEFICIENT_NPK_SOIL_RATIO = {"N": 20,"P": 10,"K": 25} # ppm
# wheat growth guide, https://webdoc.agsci.colostate.edu/wheat/linksfiles/UKAHDBgrowthstageguide.pdf
STAGE_SEQUENCE = ["GS30", "GS31", "GS39", "GS59", "GS61", "GS71", "GS87", "GS93"]
PLANT_STAGE_TIME = {"GS30": 10, "GS31": 39, "GS39": 18, "GS59": 5, "GS61": 9, "GS71": 39, "GS87": 11, "GS93": 7}

PLANT_STAGE_DIC = {"GS30": "pseudostem erect", "GS31": "First node detectable", "GS39": "Flag leaf blade all visible",
                   "GS59": "Ear completely emerged above flag leaf ligule", "GS61": "Start of flowering",
                   "GS71": "Grain watery ripe", "GS87": "Hard dough", "GS93": "Grain loosening in daytime"}
# Effect of NPK uptake at different growth stages of wheat (Triticum aestivum L.) for yield maximization. AN ASIAN JOURNAL OF SOIL SCIENCE, 9(2) https://doi.org/10.15740/HAS/AJSS/9.2/265-270
NPK_UPTAKE_RATIO = {"1-30": [20,16,19], "30-60": [37, 32, 30], "60-90": [16, 39, 36], "90-120": [23, 8, 12], "120-H": [4, 5, 3]}
NPK_REQUIRED_RATIO = [83,12.27,113.75] # in kg/ha [40, 6, 54] with 125 kg/ha of seeds
# Convert kg/ha to ppm (Ratio 2:1)
TOTAL_NPK_PPM = [val/2 for val in NPK_REQUIRED_RATIO]

# Uptake ratios mapped to your STATE_SEQUENCE
# We split the FI-FN [37, 32, 30] into two parts for TS
UPTAKE_MAPPING = {
    "GS30": [6.7, 5.3, 6.3],   # 1/3 of 1-30
    "GS31": [38, 32, 32.6],    # 2/3 of 1-30 + 2/3 of 30-60
    "GS39": [17.7, 24.6, 22],  # 1/3 30-60 + 1/3 60-90
    "GS59": [2.7, 6.5, 6],     # 1/6 60-90
    "GS61": [5.3, 13, 12],     # 1/3 60-90
    "GS71": [25.7, 14.5, 18],  # 1/6 60-90 + 90-120
    "GS87": [4, 5, 3],         # 120-H
    "GS93": [0, 0, 0]          # No more uptake at maturity
}
# GAI (Green Area Index) for wheat stage, will also be used for the FAO56
GAI = {
    "GS30": 1.6,
    "GS31": 2,
    "GS39": 6.2,
    "GS59": 6.3,
    "GS61": 6.3,
    "GS71": 5.7,
    "GS87": 1.3,
    "GS93": 0
}

# Height of the plant based on the wheat stage
HEIGHT = {
    "GS30": 5,
    "GS31": 9,
    "GS39": 34,
    "GS59": 53,
    "GS61": 69,
    "GS71": 69,
    "GS87": 69,
    "GS93": 69
}

# ----- CO2 & light-----
# The optimal atmospheric CO2 concentration for the growth of winter wheat (Triticum aestivum). Journal of Plant Physiology, 184, 89-97. https://doi.org/10.1016/j.jplph.2015.07.003
TARGET_CO2_RANGE = [890,910] # Wheat CO2 for optimal growth (ppm) EDIT: Realy depends on the light intensity
BASE_CO2 = 900
# Li, J., Zhang, Y., Cheng, R., & Li, T. (2025). Light Spectrum, Intensity, and Photoperiod Are Key for Production as Well as Speed Breeding of Spring Wheat in Indoor Farming. Plant-Environment Interactions, 6(5), e70085. https://doi.org/10.1002/pei3.70085
# Bugbee, B. G., & Salisbury, F. B. (1988). Exploring the Limits of Crop Productivity : I. Photosynthetic Efficiency of Wheat in High Irradiance Environments. Plant Physiology, 88(3), 869‑878. https://doi.org/10.1104/pp.88.3.869
IDEAL_LIGHT_INTENSITY = 2000 # PPFD #The higher the better based on bugbee and al
# IBID
CO2_TARGET_LIMITS = [400, 1200] # ppm

# Maybe useless ↓
# CO2 flux in a wheat-soybean succession in subtropical Brazil: A carbon sink. Journal of Environmental Quality, 51, 899–915. https://doi.org/10.1002/jeq2.20362
CO2_CONSUMPTION = 5.31 # Wheat CO2 Consumption (g CO₂ m⁻² day⁻¹)

# Let's assume 1,000,000 px = 1 m²
PX_TO_SQRT_METER = 1_000_000
PLANT_POT_SIZE_PX = round(0.12 * 0.15 * PX_TO_SQRT_METER) # 12*15 cm
ROOM_VOLUME_LITERS = 0.4*0.6*0.4 + (0.4*0.6*0.15)/2

# ----- Moisture & ET -----
# Fertilizer Requirements of Irrigated Grain and Oilseed Crops, alberta
SOIL_MOISTURE_RANGE = [60, 90] # The higher the vpd, the higher the soil moisture to avoid stress
CRITIC_SOIL_MOISTURE_RANGE = [40, 100]

# based on https://www.fao.org/4/x0490e/x0490e0b.htm Table 12
KC_MAPPING = {
    "GS30": 0.15,
    "GS31": 0.3,
    "GS39": 1.15,
    "GS59": 1.15,
    "GS61": 1.15,
    "GS71": 0.9,
    "GS87": 0.3,
    "GS93": 0.15}

# based on https://www.fao.org/4/x0490e/x0490e0b.htm
SOIL_KE = 1
REW = 9.0
TEW = 22.0
ZE = 0.1
KC_MIN = 0.15
FW = 0.6

# The maximum depth of available water in the pot (in mm/m) at 100% moisture. https://www.fao.org/4/r4082e/r4082e03.htm
MAX_WATER_DEPTH_MM = 170
SOIL_WATER_CAPACITY = 0.15 # https://www.fao.org/4/x0490e/x0490e0c.htm Table 19 (0FC - 0WP)
HEIGH_TO_ROOT_RATIO = 1.5 # NOTE: This is use for simplicity, it is not exactly true for every variety of wheat
P_RAW = 0.55 # https://www.fao.org/4/x0490e/x0490e0e.htm Table 22


# The Plant-Transpiration Response to Vapor Pressure Deficit (VPD) in Durum Wheat Is Associated With Differential Yield Performance and Specific Expression of Genes Involved in Primary Metabolism and Water Transport. Frontiers in Plant Science, 9, 1994. https://doi.org/10.3389/fpls.2018.01994
IDEAL_VPD_RANGE = [0.8, 1.2] # kPa
# Future heatwave conditions inhibit CO2-induced stomatal closure in wheat. The New phytologist, 249(3), 1234–1252. https://doi.org/10.1111/nph.70722
CRITICAL_VPD_RANGE = [0.5, 3.14] # kPa

LATITUDE_BANGKOK, LONGITUDE_BANGKOK = 13.754, 100.501
UTC_BANGKOK = +7

# ----- Other -----
# https://eos.com/blog/growing-wheat/#ref-anchor-1
IDEAL_TEMPERATURE_RANGE = [21, 24] # °C
CRITICAL_TEMPERATURE_RANGE = [4, 35] # °C

# https://eos.com/crop-management-guide/wheat-growth-stages/
SOIL_PH_RANGE = [6,7]


# ----- Timing Constants -----
DATA_UPDATE_MIN = 15  # 15 Minute intervals
ROBOT_IDLE_HOURS = 2  # Robot probes every 2 hours
TICKS_DAY = 24 * 60 / DATA_UPDATE_MIN

# Weather file
CSV_FILE = r"ExternalVariable/Meteo_Bangkok_2025.csv"

OUTPUT_CSV = "testv2.csv"