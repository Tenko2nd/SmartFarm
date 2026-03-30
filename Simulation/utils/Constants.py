import math

# ----- NPK -----
# Based on the supply at BU-CROCCS
NPK_POWDER_RATIO = [[21,21,21], [28,6,5], [6,32,25], [15,10,35]]

# ----- SIZE -----
# Let's assume 1,000,000 px = 1 m²
PX_TO_SQRT_METER = 1_000_000
PLANT_POT_DIMENSIONS_M = [0.12, 0.15] # 12, 15 cm
PLANT_POT_SIZE_M2= math.prod(PLANT_POT_DIMENSIONS_M)
ROOM_DIMENSIONS_M = [1, 1, 1] # Width, Height, Depth (meters)
ROOM_VOLUME_LITERS = math.prod(ROOM_DIMENSIONS_M) * 1000

# ----- LOCATION -----
LATITUDE_BANGKOK, LONGITUDE_BANGKOK = 13.754, 100.501
UTC_BANGKOK = +7

# ----- TIMING CONSTANTS -----
DATA_UPDATE_MIN = 15  # 15 Minute intervals
ROBOT_IDLE_HOURS = 2  # Robot probes every 2 hours
TICKS_DAY = 24 * 60 / DATA_UPDATE_MIN

# ----- FILES -----
CSV_FILE = r"ExternalVariable/Meteo_Bangkok_2025.csv"
OUTPUT_CSV = "testv2.csv"