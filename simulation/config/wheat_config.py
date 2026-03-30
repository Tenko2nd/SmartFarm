from simulation.classes.stress_buffer import StressBuffer

# Useful links:
#   -> farming article library: https://open.alberta.ca/interact/ropin-the-web
#   -> summary quick summary (seem reliable): https://icl-growingsolutions.com/agriculture/crops/wheat/
#   -> water consumption and other for multiple types of plants: https://www.fao.org/4/x0490e/x0490e00.htm
#   -> Farming library articles: https://www.fao.org/

#----- STAGES -----
# wheat growth guide, https://webdoc.agsci.colostate.edu/wheat/linksfiles/UKAHDBgrowthstageguide.pdf
STAGE_SEQUENCE = ["GS30", "GS31", "GS39", "GS59", "GS61", "GS71", "GS87", "GS93"]
PLANT_STAGE_TIME = {"GS30": 10, "GS31": 39, "GS39": 18, "GS59": 5, "GS61": 9, "GS71": 39, "GS87": 11, "GS93": 7}

PLANT_STAGE_DIC = {"GS30": "pseudostem erect", "GS31": "First node detectable", "GS39": "Flag leaf blade all visible",
                   "GS59": "Ear completely emerged above flag leaf ligule", "GS61": "Start of flowering",
                   "GS71": "Grain watery ripe", "GS87": "Hard dough", "GS93": "Grain loosening in daytime"}

# ----- NPK -----
# Fertilizer Requirements of Irrigated Grain and Oilseed Crops, alberta research for wheat
# Minimal NPK concentration (1 ppm = 2 lb/ac) for soil. If below, add to get at least those values + margin
NPK_SOIL_RATIO_RANGE = {"N": [75,100],"P": [35, 50],"K": [125, 150]} # ppm
DEFICIENT_NPK_SOIL_RATIO = {"N": 20,"P": 10,"K": 25} # ppm

# Effect of NPK uptake at different growth stages of wheat (Triticum aestivum L.) for yield maximization. AN ASIAN JOURNAL OF SOIL SCIENCE, 9(2) https://doi.org/10.15740/HAS/AJSS/9.2/265-270
NPK_UPTAKE_RATIO = {"1-30": [20,16,19], "30-60": [37, 32, 30], "60-90": [16, 39, 36], "90-120": [23, 8, 12], "120-H": [4, 5, 3]} # %
NPK_REQUIRED_RATIO = [83,12.27,113.75] # in kg/ha [40, 6, 54] with 125 kg/ha of seeds
# Convert kg/ha to ppm (Ratio 2:1)
TOTAL_NPK_PPM = [val/2 for val in NPK_REQUIRED_RATIO]

# Uptake ratios mapped to your STATE_SEQUENCE
# We split the FI-FN [37, 32, 30] into two parts for TS
NPK_UPTAKE_MAPPING = {
    "GS30": [6.7, 5.3, 6.3],   # 1/3 of 1-30
    "GS31": [38, 32, 32.6],    # 2/3 of 1-30 + 2/3 of 30-60
    "GS39": [17.7, 24.6, 22],  # 1/3 30-60 + 1/3 60-90
    "GS59": [2.7, 6.5, 6],     # 1/6 60-90
    "GS61": [5.3, 13, 12],     # 1/3 60-90
    "GS71": [25.7, 14.5, 18],  # 1/6 60-90 + 90-120
    "GS87": [4, 5, 3],         # 120-H
    "GS93": [0, 0, 0]          # No more uptake at maturity
}

# ----- GAI -----
# wheat growth guide, https://webdoc.agsci.colostate.edu/wheat/linksfiles/UKAHDBgrowthstageguide.pdf
# GAI (Green Area Index) for wheat stage
GAI_MAPPING = {
    "GS30": 1.6,
    "GS31": 2,
    "GS39": 6.2,
    "GS59": 6.3,
    "GS61": 6.3,
    "GS71": 5.7,
    "GS87": 1.3,
    "GS93": 0
}

# ----- HEIGHT -----
# wheat growth guide, https://webdoc.agsci.colostate.edu/wheat/linksfiles/UKAHDBgrowthstageguide.pdf
# Height of the plant based on the wheat stage
HEIGHT_MAPPING = {
    "GS30": 5,
    "GS31": 9,
    "GS39": 34,
    "GS59": 53,
    "GS61": 69,
    "GS71": 69,
    "GS87": 69,
    "GS93": 69
}
HEIGH_TO_ROOT_RATIO = 1.5 # NOTE: This is use for simplicity, it is not exactly true for every variety of wheat

# ----- CO2 -----
# The optimal atmospheric CO2 concentration for the growth of winter wheat (Triticum aestivum). Journal of Plant Physiology, 184, 89-97. https://doi.org/10.1016/j.jplph.2015.07.003
TARGET_CO2_RANGE = [890,910] # Wheat CO2 for optimal growth (ppm) EDIT: Realy depends on the light intensity
# Bugbee, B. G., & Salisbury, F. B. (1988). Exploring the Limits of Crop Productivity : I. Photosynthetic Efficiency of Wheat in High Irradiance Environments. Plant Physiology, 88(3), 869‑878. https://doi.org/10.1104/pp.88.3.869
CO2_TARGET_LIMITS = [400, 1200] # ppm

# ----- LIGHT -----
# Li, J., Zhang, Y., Cheng, R., & Li, T. (2025). Light Spectrum, Intensity, and Photoperiod Are Key for Production as Well as Speed Breeding of Spring Wheat in Indoor Farming. Plant-Environment Interactions, 6(5), e70085. https://doi.org/10.1002/pei3.70085
# Bugbee, B. G., & Salisbury, F. B. (1988). Exploring the Limits of Crop Productivity : I. Photosynthetic Efficiency of Wheat in High Irradiance Environments. Plant Physiology, 88(3), 869‑878. https://doi.org/10.1104/pp.88.3.869
IDEAL_LIGHT_INTENSITY = 2000 # PPFD #The higher the better based on bugbee and al
# Extinction coefficient for vertical-leaf wheat (Monsi & Saeki, 1953)
K_EXTINCTION = 0.6

# ----- Moisture -----
# Fertilizer Requirements of Irrigated Grain and Oilseed Crops, alberta
SOIL_MOISTURE_RANGE = [60, 90] # The higher the vpd, the higher the soil moisture to avoid stress
CRITIC_SOIL_MOISTURE_RANGE = [40, 100]

# ----- ET0 -----
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
P_RAW = 0.55 # https://www.fao.org/4/x0490e/x0490e0e.htm Table 22

# ----- VDP -----
# The Plant-Transpiration Response to Vapor Pressure Deficit (VPD) in Durum Wheat Is Associated With Differential Yield Performance and Specific Expression of Genes Involved in Primary Metabolism and Water Transport. Frontiers in Plant Science, 9, 1994. https://doi.org/10.3389/fpls.2018.01994
IDEAL_VPD_RANGE = [0.8, 1.2] # kPa
# Future heatwave conditions inhibit CO2-induced stomatal closure in wheat. The New phytologist, 249(3), 1234–1252. https://doi.org/10.1111/nph.70722
CRITICAL_VPD_RANGE = [0.5, 3.14] # kPa

# ----- TEMPERATURE -----
# https://eos.com/blog/growing-wheat/#ref-anchor-1
IDEAL_TEMPERATURE_RANGE = [21, 24] # °C
CRITICAL_TEMPERATURE_RANGE = [4, 35] # °C

# ----- PH -----
# https://eos.com/crop-management-guide/wheat-growth-stages/
SOIL_PH_RANGE = [6,7]

# ----- WAYS TO DIE (or to suffer) -----
# Research Threshold to effect and kind of effect
stress_buffers = {
    # ASPHYXIATION (Waterlogging)
    # Ref: Roots die in 48-72h. Visual yellowing is delayed.
    # Adjustment: Reduced days_to_effect. If a field is underwater for 10 days, wheat is usually dead.
    "Asphyxiation": StressBuffer(
        name="Asphyxiation",
        days_to_effect=(7, 10),    # Kills faster than 14 days in most soil types.
        recovery_days=7,           # Slow: needs to regrow root hairs.
        threshold_visible_effect=0.3, # 30% stress in: starts yellowing.
        is_fatal=True
    ),

    # DROUGHT
    # Ref: Wheat is drought-tolerant but has a "Permanent Wilting Point."
    # Adjustment: Threshold is very low because wilting/leaf-rolling is an immediate visible signal.
    "Drought": StressBuffer(
        name="Drought",
        days_to_effect=(14, 24),   # Depending on humidity/temp.
        recovery_days=4,           # Fast: Turgor pressure restores quickly if re-watered.
        threshold_visible_effect=0.1, # 10% stress in: leaves roll/turn blue-gray.
        is_fatal=True
    ),

    # SALINITY
    # Ref: Two-phase death: 1. Osmotic (immediate), 2. Ionic (slow poisoning).
    # Adjustment: Long time to kill, but recovery is very difficult (near impossible without leaching).
    "Salinity": StressBuffer(
        name="Salinity",
        days_to_effect=(25, 35),   # Takes a long time for salt to reach lethal levels in leaves.
        recovery_days=15,          # Very slow: plant must physically export ions or grow new tissue.
        threshold_visible_effect=0.05, # Visible almost immediately as stunted growth.
        is_fatal=True
    ),

    # HEAT
    # Ref: Wheat enzymes (Rubisco) denature at 35°C+.
    # Adjustment: Fatal if sustained. Threshold high because wheat hides heat stress until it "scorches."
    "Heat": StressBuffer(
        name="Heat",
        days_to_effect=(5, 8),     # 7-10 was too generous; 5 days of 40°C kills wheat.
        recovery_days=3,           # Fast metabolic reset if temp drops.
        threshold_visible_effect=0.5, # Plant looks okay until tips suddenly turn white/brown.
        is_fatal=True
    ),

    # FROST
    # Ref: Ice crystals puncture cells.
    # Adjustment: This is the fastest killer.
    "Frost": StressBuffer(
        name="Frost",
        days_to_effect=(1, 2),     # One bad night can kill the crown.
        recovery_days=14,          # Massive recovery time: must regrow entire tillers from the base.
        threshold_visible_effect=0.3, # Visible within 24h as "water-soaked" leaves.
        is_fatal=True
    ),

    # ETIOLATION (Low Light)
    # Ref: Carbon starvation.
    "Etiolation": StressBuffer(
        name="Etiolation",
        days_to_effect=(15, 25),
        recovery_days=10,          # Stem weakness is a "permanent" scar that takes time to reinforce.
        threshold_visible_effect=0.2,
        is_fatal=True
    ),

    # NITROGEN DEFICIENCY
    # Ref: N is mobile. Plant moves it from old leaves to new.
    "N_Deficiency": StressBuffer(
        name="N_Deficiency",
        days_to_effect=(35, 50),   # Hard to kill wheat with just low N; it just stays tiny.
        recovery_days=7,           # Green-up happens in a week once N is applied.
        threshold_visible_effect=0.2,
        is_fatal=True
    ),

    # PHOSPHORUS DEFICIENCY
    # Ref: Critical for energy. "Zombie" syndrome.
    "P_Deficiency": StressBuffer(
        name="P_Deficiency",
        days_to_effect=(50, 70),
        recovery_days=14,          # P moves slowly in the plant; recovery is sluggish.
        threshold_visible_effect=0.3, # Purple stems appear late.
        is_fatal=False             # Usually stunts rather than kills.
    ),

    # POTASSIUM DEFICIENCY
    # Ref: Controls stomata. Low K = fake drought.
    "K_Deficiency": StressBuffer(
        name="K_Deficiency",
        days_to_effect=(25, 35),
        recovery_days=7,
        threshold_visible_effect=0.25, # Leaf margin "burn" is the tell.
        is_fatal=True
    ),
}
