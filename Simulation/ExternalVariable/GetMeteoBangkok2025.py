from datetime import date
import matplotlib.pyplot as plt
import meteostat as ms

# Specify location and time range
POINT = ms.Point(13.75398, 100.50144, 12) # BANGKOK LOCATION
START = date(2025, 1, 1)
END = date(2025, 12, 31)

# Get nearby weather stations
stations = ms.stations.nearby(POINT, limit=4)

# Get daily data & perform interpolation
ts = ms.hourly(stations, START, END)
df = ms.interpolate(ts, POINT).fetch()

df = df[["temp", "rhum", "prcp", "wspd", "pres"]]

df.to_csv('Meteo_Bangkok_2025.csv')
