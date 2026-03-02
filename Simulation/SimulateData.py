import os
import random
from datetime import timedelta, datetime
from tqdm import tqdm

import pandas as pd
import numpy as np
import math

from Simulation.Class.FarmClass import Farm
import Simulation.utils.Constants as C

#TODO: At the end if this script is robust, use it to train the model in reinforcement learning.

#TODO: Effect are delayed (low moisture cause yellowish some hours laters)
#TODO: Effect of long time ( if light is low for too long it causes damage but if it is low only 1 hours it is okay)
#TODO: Implement stress state for data (data should have plants that growth well and plant that dies for the model to learn)
#TODO: Refill NPK and CO2 if critical level are reached (might depend on the stress states)
#TODO: Make a dead state for the plant (if vital has been too low for too long or necrosis >90%, stop the data of this plant)
#TODO: Make a file for different farm environment in order to give the model multiple scenarii

#TODO: Based on Plant growth stage, refill with water put ratio NPK
#TODO: Make sigmoid for a more realistic data in different stages (NPK, Growth, etc) at beginning of stage slow then fast then slow

class SimulationManager:
    def __init__(self, weather_csv):
        self.weather_df = self._init_weather_dataframe(weather_csv)
        self.farms = [Farm(i, 3) for i in range(1)]

        self.data_storage = []
        self.current_running_month = None

        self.today = self.weather_df.index[0].normalize()
        self.daily_temp_extremes = self.weather_df['temp'].resample('D').agg(['min', 'max'])

    def run(self):
        # 1. Vectorize what we can (Speed up calculation)
        self.weather_df['is_probed'] = ((self.weather_df.index.hour % C.ROBOT_IDLE_HOURS == 0) &
                                        (self.weather_df.index.minute == 0)).astype(int)

        # GROUP BY DAY
        daily_groups = self.weather_df.groupby(pd.Grouper(freq='D'))

        # Wrap the groups in tqdm to see the progress bar moving day-by-day
        for day_timestamp, day_df in tqdm(daily_groups, desc="Simulating Days"):

            # DAILY UPDATES (Run once per day)
            self.today = day_timestamp
            self._update_today_data()

            for row in day_df.itertuples():
                ts = row.Index

                # Daily updates
                if ts.normalize() > self.today:
                    self.today = ts.normalize()
                    self._update_today_data()

                env = {
                    "Temperature": row.temp,
                    "Humidity": row.rhum,
                    "Pressure": row.pres,
                    "LightIntensity": self._calculate_sunlight(timestamp=ts)
                }

                for farm in self.farms:
                    farm.current_time = ts
                    farm.update_farm_environment(env=env)
                    for plant in farm.plants:
                        if row.is_probed:
                            plant.record_probe()
                        self.data_storage.append({
                            "timestamp": ts,
                            "plant_id": plant.id,
                            **farm.current_condition,
                            "is_probed": row.is_probed,
                            **plant.last_observation  # Unpack observation dictionary
                        })

            # daily flush
            self._flush_to_csv()

    def _flush_to_csv(self):
        """Converts the current list to a DataFrame and appends to the CSV file."""
        if not self.data_storage:
            return

        df = pd.DataFrame(self.data_storage)

        file_exists = os.path.isfile(C.OUTPUT_CSV)
        df.to_csv(C.OUTPUT_CSV, mode='a', index=False, header=not file_exists)

        self.data_storage = []

    @staticmethod
    def _init_weather_dataframe(weather_csv):
        weather_df = pd.read_csv(weather_csv)
        weather_df['time'] = pd.to_datetime(weather_df['time'])
        weather_df = weather_df.set_index('time').sort_index()
        # Upsample from 1 hour to 15 minutes
        weather_df = weather_df.resample(f'{C.DATA_UPDATE_MIN}min').interpolate(method='cubicspline')
        return weather_df

    def _update_today_data(self):
        temp_extreme = self._get_day_temp_extremes(self.today)
        if temp_extreme is not None:
            for farm in self.farms:
                farm.update_extreme_temp(temp_extreme)

    def _get_day_temp_extremes(self, target_date):
        try:
            stats = self.daily_temp_extremes.loc[target_date]
            return stats.to_dict()
        except KeyError:
            return None

    # NOTE: For smartfarm with possibility to control the light, using the DLI (daily light integral) might be better
    @staticmethod
    def _calculate_sunlight(timestamp: datetime, max_ppfd=2000):
        """
        Calculates instantaneous PPFD based on Latitude, Longitude, and Time.

        Args:
            timestamp (datetime): Time at the moment
            max_ppfd (int): Peak intensity under clear sky (standard is ~2000)
        """
        # Convert Latitude to Radians
        lat_rad = math.radians(C.LATITUDE_BANGKOK)

        # Get time information
        day_of_year = int(timestamp.strftime("%j"))
        hour_of_day = timestamp.hour + timestamp.minute / 60.0

        # Calculate Equation of Time (EoT)
        b = math.radians((360 / 365) * (day_of_year - 81))
        eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)

        # Calculate Solar Time Correction
        # Standard Meridian is 15 degrees per hour of UTC offset
        standard_meridian = C.UTC_BANGKOK * 15
        # Difference in minutes between local time and solar time
        time_correction = 4 * (C.LONGITUDE_BANGKOK - standard_meridian) + eot
        solar_time = hour_of_day + (time_correction / 60)

        # Solar Declination (Angle of sun relative to equator)
        declination = math.radians(23.45 * math.sin(math.radians((360 / 365) * (day_of_year - 80))))

        # Hour Angle (How many degrees the sun has moved from solar noon)
        # Solar noon is at 12:00 Solar Time
        hour_angle = math.radians((solar_time - 12) * 15)

        # Solar Elevation Angle (alpha)
        sin_elevation = (math.sin(lat_rad) * math.sin(declination) +
                         math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle))

        # Final PPFD Calculation
        if sin_elevation <= 0:
            return 0.0  # It's night

        # PPFD is proportional to the sine of the elevation angle
        return max_ppfd * sin_elevation


if __name__ == '__main__':
    sim = SimulationManager(weather_csv=C.CSV_FILE)
    sim.run()