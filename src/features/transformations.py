import pandas as pd
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp_booking"].apply(robust_hour_of_iso_date)
    return df



def hour_of_day_test(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df



def driver_historical_completed_bookings(booking: pd.DataFrame, merged_data_v2: pd.DataFrame) -> pd.DataFrame:
    
    complete_count = booking[booking['booking_status']=='COMPLETED'].groupby('driver_id').size().reset_index(name='compl_count')
    merged_data_v2 = pd.merge(complete_count, merged_data_v2, left_on="driver_id", right_on = "driver_id_booking", how="inner").drop(columns='driver_id')
    
    return merged_data_v2


def driver_historical_completed_bookings_test(booking: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    
    complete_count = booking[booking['booking_status']=='COMPLETED'].groupby('driver_id').size().reset_index(name='compl_count')
    df = pd.merge(complete_count, df, on="driver_id", how="inner")
    
    return df