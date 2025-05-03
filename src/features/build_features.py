import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.transformations import (
    driver_distance_to_pickup,
    driver_historical_completed_bookings,
    driver_historical_completed_bookings_test,
    hour_of_day, hour_of_day_test
)
from src.utils.store import AssignmentStore


def main():
    store = AssignmentStore()

    booking = store.get_processed("booking_log.csv")
    dataset = store.get_processed("dataset.csv")
    
    dataset = apply_feature_engineering(booking, dataset)

    store.put_processed("transformed_dataset.csv", dataset)


def apply_feature_engineering(booking: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    
    df = driver_historical_completed_bookings(booking,df)

    df = df.pipe(driver_distance_to_pickup).pipe(hour_of_day)
    df.drop(columns=['event_timestamp_participant','customer_id','experiment_key','event_timestamp_booking','pickup_latitude','pickup_longitude','driver_latitude','driver_longitude','driver_id_booking'], inplace=True)    

    df = pd.get_dummies(df, columns=['booking_status', 'participant_status'], dtype=int)

    return df    
        
    

def apply_feature_engineering_test(booking: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    
    df = driver_historical_completed_bookings_test(booking,df)

    df = df.pipe(driver_distance_to_pickup).pipe(hour_of_day_test)
    df.drop(columns=['event_timestamp','pickup_latitude','pickup_longitude','driver_latitude','driver_longitude'], inplace=True)    

    return df  

if __name__ == "__main__":
    main()
