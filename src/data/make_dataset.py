import pandas as pd

from src.utils.config import load_config
from src.utils.store import AssignmentStore


def main():
    store = AssignmentStore()
    config = load_config()

    #order info clean- remove dup based on all cols 
    booking_df = store.get_raw("booking_log.csv")
    booking_df = clean_booking_df(booking_df)

    #driver info clean- remove dup based on all cols
    participant_df = store.get_raw("participant_log.csv")
    participant_df = clean_participant_df(participant_df)

    #left join part w booking
    dataset = merge_dataset(booking_df, participant_df)
    
    #create col target 
    dataset = create_target(dataset, config["target"])

    df_list = [dataset,participant_df,booking_df]
    df_names = ["dataset.csv","participant_log.csv","booking_log.csv"]

    for df,name in zip(df_list,df_names):
        store.put_processed(name, df)

def clean_booking_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df
    # unique_columns = [
    #     "order_id",
    #     "trip_distance",
    #     "pickup_latitude",
    #     "pickup_longitude",
    # ]
    # df = df.drop_duplicates(subset=unique_columns)
    # return df[unique_columns] 


def clean_participant_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df


def merge_dataset(bookings: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(bookings, participants, on="order_id", how="inner", suffixes=('_booking','_participant'))
    return df


def create_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df[target_col] = df["participant_status"].apply(lambda x: int(x == "ACCEPTED"))
    return df


if __name__ == "__main__":
    main()
