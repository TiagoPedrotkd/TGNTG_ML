import pandas as pd

def summarize_and_sample(data, column, sample_size=5):
    display_df = pd.DataFrame(data[column].value_counts()).reset_index()
    display_df.columns = [column, "Counts"]
    display_df["Percentage"] = (display_df["Counts"] / display_df["Counts"].sum()) * 100

    actual_sample_size = min(sample_size, len(data))

    sample_rows = data.sample(n=actual_sample_size)

    return display_df, sample_rows