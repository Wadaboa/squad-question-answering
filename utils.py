import datetime

from IPython.display import display, HTML


def show_df_row(df, index):
    """
    Show the given DataFrame row in a way
    that's enjoyable on Jupyter
    """
    row = df.iloc[index]
    display(HTML(pd.DataFrame([row]).to_html()))


def get_run_name():
    """
    Return a unique wandb run name
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
