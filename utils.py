import datetime

import gensim
import gensim.downloader as gloader
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


def load_embedding_model(model_name, embedding_dimension=50):
    """
    Loads a pre-trained word embedding model via gensim library
    """
    model = f"{model_name}-{embedding_dimension}"
    try:
        return gloader.load(model)
    except Exception as e:
        print("Invalid embedding model name.")
        raise e