from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

import pandas as pd
import seaborn as sns
import joblib
import io
from matplotlib import pyplot as plt

pipe = joblib.load("pipeline.joblib")


class ModelOperation:
    df_mod = pd.DataFrame()

    def __init__(self, df):
        self.df = df

    @staticmethod
    def pipeline(df):
        arr = pipe[:-1].fit_transform(df)
        columns = pipe[:-1].get_feature_names_out()
        return arr, columns

    def return_eng_df(self):
        (df, columns) = self.pipeline(self.df)
        df = pd.DataFrame(df, columns=columns)

        self.df_mod = df

    def before_plot_kde(self):
        kde = sns.kdeplot(self.df)
        fig = kde.get_figure()
        return fig

    def after_plot_kde(self):
        df = self.df_mod
        columns_unique = [column for column in df.columns if df[column].nunique() != 2]
        kde = sns.kdeplot(df[columns_unique])
        fig = kde.get_figure()
        return fig


app = FastAPI()


@app.get("/iris-before")
async def read_root():
    df_iris = sns.load_dataset("iris")
    obj = ModelOperation(df_iris)
    obj.return_eng_df()
    fig = obj.before_plot_kde()

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    plt.close(fig)

    return Response(img_buf.getvalue(), media_type="image/png")


@app.get("/iris-after")
async def read_root():
    df_iris = sns.load_dataset("iris")
    obj = ModelOperation(df_iris)
    obj.return_eng_df()
    fig = obj.after_plot_kde()

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    plt.close(fig)

    return Response(img_buf.getvalue(), media_type="image/png")
