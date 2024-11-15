from fastapi import FastAPI, Response, Request


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import io


class ModelOperation:
    df_mod = pd.DataFrame()

    def __init__(self, df):
        self.df = df

    def creating_pipeline(self):
        numeric_features = self.df.select_dtypes(include=["float64", "int64"]).columns
        categorical_features = self.df.select_dtypes(include=["object"]).columns

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="passthrough",
        )
        pipe = Pipeline(
            [
                ("Preprocessing", preprocessor),
                ("Outlier", LocalOutlierFactor(n_neighbors=20, contamination=0.1)),
            ],
            verbose=True,
        )
        return pipe

    def transform(self):
        pipe = self.creating_pipeline()
        arr = pipe[:-1].fit_transform(self.df)
        columns = pipe[:-1].get_feature_names_out()
        df = pd.DataFrame(arr, columns=columns)

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


@app.post("/data-before")
async def read_root(request: Request):
    data = await request.json()
    dataset = data["dataset"]
    df = sns.load_dataset(dataset)
    obj = ModelOperation(df)
    fig = obj.before_plot_kde()

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    plt.close(fig)

    return Response(img_buf.getvalue(), media_type="image/png")


@app.post("/data-after")
async def read_root(request: Request):
    data = await request.json()
    dataset = data["dataset"]
    df = sns.load_dataset(dataset)
    obj = ModelOperation(df)
    obj.transform()
    fig = obj.after_plot_kde()

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    plt.close(fig)

    return Response(img_buf.getvalue(), media_type="image/png")
