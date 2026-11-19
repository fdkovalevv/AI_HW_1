from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "ridge_cat_fe.pickle"
TRAIN_URL = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
st.set_page_config(page_title="Оценка автомобиля", layout="wide")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_data():
    df = pd.read_csv(TRAIN_URL)
    for col in ["mileage", "engine", "max_power"]:
        df[col] = pd.to_numeric(
            df[col].astype("string").str.extract(r"(\d+(?:\.\d+)?)", expand=False),
            errors="coerce"
        )
    return df


if not MODEL_PATH.exists():
    st.error("Нет файла ridge_cat_fe.pickle. Сначала сохраните модель из ноутбука.")
    st.stop()

artifact = load_model()
model = artifact["model"]
encoder = artifact["encoder"]
data = load_data()

features = [
    "name", "year", "km_driven", "fuel", "seller_type", "transmission",
    "owner", "mileage", "engine", "max_power", "torque", "seats", "max_torque_rpm"
]


def prepare_features(df):
    X = df[features].copy()

    for col in artifact["missing_columns"]:
        X[f"{col}_missing"] = X[col].isna().astype(int)

    year_centered = X["year"] - artifact["year_center"]
    engine_liters = X["engine"].where(X["engine"] > 0) / 1000
    X["power_per_liter"] = X["max_power"] / engine_liters
    X["year_squared"] = year_centered ** 2
    X["log_km_driven"] = np.log1p(X["km_driven"])
    X["log_torque"] = np.log1p(X["torque"])
    X["power_squared"] = X["max_power"] ** 2
    X["power_x_year"] = X["max_power"] * year_centered
    X["log_km_x_year"] = np.log1p(X["km_driven"]) * year_centered

    numeric_cols = artifact["numeric_columns"]
    X[numeric_cols] = X[numeric_cols].fillna(artifact["medians"])

    X["brand"] = X["name"].apply(lambda value: value.split(" ")[0])
    X["model"] = X["name"].apply(lambda value: value.split(" ")[1])
    X = X.drop(columns="name")

    cat_cols = artifact["categorical_columns"]
    encoded = pd.DataFrame(
        encoder.transform(X[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X.index
    )
    X = pd.concat([X.drop(columns=cat_cols), encoded], axis=1)
    return X[artifact["feature_columns"]]


st.title("Оценка стоимости автомобиля")
st.caption("Ridge-регрессия с новыми признаками")

prediction_tab, eda_tab, weights_tab = st.tabs(["Прогноз", "EDA", "Коэффициенты"])

with prediction_tab:
    st.subheader("Один автомобиль")
    with st.form("car"):
        left, right = st.columns(2)
        with left:
            name = st.text_input("Название", "Maruti Swift Dzire VDI")
            year = st.number_input("Год выпуска", value=2015, step=1)
            km_driven = st.number_input("Пробег, км", min_value=0, value=60000)
            mileage = st.number_input("Экономичность, как в датасете", min_value=0.0, value=18.9)
            engine = st.number_input("Объём двигателя, см^3", min_value=1.0, value=1248.0)
            max_power = st.number_input("Мощность, bhp", min_value=0.0, value=74.0)
            torque = st.number_input("Крутящий момент, Н*м", min_value=0.0, value=190.0)
        with right:
            fuel = st.selectbox("Топливо", data["fuel"].dropna().unique())
            seller_type = st.selectbox("Тип продавца", data["seller_type"].dropna().unique())
            transmission = st.selectbox("Коробка передач", data["transmission"].dropna().unique())
            owner = st.selectbox("Владелец", data["owner"].dropna().unique())
            seats = st.number_input("Количество мест", min_value=1, value=5)
            max_torque_rpm = st.number_input("Обороты максимального момента", min_value=0.0, value=2000.0)
        submitted = st.form_submit_button("Предсказать цену")

    if submitted:
        car = pd.DataFrame([{
            "name": name, "year": year, "km_driven": km_driven,
            "fuel": fuel, "seller_type": seller_type, "transmission": transmission,
            "owner": owner, "mileage": mileage, "engine": engine,
            "max_power": max_power, "torque": torque, "seats": seats,
            "max_torque_rpm": max_torque_rpm
        }])
        prediction = model.predict(prepare_features(car))[0]
        st.metric("Прогноз цены", f"{prediction:,.0f}")

    st.subheader("Прогноз по файлу")
    st.caption("CSV после преобразования единиц измерения в числа, до новых признаков и OHE — как df_test в ноутбуке.")
    uploaded = st.file_uploader("CSV с автомобилями", type="csv")
    if uploaded is None:
        st.info("Загрузите CSV с теми же признаками, что использовались в ноутбуке.")
    else:
        batch = pd.read_csv(uploaded)
        missing = [column for column in features if column not in batch.columns]
        if missing:
            st.error(f"В файле нет колонок: {', '.join(missing)}")
        else:
            batch["predicted_price"] = model.predict(prepare_features(batch))
            st.dataframe(batch, hide_index=True)
            st.download_button(
                "Скачать прогнозы",
                data=batch.to_csv(index=False).encode("utf-8-sig"),
                file_name="predictions.csv",
                mime="text/csv"
            )

with eda_tab:
    st.subheader("Распределения признаков")
    column = st.selectbox("Признак", ["selling_price", "km_driven", "year", "max_power", "engine"])
    log_scale = st.checkbox("Показать log(1 + x)")
    values = data[column].dropna()
    if log_scale:
        values = np.log1p(values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=35, edgecolor="white")
    ax.set_xlabel(f"log(1 + {column})" if log_scale else column)
    ax.set_ylabel("Количество автомобилей")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Гистограмма показывает типичные значения и асимметрию распределения.")

    st.subheader("Связь признаков с ценой")
    x = st.selectbox("Признак по оси X", ["year", "max_power", "km_driven", "engine"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(data[x], data["selling_price"], alpha=0.35, s=12)
    ax.set_xlabel(x)
    ax.set_ylabel("selling_price")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Категориальные признаки")
    category = st.selectbox("Категория", ["fuel", "transmission", "owner", "seller_type"])
    st.bar_chart(data[category].value_counts())

with weights_tab:
    st.subheader("Коэффициенты Ridge")
    coefficients = pd.Series(model.coef_, index=artifact["feature_columns"], name="coefficient")
    count = st.slider("Сколько коэффициентов показать", 1, len(coefficients), min(20, len(coefficients)))
    selected = coefficients.loc[coefficients.abs().nlargest(count).index].sort_values()

    fig, ax = plt.subplots(figsize=(9, max(4, count * 0.3)))
    ax.barh(selected.index, selected.values)
    ax.axvline(0, color="gray")
    ax.set_xlabel("Коэффициент")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write(f"Свободный коэффициент: {model.intercept_:,.2f}")
    st.caption("Признаки не стандартизованы, поэтому модули коэффициентов нельзя напрямую сравнивать как важность признаков.")
    st.dataframe(coefficients.to_frame())
