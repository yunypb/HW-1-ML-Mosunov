import joblib
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from io import StringIO
import re
import sklearn
app = FastAPI()

def preprocessor_df(df_test):

    df_test.drop(columns='selling_price', inplace=True)

    def to_num(df, column_name, units):
        for i in units:
            df[column_name] = df[column_name].str.replace(f' {i}', '')

        df[column_name] = pd.to_numeric(df[column_name])
        return df

    df_test = to_num(df_test, 'max_power', ['bhp'])
    df_test = to_num(df_test, 'engine', ['CC'])
    df_test = to_num(df_test, 'mileage', ['kmpl', 'km/kg'])

    def process_torque(df):
        torque_vals = []
        rpm_vals = []

        for i in df['torque']:
            if pd.isna(i):
                torque_vals.append(i)
                rpm_vals.append(i)
                continue

            elif not isinstance(i, str):
                print(i)
                torque_vals.append(None)
                rpm_vals.append(None)
                continue

            i = i.replace(',', '')

            torque_match = re.search(r"(\d+(\.\d+)?)\s*(Nm|kgm|)", i, re.IGNORECASE)

            if torque_match:
                torque_value = float(torque_match.group(1))
                unit = torque_match.group(3).lower()

                if unit == 'kgm':
                    torque_value *= 9.80665
                    # 1 kgm ≈ 9.80665 Nm
            else:
                torque_value = None

            rpm_match = re.search(r"@ (\d{1,4})(?:-(\d{1,4}))?\s?(rpm\s?|\(kgm@ rpm\)\s?)", i)
            fallback_match = re.search(r"(\d{1,4})\s?(rpm|RPM)?\s*$", i)

            if fallback_match:
                rpm = fallback_match.group(1)

            elif rpm_match:
                if rpm_match.group(2):
                    rpm = rpm_match.group(2)
                else:
                    rpm = rpm_match.group(1)

            else:
                rpm = None

            torque_vals.append(torque_value)
            rpm_vals.append(rpm)

        df['torque'] = torque_vals
        df['max_torque_rpm'] = rpm_vals
        return df

    df_test = process_torque(df_test.copy())
    df_test['torque'] = pd.to_numeric(df_test['torque'])
    df_test['max_torque_rpm'] = pd.to_numeric(df_test['max_torque_rpm'])

    medians = df_test[['mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']].median()
    df_test.fillna(medians, inplace=True)

    df_test['engine'] = df_test['engine'].astype('int')
    df_test['seats'] = df_test['seats'].astype('int')

    df_test['brand'] = df_test['name'].apply(lambda x: x.split()[0])
    df_test['brand_count_encoded'] = df_test['name'].map(df_test['name'].value_counts())

    df_test = df_test.drop(columns=['name'])

    return df_test

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

pipeline = joblib.load('model_pipeline.pkl')

@app.post("/predict_item")
async def predict_item(item: Item) -> dict:
    try:
        df = pd.DataFrame([item.dict()])
        df_test = preprocessor_df(df)
        prediction = pipeline.predict(df_test)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df_test = preprocessor_df(df)
        df_test['predictions'] = pipeline.predict(df_test)

        output = StringIO()
        df_test.to_csv(output, index=False)
        output.seek(0)

        return {"filename": "predictions.csv", "file": output.getvalue()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))