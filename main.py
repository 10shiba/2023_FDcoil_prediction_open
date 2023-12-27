import streamlit as st
import pandas as pd
import joblib
import numpy as np

def load_model(model_path):
    """モデルを読み込む関数"""
    return joblib.load(model_path)

def predict(input_data, scaler, models):
    """予測を行う関数"""
    # 入力データの標準化
    scaled_data = scaler.transform(np.array([input_data]))

    # モデルによる予測
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(scaled_data)[0]
    return predictions

def main():
    # タイトル
    st.title("1stコイルサイズ予測デモ")

    # モデルとscalerの読み込み
    model_dir = './models'
    scaler = load_model(f'{model_dir}/scaler.pkl')
    models = {
        'RF': load_model(f'{model_dir}/rf.pkl'),
        'LGBM': load_model(f'{model_dir}/lgbm.pkl')
    }

    # ユーザー入力
    input_value = st.number_input("入力値 (Distal diameter)", value=0.0)

    # 予測実行ボタン
    if st.button('予測実行'):
        # 予測
        predictions = predict(input_value, scaler, models)

        # 予測結果の表示
        st.write(f"RFモデルによる予測: {predictions['RF']}")
        st.write(f"LGBMモデルによる予測: {predictions['LGBM']}")

if __name__ == '__main__':
    main()
