import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np


def load_model(model_path):
    """モデルを読み込む関数"""
    st.write(f"モデルを読み込み中: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)


def predict(input_data, scaler, models):
    try:
        # 入力データが1次元の場合、2次元に変換
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        # 入力データの標準化
        scaled_data = scaler.transform(input_data)

        # モデルによる予測
        predictions = {}
        for model_name, model in models.items():
            if model is not None:  # モデルがNoneでないことを確認
                predictions[model_name] = model.predict(scaled_data)[0]
            else:
                predictions[model_name] = None
        return predictions
    except Exception as e:
        st.error(f"予測中にエラーが発生しました: {e}")
        return None


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
    input_value = st.number_input("入力値 (Distal diameter)", value=5.0)
    ## numpy配列に変換
    input_value = np.array([[input_value]])

    # 予測実行ボタン
    if st.button('予測実行'):
        # 予測
        predictions = predict(input_value, scaler, models)

        st.markdown(f"## AIによるFDコイル予測結果")
        # 予測結果の表示
        st.write(f"RFモデルによる予測: {predictions['RF']:.03}")
        st.write(f"LGBMモデルによる予測: {predictions['LGBM']:.03}")

if __name__ == '__main__':
    main()
