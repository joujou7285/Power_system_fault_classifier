import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import plotly.express as px

# 在側邊欄創建頁面選項
page = st.sidebar.selectbox("選擇頁面", ["模擬實際數據", "故障分析", "互動式故障預測"])

# 第一個頁面：模擬實際數據
if page == "模擬實際數據":
    st.title("模擬實際數據(逐時紀錄一星期)")

    # Load the CSV data
    file_path = 'classData.csv'  # 使用相对路径
    df = pd.read_csv(file_path)

    # 設定模擬參數
    time_series = pd.date_range(start='2024-10-01', periods=168, freq='h')  # 一星期逐時紀錄

    # 計算每個列的均值和標準差
    mean_Va, std_Va = df['Va'].mean(), df['Va'].std()
    mean_Vb, std_Vb = df['Vb'].mean(), df['Vb'].std()
    mean_Vc, std_Vc = df['Vc'].mean(), df['Vc'].std()

    mean_Ia, std_Ia = df['Ia'].mean(), df['Ia'].std()
    mean_Ib, std_Ib = df['Ib'].mean(), df['Ib'].std()
    mean_Ic, std_Ic = df['Ic'].mean(), df['Ic'].std()

    # 模擬電壓和電流數據
    voltage_A = mean_Va + np.random.normal(0, std_Va, 168)
    current_A = mean_Ia + np.random.normal(0, std_Ia, 168)

    voltage_B = mean_Vb + np.random.normal(0, std_Vb, 168)
    current_B = mean_Ib + np.random.normal(0, std_Ib, 168)

    voltage_C = mean_Vc + np.random.normal(0, std_Vc, 168)
    current_C = mean_Ic + np.random.normal(0, std_Ic, 168)

    # 建立數據表格
    data = {
        'Time': time_series,
        'Voltage_A': voltage_A,
        'Current_A': current_A,
        'Voltage_B': voltage_B,
        'Current_B': current_B,
        'Voltage_C': voltage_C,
        'Current_C': current_C
    }

    df_simulated = pd.DataFrame(data)

    # 顯示模擬數據
    st.write("模擬數據：")
    st.dataframe(df_simulated)

    # 電壓時間序列圖
    fig_voltage, ax_voltage = plt.subplots(figsize=(10, 5))
    ax_voltage.plot(df_simulated['Time'], df_simulated['Voltage_A'], label='Voltage A', color='blue')
    ax_voltage.plot(df_simulated['Time'], df_simulated['Voltage_B'], label='Voltage B', color='orange')
    ax_voltage.plot(df_simulated['Time'], df_simulated['Voltage_C'], label='Voltage C', color='green')
    ax_voltage.set_title('Voltage Time Series Plot')
    ax_voltage.set_xlabel('Time')
    ax_voltage.set_ylabel('Voltage (V)')
    ax_voltage.legend()
    st.pyplot(fig_voltage)

    # 電流時間序列圖
    fig_current, ax_current = plt.subplots(figsize=(10, 5))
    ax_current.plot(df_simulated['Time'], df_simulated['Current_A'], label='Current A', color='red')
    ax_current.plot(df_simulated['Time'], df_simulated['Current_B'], label='Current B', color='purple')
    ax_current.plot(df_simulated['Time'], df_simulated['Current_C'], label='Current C', color='pink')
    ax_current.set_title('Current Time Series Plot')
    ax_current.set_xlabel('Time')
    ax_current.set_ylabel('Current (A)')
    ax_current.legend()
    st.pyplot(fig_current)

# 第二個頁面：故障分析監控系統
elif page == "故障分析":
    st.title("電力系統故障分析")

    # Load the CSV data
    file_path = 'classData.csv'  # 使用相对路径
    df = pd.read_csv(file_path)

    # 設定模擬參數
    time_series = pd.date_range(start='2024-10-01', periods=168, freq='h')  # 一星期逐時紀錄

    # 計算每個列的均值和標準差
    mean_Va, std_Va = df['Va'].mean(), df['Va'].std()
    mean_Vb, std_Vb = df['Vb'].mean(), df['Vb'].std()
    mean_Vc, std_Vc = df['Vc'].mean(), df['Vc'].std()

    mean_Ia, std_Ia = df['Ia'].mean(), df['Ia'].std()
    mean_Ib, std_Ib = df['Ib'].mean(), df['Ib'].std()
    mean_Ic, std_Ic = df['Ic'].mean(), df['Ic'].std()

    # 模擬電壓和電流數據
    voltage_A = mean_Va + np.random.normal(0, std_Va, 168)
    current_A = mean_Ia + np.random.normal(0, std_Ia, 168)

    voltage_B = mean_Vb + np.random.normal(0, std_Vb, 168)
    current_B = mean_Ib + np.random.normal(0, std_Ib, 168)

    voltage_C = mean_Vc + np.random.normal(0, std_Vc, 168)
    current_C = mean_Ic + np.random.normal(0, std_Ic, 168)

    # 建立數據表格
    data = {
        'Time': time_series,
        'Voltage_A': voltage_A,
        'Current_A': current_A,
        'Voltage_B': voltage_B,
        'Current_B': current_B,
        'Voltage_C': voltage_C,
        'Current_C': current_C
    }

    df_simulated = pd.DataFrame(data)
    # 使用上述模擬數據
    df = df_simulated

    model_file_path = 'saved_random_forest_model.pkl'  # 使用相对路径
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)

    # 故障類型對應字典
    reverse_mapping = {
        0: 'No Fault',
        1: 'LG Fault',
        2: 'LL Fault',
        3: 'LLG Fault',
        4: 'LLL Fault',
        5: 'LLLG Fault'
    }

    # 數據標準化
    mean = df[['Voltage_A', 'Current_A', 'Voltage_B', 'Current_B', 'Voltage_C', 'Current_C']].mean()
    std = df[['Voltage_A', 'Current_A', 'Voltage_B', 'Current_B', 'Voltage_C', 'Current_C']].std()
    standardized_data = (df[['Voltage_A', 'Current_A', 'Voltage_B', 'Current_B', 'Voltage_C', 'Current_C']] - mean) / std
    standardized_data = standardized_data.fillna(0)

    # 預測故障類型
    y_pred = model.predict(standardized_data)
    df['Fault_Type'] = [reverse_mapping[label] for label in y_pred]

    # 顯示當前的故障分類結果
    st.write("實時故障分類結果：")
    st.dataframe(df)

    # 故障分類結果可視化：柱狀圖顯示故障分佈
    st.write("故障類型分佈：")
    fault_counts = df['Fault_Type'].value_counts()
    st.bar_chart(fault_counts)

    # 趨勢分析：隨時間監控不同故障類型的變化
    fig_fault_trend = px.line(df, x='Time', y='Fault_Type', title='故障類型隨時間變化趨勢')
    st.plotly_chart(fig_fault_trend)

    # 歷史數據回顧：事件時間線
    st.write("故障事件時間線：")
    df['Time'] = pd.date_range(start='2024-10-01', periods=len(df), freq='H')
    fig_timeline = px.scatter(df, x='Time', y='Fault_Type', title='故障事件時間線', color='Fault_Type')
    st.plotly_chart(fig_timeline)

    # 自動生成報告
    st.write("歷史數據摘要報告：")
    report = df.groupby('Fault_Type').size().reset_index(name='Counts')
    st.dataframe(report)
# 第三個頁面：互動式故障預測
elif page == "互動式故障預測":
    st.title("互動式故障預測")

    # 用戶輸入三個電流和三個電壓
    st.subheader("請輸入電流和電壓數據：")
    Ia = st.number_input("輸入 Ia (A)", min_value=-1000.0, max_value=1000.0, step=0.1)
    Ib = st.number_input("輸入 Ib (A)", min_value=-1000.0, max_value=1000.0, step=0.1)
    Ic = st.number_input("輸入 Ic (A)", min_value=-1000.0, max_value=1000.0, step=0.1)
    Va = st.number_input("輸入 Va (V)", min_value=-1.0, max_value=1.0, step=0.001)
    Vb = st.number_input("輸入 Vb (V)", min_value=-1.0, max_value=1.0, step=0.001)
    Vc = st.number_input("輸入 Vc (V)", min_value=-1.0, max_value=1.0, step=0.001)

 # 加載隨機森林模型
    model_file_path = 'saved_random_forest_model.pkl'  # 使用相对路径
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)

    # 故障類型對應字典
    reverse_mapping = {
        0: 'No Fault',
        1: 'LG Fault',
        2: 'LL Fault',
        3: 'LLG Fault',
        4: 'LLL Fault',
        5: 'LLLG Fault'
    }

    # 進行預測
    if st.button("預測故障類型"):
        input_data = pd.DataFrame([[Ia, Ib, Ic, Va, Vb, Vc]], columns=['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'])

        # 數據標準化
        mean = input_data.mean()
        std = input_data.std()
        standardized_data = (input_data - mean) / std

        y_pred = model.predict(standardized_data)
        fault_type = reverse_mapping[y_pred[0]]

        # 顯示預測結果
        st.write(f"預測的故障類型是：{fault_type}")

