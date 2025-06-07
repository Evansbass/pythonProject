# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_and_analyze_data(file_path):
    """
    加载数据集并生成基本分析报告
    """
    print("步骤1: 加载数据集并生成基本分析报告")
    df = pd.read_csv(file_path)

    # 生成基本统计报告
    print("\n数据集基本信息:")
    print(f"- 样本量: {df.shape[0]}条")
    print(f"- 特征数: {df.shape[1]}个")

    print("\n字段类型分布:")
    print(df.dtypes.value_counts())

    print("\n缺失值分析:")
    missing_analysis = df.isnull().sum().reset_index()
    missing_analysis.columns = ['字段', '缺失数量']
    missing_analysis['缺失比例'] = round(missing_analysis['缺失数量'] / len(df) * 100, 2)
    print(missing_analysis[missing_analysis['缺失数量'] > 0])

    # 目标变量分布
    print("\n目标变量分布 (loan_status):")
    print(df['loan_status'].value_counts(normalize=True) * 100)

    # 保存原始数据分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['person_income'], bins=50, kde=True)
    plt.title('原始收入分布')
    plt.savefig('原始收入分布.png')

    return df


def handle_missing_values(df):
    """
    处理缺失值
    """
    print("\n步骤2: 处理缺失值")

    # 1. 工龄中位数填充
    print("- 处理person_emp_length缺失值 (中位数填充)")
    df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)

    # 2. 利率分组填充
    print("- 处理loan_int_rate缺失值 (按贷款等级分组平均填充)")
    df['loan_int_rate'] = df.groupby('loan_grade')['loan_int_rate'].transform(
        lambda x: x.fillna(x.mean()))

    # 3. 收入随机森林填充
    print("- 处理person_income缺失值 (随机森林预测填充)")
    income_missing = df[df['person_income'].isnull()]
    income_not_missing = df.dropna(subset=['person_income'])

    if not income_missing.empty:
        features = ['person_age', 'person_emp_length', 'loan_amnt', 'cb_person_cred_hist_length']
        X_train = income_not_missing[features]
        y_train = income_not_missing['person_income']

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        pred_income = rf.predict(income_missing[features])
        df.loc[df['person_income'].isnull(), 'person_income'] = pred_income

    # 验证缺失值处理结果
    print("\n缺失值处理后验证:")
    print(f"剩余缺失值数量: {df.isnull().sum().sum()}")

    return df


def handle_outliers(df):
    """
    处理异常值
    """
    print("\n步骤3: 处理异常值")

    # 1. 年龄异常处理
    print(f"- 删除年龄>100的样本 (原始: {df.shape[0]}条)")
    df = df[df['person_age'] <= 100]
    print(f"  处理后: {df.shape[0]}条")

    # 2. IQR法处理数值型字段
    print("- 处理数值型字段异常值 (IQR方法)")
    numeric_cols = ['person_income', 'loan_amnt']
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = ~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)
    df = df[mask]
    print(f"  处理后样本量: {df.shape[0]}条")

    # 保存处理后收入分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['person_income'], bins=50, kde=True)
    plt.title('处理后收入分布')
    plt.savefig('处理后收入分布.png')

    return df


def feature_engineering(df):
    """
    特征工程
    """
    print("\n步骤4: 特征工程")

    # 1. 二元编码
    print("- cb_person_default_on_file: Y→1, N→0")
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

    # 2. One-Hot编码
    print("- 对分类变量进行One-Hot编码")
    categorical_cols = ['person_home_ownership', 'loan_intent']
    df = pd.get_dummies(df, columns=categorical_cols)

    # 3. 创建新特征
    print("- 创建新特征: 债务收入比")
    df['debt_to_income'] = df['loan_amnt'] / df['person_income']

    # 4. 年龄分箱
    print("- 年龄分箱: 青年(<30), 中年(30-50), 老年(≥50)")
    bins = [0, 30, 50, 100]
    labels = ['青年', '中年', '老年']
    df['age_group'] = pd.cut(df['person_age'], bins=bins, labels=labels)

    # 5. 贷款金额分级
    print("- 贷款金额分级: 低(<10k), 中(10k-20k), 高(≥20k)")
    df['loan_amnt_group'] = pd.cut(df['loan_amnt'],
                                   bins=[0, 10000, 20000, float('inf')],
                                   labels=['低', '中', '高'])

    return df


def generate_report(df):
    """
    生成预处理报告
    """
    print("\n步骤5: 生成预处理报告")

    # 数据集基本信息
    report = {
        "处理日期": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "原始样本量": 250000,
        "处理后样本量": df.shape[0],
        "删除样本比例": f"{round((250000 - df.shape[0]) / 250000 * 100, 2)}%",
        "剩余特征数": df.shape[1],
        "目标变量分布": dict(df['loan_status'].value_counts())
    }

    # 保存处理后的数据集
    df.to_csv('credit_risk_cleaned.csv', index=False)
    print("清洗后的数据集已保存为: credit_risk_cleaned.csv")

    # 生成报告
    report_df = pd.DataFrame(list(report.items()), columns=['指标', '值'])
    report_df.to_csv('预处理报告.csv', index=False)
    print("预处理报告已保存为: 预处理报告.csv")

    # 生成特征相关性热力图
    plt.figure(figsize=(12, 8))
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('特征相关性热力图')
    plt.savefig('特征相关性热力图.png')

    return report_df


# 主函数
if __name__ == "__main__":
    # 1. 加载数据并分析
    df = load_and_analyze_data('credit_risk_dataset.csv')

    # 2. 处理缺失值
    df = handle_missing_values(df)

    # 3. 处理异常值
    df = handle_outliers(df)

    # 4. 特征工程
    df = feature_engineering(df)

    # 5. 生成报告和保存结果
    report = generate_report(df)

    print("\n预处理流程已完成!")