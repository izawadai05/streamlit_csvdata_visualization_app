import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# アプリのタイトル
st.title("Data Laboratory")

# サイドバーで機能選択
st.sidebar.title("Option")

# データのアップロード
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])


option = st.sidebar.selectbox("Please select a function", 
    ["Upload data", "Statistical Information", "Data cleaning", "Data visualization", "Correlation analysis", "Machine Learning Model"])



if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if option == "Upload data":
        st.write("Uploaded data:")
        st.dataframe(df)

    if option == "Statistical Information":
        st.write("Statistical Information:")
        st.write(df.describe())
        st.write("Datatype information:")
        st.write(df.dtypes)


    if option == "Data cleaning":
        st.write("Data before completion:")
        st.dataframe(df)
        st.write("Missing-Value information:")
        st.write(df.isnull().sum())
        st.write("Missing value heat map:")
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        st.pyplot(fig)
        # 欠損値補完のオプション
        impute_option = st.selectbox("Please select how you would like to complete missing values", ["mean", "median", "mode", "complement with 0", "delete"])
        if st.button("Complete missing values"):
            if impute_option == "mean":
                imputer = SimpleImputer(strategy='mean')
                df.iloc[:, :] = imputer.fit_transform(df)
            elif impute_option == "median":
                imputer = SimpleImputer(strategy='median')
                df.iloc[:, :] = imputer.fit_transform(df)
            elif impute_option == "mode":
                imputer = SimpleImputer(strategy='most_frequent')
                df.iloc[:, :] = imputer.fit_transform(df)
            elif impute_option == "complement with 0":
                df.fillna(0, inplace=True)
                st.write("Missing values were complemented with 0")
            elif impute_option == "delete":
                df.dropna(inplace=True)
                st.write("Deleted rows containing missing values")
            
            st.write(f"Missing values were completed with {impute_option}")
            st.write("Data after completion:")
            st.dataframe(df)
            st.write(df.isnull().sum())
            
            #csv = df.to_csv(index=False).encode('utf-8')
            #st.download_button(label="Download as CSV", data=csv, file_name='cleaned_data.csv', mime='text/csv')
            st.write("Missing value heat map after completion:")
            fig, ax = plt.subplots()
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
            st.pyplot(fig)



    if option == "Data visualization":
        st.write("Please select a column for the graph:")
        col1 = st.selectbox("X-axis columns", df.columns)
        col2 = st.selectbox("Y-axis columns", df.columns)
        graph_type = st.selectbox("Graph Type", ["Scatter Plot", "Histogram", "Box and Whisker", "Bar Chart", "Line Chart", "Pair Plot"])
        
        if graph_type == "Scatter Plot":
            fig = px.scatter(df, x=col1, y=col2)
            st.plotly_chart(fig)
        elif graph_type == "Histogram":
            fig = px.histogram(df, x=col1)
            st.plotly_chart(fig)
        elif graph_type == "Box and Whisker":
            fig = px.box(df, x=col1, y=col2)
            st.plotly_chart(fig)
        elif graph_type == "Bar Chart":
            fig = px.bar(df, x=col1, y=col2)
            st.plotly_chart(fig)
        elif graph_type == "Line Chart":
            fig = px.line(df, x=col1, y=col2)
            st.plotly_chart(fig)
        elif graph_type == "Pair Plot":
            pairplot_columns = st.multiselect("Please select the columns you would like to include in the paired plot", df.columns)
            if len(pairplot_columns) > 1:
                fig = sns.pairplot(df[pairplot_columns])
                st.pyplot(fig)

    if option == "Correlation analysis":
        st.write("correlation matrix:")
        # 数値データのみを抽出
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        st.write(corr)
        st.write("Correlation Heat Map:")
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot(fig)


    if option == "Machine Learning Model":
        st.write("Simple linear regression model")
        feature_col = st.selectbox("Feature Columns", df.columns)
        target_col = st.selectbox("Target columns", df.columns)
        
        # 欠損値の削除
        ml_df = df[[feature_col, target_col]].dropna()
        
        X = ml_df[[feature_col]].values
        y = ml_df[target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        st.write("Regression Model Coefficients:")
        st.write(f"coefficient: {model.coef_[0]}")
        st.write(f"intercept: {model.intercept_}")
        st.write(f"coefficient of determination: {model.score(X_test, y_test)}")

        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test, color='blue', label='Actual')
        ax.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
        ax.legend()
        st.pyplot(fig)



else:
    st.write("Upload CSV file")

