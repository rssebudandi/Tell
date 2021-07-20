import matplotlib.pyplot as plt
import pandas as pd
import warnings
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.title("Telco Analysis")

choice = st.sidebar.selectbox("Select Analysis", (
    "User_Overview_analysis", "User_Engagement_analysis", "Experience_Analytics", "Satisfaction_Analysis"))


@st.cache
def loadData():
    warnings.filterwarnings('ignore')
    pd.set_option('max_column', None)
    Telco = pd.read_csv('Week1_challenge_data_source.csv', na_values=['?', None])
    return Telco


@st.cache
def dataVariables():
    df = loadData()

    data_type = df.dtypes
    return data_type


@st.cache
def sampleData():
    df = loadData()
    sample = df.head()
    return sample


@st.cache
def combineColum():
    # combining columns for total data voulume in bytes
    Telco = loadData()

    Telco['Total Gaming data volume (in Bytes)'] = Telco.apply(
        lambda x: x['Gaming DL (Bytes)'] + x['Gaming UL (Bytes)'], axis=1)
    Telco['Total Social Media data volume (in Bytes)'] = Telco.apply(
        lambda x: x['Social Media UL (Bytes)'] + x['Social Media DL (Bytes)'], axis=1)
    Telco['Total Google data volume (in Bytes)'] = Telco.apply(
        lambda x: x['Google DL (Bytes)'] + x['Google UL (Bytes)'], axis=1)
    Telco['Total Email data volume (in Bytes)'] = Telco.apply(lambda x: x['Email DL (Bytes)'] + x['Email UL (Bytes)'],
                                                              axis=1)
    Telco['Total Youtube data volume (in Bytes)'] = Telco.apply(
        lambda x: x['Youtube DL (Bytes)'] + x['Youtube UL (Bytes)'], axis=1)
    Telco['Total Netflix data volume (in Bytes)'] = Telco.apply(
        lambda x: x['Netflix DL (Bytes)'] + x['Netflix UL (Bytes)'], axis=1)
    Telco['Total Other data volume (in Bytes)'] = Telco.apply(lambda x: x['Other DL (Bytes)'] + x['Other UL (Bytes)'],
                                                              axis=1)
    Telco['Total data volume (in Bytes)'] = Telco.apply(lambda x: x['Total UL (Bytes)'] + x['Total DL (Bytes)'], axis=1)
    return Telco


@st.cache
def aggPerUser(df):
    Agg_data_goupedby_user_df = df.groupby('MSISDN/Number').agg(
        {'Dur. (ms)': 'sum', 'Total UL (Bytes)': 'sum', 'Total DL (Bytes)': 'sum',
         'Total Gaming data volume (in Bytes)': 'sum', 'Total Social Media data volume (in Bytes)': 'sum',
         'Total Google data volume (in Bytes)': 'sum', 'Total Email data volume (in Bytes)': 'sum',
         'Total Youtube data volume (in Bytes)': 'sum', 'Total Netflix data volume (in Bytes)': 'sum',
         'Total Other data volume (in Bytes)': 'sum','Total data volume (in Bytes)':'sum'})
    return Agg_data_goupedby_user_df


@st.cache
def basicMetricAnalysis(df):
    Sumarry_description = df.describe()
    return Sumarry_description


@st.cache
def histogram(df, column):
    # fig = px.histogram(df,x=df[df[column]], title='Rating distribution')
    fig = px.histogram(df, x=df[column])
    return fig


@st.cache
def boxPlot(df, column):
    bx = df.boxplot(column=column, return_type='axes');
    return bx


@st.cache
def checkSkew(df):
    skewValue = df.skew(axis=1)
    return skewValue

@st.cache
def scatterPlot(df, colum1, column2):
    fig = px.scatter(
        x=df[colum1],
        y=df[column2],
    )
    fig.update_layout(
        xaxis_title=colum1,
        yaxis_title=column2,
    )
    return fig
def scatterPlot(colum1, column2):
    fig = px.scatter(
        x=colum1,
        y=column2,
    )
    fig.update_layout(
        xaxis_title=colum1,
        yaxis_title=column2,
    )
    return fig

def calcCorr(df):
    corrM = df.corr()
    return corrM
def scaleData(df):


    X_std = StandardScaler().fit_transform(df)
    return X_std
def apply_PCA(df):
    sd=scaleData(df)
    sklearn_pca = PCA(n_components=2)  # intialise the PCA algorithm
    Y_sklearn = sklearn_pca.fit_transform(sd)

    return Y_sklearn






def main():
    if choice == "User_Overview_analysis":
        st.subheader("User Overview analysis")
        sel = st.selectbox("Select choice", (
            "Data variables", "Sample data", "XDR sessions per user",
            'Non - Graphical Univariate Analysis', 'Graphical Univariate Analysis', 'Bivariate Analysis','Correlation Analysis','Dimensionality Reduction'))
        if sel == "Data variables":
            st.subheader("Data variables")
            variables = dataVariables()
            st.write(variables)
        elif sel == "Sample data":
            st.subheader("Sample data")
            samp = sampleData()
            st.write(samp)
        elif sel == "XDR sessions per user":
            df = combineColum()
            st.subheader(
                "Aggregating number of xDR sessions, session duration , total download(DL) and upload (UL) data and "
                "total data voulume during sesssion by user")
            agg_per_user_df = aggPerUser(df)
            st.write(agg_per_user_df)

        elif sel == "Non - Graphical Univariate Analysis":
            st.subheader("Basic metrics analysis")
            ch = st.selectbox("Select choice", ("Basic Metric", "Skewness"))
            if ch == "Basic Metric":
                st.subheader("Basic metrics analysis")
                df = combineColum()
                agg_per_user_df = aggPerUser(df)

                st.write(basicMetricAnalysis(agg_per_user_df))
                st.write("""
                          - The total google data volume has a high standard deviation meaning its values are more spread from the mean.
                          - 25% of total YouTube data volume in bytes by the user is smaller than 18,631,090 bytes and 75% of the total YouTube data is smaller than 37,927,980 bytes and half of the YouTube data volume in bytes  lies below 26,800,380 bytes Half of the data volume for gaming is above 54,234,920 bytes.
                          - The users use on average 599,769,000 bytes volume of data, this means that users like gaming.
                          """)


            elif ch == "Skewness":
                st.subheader("Check for Skewness")
                df = combineColum()
                agg_per_user_df = aggPerUser(df)
                bp = checkSkew(agg_per_user_df)
                st.write(bp)
                st.write("""
                    - As it can be seen above our data shows a positive skew meaning  the tail is larger towards the right hand side of the distribution
                    """)
        elif sel == "Graphical Univariate Analysis":
            st.subheader("Graphical Univariate Analysis")
            cb = st.selectbox("chose graph", ('histogram', 'box plot'))
            if cb == 'histogram':
                df = combineColum()
                agg_per_user_df = aggPerUser(df)
                rd = st.radio("Select feature", ('Gaming', 'YouTube', 'Email', 'Google', 'other'))
                if rd == "Gaming":
                    hist = histogram(agg_per_user_df, "Total Gaming data volume (in Bytes)")
                    st.plotly_chart(hist)
                elif rd == "YouTube":
                    hist = histogram(agg_per_user_df, "Total Youtube data volume (in Bytes)")
                    st.plotly_chart(hist)
                elif rd == "Email":
                    hist = histogram(agg_per_user_df, "Total Email data volume (in Bytes)")
                    st.plotly_chart(hist)
                elif rd == "Google":
                    hist = histogram(agg_per_user_df, "Total Google data volume (in Bytes)")
                    st.plotly_chart(hist)
                elif rd == "other":
                    hist = histogram(agg_per_user_df, "Total Other data volume (in Bytes)")
                    st.plotly_chart(hist)

        elif sel == "Bivariate Analysis":
            st.subheader("Bivariate Analysis")
            rdb = st.radio("Select feature", ("Total Volume vs Gaming volume", "Total volume vs YouTube volume", "Total volume vs Email Volume"))
            if rdb == "Total Volume vs Gaming volume":
                st.subheader("Total Volume vs Gaming volume")
                df = combineColum()
                agg_per_user_df = aggPerUser(df)
                sc = scatterPlot(agg_per_user_df, 'Total Gaming data volume (in Bytes)', 'Total data volume (in Bytes)')
                st.write(sc)
            elif rdb == "Total volume vs YouTube volume":
                st.subheader("Total volume vs YouTube volume")
                df = combineColum()
                agg_per_user_df = aggPerUser(df)
                sc = scatterPlot(agg_per_user_df, 'Total Youtube data volume (in Bytes)', 'Total data volume (in Bytes)')
                st.write(sc)
            elif rdb == "Total volume vs Email Volume":
                st.subheader("Total volume vs Email volume")
                df = combineColum()
                agg_per_user_df = aggPerUser(df)
                sc = scatterPlot(agg_per_user_df, 'Total Email data volume (in Bytes)',
                                 'Total data volume (in Bytes)')
                st.write(sc)
        elif sel == "Correlation Analysis":
            st.subheader("Correlation Analysis")
            df = combineColum()
            agg_per_user_df = aggPerUser(df)
            corr= calcCorr(agg_per_user_df)
            st.write(corr)
            st.write("""
            - The correration coefficients along the diagonal of the table are all equal to 1 because each variable is perfectly correlated with it's self.
            - It can also be noticed that the correlation matrix is perfectly symmetrical because the values in the top left corner is equal to the value in the bottom right corner
            - It can also be seen that the highest correration is between data used on google and youtube , which is understanble since people usually go to google to search youtube.
            - Data spent on gaming has the lowest correrations with each variable, which can be due to the fact that when gaming you wouldn't have time to user other forms""")
        elif sel == "Dimensionality Reduction":
            st.subheader("Dimensionality Reduction")
            df = combineColum()
            agg_per_user_df = aggPerUser(df)
            pc=apply_PCA(agg_per_user_df)
            st.write(pc)
            b=scatterPlot()























    elif choice == "User_Engagement_analysis":
        st.subheader("User Engagement analysis")


    elif choice == "Experience_Analytics":

        st.subheader("Experience Analytics")
    elif choice == "Satisfaction_Analysis":
        st.subheader("Satisfaction  Analytics")


if __name__ == '__main__':
    main()
