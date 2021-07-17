import pandas as pd
import numpy as np
import warnings
import streamlit as st
import plotly.express as px
from matplotlib.pyplot import hist
import seaborn as sns

st.title("Telco Analysis")

choice = st.sidebar.selectbox("Select Analysis", (
    "User_Overview_analysis", "User_Engagement_analysis", "Experience_Analytics", "Satisfaction_Analysis"))


@st.cache
def loadData():
    warnings.filterwarnings('ignore')
    pd.set_option('max_column', None)
    Telco = pd.read_csv('../data/Week1_challenge_data_source.csv', na_values=['?', None])
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
    return Telco


@st.cache
def aggPerUser(df):
    Agg_data_goupedby_user_df = df.groupby('MSISDN/Number').agg(
        {'Dur. (ms)': 'sum', 'Total UL (Bytes)': 'sum', 'Total DL (Bytes)': 'sum',
         'Total Gaming data volume (in Bytes)': 'sum', 'Total Social Media data volume (in Bytes)': 'sum',
         'Total Google data volume (in Bytes)': 'sum', 'Total Email data volume (in Bytes)': 'sum',
         'Total Youtube data volume (in Bytes)': 'sum', 'Total Netflix data volume (in Bytes)': 'sum',
         'Total Other data volume (in Bytes)': 'sum'})
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
    p = sns.distplot(df[column])
    return p



@st.cache
def checkSkew(df):
    skewValue = df.skew(axis=1)
    return skewValue


def main():
    if choice == "User_Overview_analysis":
        st.subheader("User Overview analysis")
        sel = st.selectbox("Select choice", (
            "Data variables", "Sample data", "XDR sessions per user",
            'Non - Graphical Univariate Analysis', 'Graphical Univariate Analysis'))
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

            elif cb == "box plot":
                df = combineColum()
                agg_per_user_df = aggPerUser(df)

                rd = st.radio("Select feature", ('Gaming', 'YouTube', 'Email', 'Google', 'other'))
                if rd == "Gaming":
                    p=boxPlot(agg_per_user_df, "Total Youtube data volume (in Bytes)")
                    st.plotly_chart(p)



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















    elif choice == "User_Engagement_analysis":
        st.subheader("User Engagement analysis")


    elif choice == "Experience_Analytics":

        st.subheader("Experience Analytics")
    elif choice == "Satisfaction_Analysis":
        st.subheader("Satisfaction  Analytics")


if __name__ == '__main__':
    main()
