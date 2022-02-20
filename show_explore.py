import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


@st.cache
def load_data():
    df = pd.read_csv("C:/location_analytics/projet/NYPD_Complaint_Data_Historic_2019_clean.csv")
    df = df[["OFNS_DESC", "BORO_NM", "ADDR_PCT_CD", "VIC_SEX", "LAW_CAT_CD" , "PARKS_NM" , "VIC_AGE_GROUP"]]
    df = df.dropna()
    return df
df = load_data()

def show_explore():
    st.title("Explore NewYork city crimes")

    data = df['VIC_AGE_GROUP'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90 )
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write("""#### Who mostly commit these crimes""")
    st.pyplot(fig1)

    data1=df['LAW_CAT_CD'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(data1, labels=data1.index, autopct="%1.1f%%", shadow=True, startangle=90 )
    ax2.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write("""#### Types of crimes that mostly occur""")
    st.pyplot(fig2)


    data2=df['VIC_SEX'].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.pie(data2, labels=data2.index, autopct="%1.1f%%", shadow=True, startangle=90 )
    ax3.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write("""#### Who are most threatened """)
    st.pyplot(fig3)


    st.write("""#### Types of offenses""")
    data3 = df['OFNS_DESC'].value_counts()
    st.bar_chart(data3)

