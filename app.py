#basic Structure
import streamlit as st 
import pandas as pd
from joblib import load
import psycopg2
#joblib is used to save and load ml models 

# load model  and columns
model=load('tree.joblib')
train_columns =load('columns.joblib')
data=load('data.joblib') #this is x

# Database

conn = psycopg2.connect(
    host="localhost",
    database="Titanic_db",
    user="postgres",
    password="gkaur04",
    port="5432"
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    pclass varchar(50),
    gender VARCHAR(10),
    age_category varchar(30),
    fare_category varchar(30),
    embarked VARCHAR(50),
    family_onboard varchar(50),
    prediction VARCHAR(20)
)
""")

conn.commit()




# Page Setting 
st.set_page_config(page_title="Titanic Predictions Interface",layout="wide")

st.title("🚢 Titanic Dashboard")
st.markdown("### *Explore passenger data + Predict Survival*")
st.divider()
# ------------------------------------------------------------------------------------------------
#sidebar


st.sidebar.header("🎯 Passengers Details")
Pclass=st.sidebar.radio("Passsenger Class",["First Class","Second Class","Third Class"])
class_map={'First Class':0,'Second Class':1,'Third Class':2}
Pclass=class_map[Pclass]


Sex= st.sidebar.radio("Sex",["Male","Female"])
Sex=1 if Sex=="Male" else 0

Embarked=st.sidebar.selectbox("Embarked",['Cherbourge','Queenstown','Southampton'])
embarked_map={'Cherbourge':0,'Queenstown':1,'Southampton':2}
Embarked=embarked_map[Embarked]


Age_cat=st.sidebar.selectbox("Age Category",['Adults','Senior Citizens','Teens','Youth'])
age_map={'Adults':0,'Senior Citizens':1,'Teens':2,'Youth':3}
Age_cat=age_map[Age_cat]


Fare_cat=st.sidebar.selectbox("Fare Category",['Low','Medium','High','Expensive'])
fare_map={'Expensive':0,'High':1,'Low':2,'Medium':3}
Fare_cat=fare_map[Fare_cat]


Family=st.sidebar.radio("Family Onboards?",["No","Yes"])
Family = 1 if Family=='Yes' else 0

#----------------------------------------------------------------------------------

inputdf=pd.DataFrame({
    'Passenger Class':[Pclass],
    'Sex':[Sex],
    'Embarked':[Embarked],
    'Age Category':[Age_cat],
    'Fare Category':Fare_cat,
    'Family':[Family]
})

# as model expects data in table form i.e Dataframe
# here in streamlit we pass/gives individual values
# So we convert inputs to a dataframe
# and ,as DataFrame needs Rows , thus we have utilized list[]

#======================================================================================

# Match Training Columns
inputdf=inputdf.reindex(columns=train_columns,fill_value=0)
# this ensures same columns,Same order and no missing Values
# ------------------------------------------------------------------------------------



# Dataset Insights
st.subheader("📊 Dataset Insights")

col1, col2, col3 = st.columns(3)

# Passenger Class
with col1:
    st.metric("Total Passengers", len(data))

    pclass_counts = data['Pclass'].value_counts().sort_index()
    pclass_counts.index = [
        "First Class",
        "Second Class",
        "Third Class"
    ]

    st.bar_chart(pclass_counts)

# Gender
with col2:
    male_count = int((data['Sex'] == 1).sum())
    female_count = int((data['Sex'] == 0).sum())

    g1, g2 = st.columns(2)

    with g1:
        st.metric("Female", female_count)

    with g2:
        st.metric("Male", male_count)

    gender_counts = data['Sex'].value_counts().sort_index()
    gender_counts.index = [
        "Female",
        "Male"
    ]

    st.bar_chart(gender_counts)

# Embarked
with col3:
    st.metric("Unique Embarked", data['Embarked'].nunique())

    embarked_counts = data['Embarked'].value_counts().sort_index()
    embarked_counts.index = [
        "Cherbourg",
        "Queenstown",
        "Southampton"
    ]

    st.bar_chart(embarked_counts)





st.divider()

# --------------------------------------------------------------------------------
st.subheader("🔮 Prediction Panel")
pred_prob_col,reset_col=st.columns(2)
with pred_prob_col:
    if st.button("Predict"):
        result=model.predict(inputdf)
        prediction=int(result[0])
        

        prob=model.predict_proba(inputdf)
    
        if result[0]==1:
            st.success("Survived ✅ ")
        else:
            st.error("Did Not Survived ❌")

        
        Pclass_db = (
    'First Class' if Pclass == 0
    else 'Second Class' if Pclass == 1
    else 'Third Class'
       )

        sex_db = 'Male' if Sex == 1 else 'Female'

        embarked_db = (
                'Cherbourg' if Embarked == 0
              else 'Queenstown' if Embarked == 1
                 else 'Southampton'
              )

        age_category_db = (
    'Adults' if Age_cat == 0
    else 'Senior Citizens' if Age_cat == 1
    else 'Teens' if Age_cat == 2
    else 'Youth'
)

        fare_category_db = (
    'Expensive' if Fare_cat == 0
    else 'High' if Fare_cat == 1
    else 'Low' if Fare_cat == 2
    else 'Medium'
)

        family_db = 'Yes' if Family == 1 else 'No'

        cursor.execute("""
        INSERT INTO predictions
        (pclass, gender, age_category, fare_category, embarked,family_onboard, prediction)
         VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (Pclass_db, sex_db, age_category_db, fare_category_db, embarked_db,family_db, prediction))

        conn.commit()

        st.success(f"Prediction: {result}")

        st.session_state['prob']=prob
        st.metric("Survival Probability: ",f"{round(st.session_state['prob'][0][1],2)*100}%")
with reset_col:
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()
st.divider()   

# ----------------------------input summary----------------------------------------------------------
st.subheader("🧾 Input Summary")

summary_df = pd.DataFrame({
    'Passenger Class':['First Class' if Embarked==0 else 'Second Class' if Embarked==1 else 'Third Class'],
    'Sex':['Male' if Sex==1 else 'Female'],
    'Embarked':['Cherbourge' if Embarked==0 else 'Queenstown' if Embarked==1 else 'Southampton'],
    'Age Category':['Adults' if Age_cat==0 else 'Senior Citizens' if Age_cat==1 else 'Teens' if Age_cat==2 else 'Youth'],
    'Fare Category':['Expensive' if Fare_cat==0 else 'High' if Fare_cat==1 else 'Low' if Fare_cat==2 else 'Medium'],
    'Family':['Yes' if Family==1 else 'No']
})

st.write(summary_df)
