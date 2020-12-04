# Security
# passlib,hashlib,bcrypt,scrypt
import hashlib
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sqlite3
import os
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# DB Management
conn = sqlite3.connect('data1.db')
c = conn.cursor()

# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username, password))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style.css")
# adding color as blue to sidebar
st.markdown(
    """                                                                     
    <style>                                                                 
    .sidebar .sidebar-content {                                             
        background-image: linear-gradient(#053552,#053552);                 
        color: white;                                                       
    }                                                                       
    </style>                                                                
    """, unsafe_allow_html=True,
)

def main():
    #st.markdown("<h1 style='text-align: center; color: white;'>  Credit Crack </h1>", unsafe_allow_html=True)
    #st.markdown("<h3 style='text-align: right; color: white;'> - Credit Card Fraud Detection System </h3>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 style='text-align: left; color: white;'>Menu</h3>", unsafe_allow_html=True)
    menu = ["Home", "Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        #st.markdown("<h1 style='text-align: left; color: white;'> Home </h1>",unsafe_allow_html=True)
        image = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/Credit1.png')
        st.markdown("<h1 style='text-align: left; color: white;'> </h1>",unsafe_allow_html=True)
        st.image(image, caption='', use_column_width=True)
        image2 = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/Credit2.png')
        st.image(image2, caption='', use_column_width=True)
        image3 = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/Credit3.png')
        st.image(image3, caption='', use_column_width=True)
        image4 = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/Credit4.png')
        st.image(image4, caption='', use_column_width=True)

    elif choice == "Login":
        #st.subheader("Login Section")
        st.markdown("<h1 style='text-align: left; color: white;'></h1>", unsafe_allow_html=True)
        st.sidebar.markdown("<h2 style='text-align: left; color: white;'>Username</h2>", unsafe_allow_html=True)
        username = st.sidebar.text_input("User Name")
        st.sidebar.markdown("<h2 style='text-align: left; color: white;'>Password</h2>", unsafe_allow_html=True)
        password = st.sidebar.text_input("Password", type='password')
        st.sidebar.markdown("<h2 style='text-align: left; color: white;'>Login</h2>", unsafe_allow_html=True)

        if st.sidebar.checkbox("Login"):
            image = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/Credit5.png')
            st.image(image, caption='', use_column_width=True)
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username, check_hashes(password, hashed_pswd))
            if result:


                st.success("Logged In as {}".format(username))
                st.markdown("<h1 style='text-align: left; color: white;'>Stages of Machine Learning </h1>", unsafe_allow_html=True)
                task = st.selectbox("Activities",["Dataset Exploration", "Data Visualization", "Credit Card Fraud Detctor", "Profiles"])
                if task == "Dataset Exploration":
                    image = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/Credit6.png')
                    st.image(image, caption='', use_column_width=True)
                    st.markdown("<h1 style='text-align: left; color: white;'> Credit Card Dataset Transcation "
                                " Exploration""</h1>", unsafe_allow_html=True)
                    st.markdown("<h2 style='text-align: left; color: white;'>Select file</h2>", unsafe_allow_html=True)
                    def file_selector(folder_path='./dataset'):
                        filenames = os.listdir(folder_path)
                        selected_filename = st.selectbox("Select",filenames)
                        return os.path.join(folder_path, selected_filename)

                    filename = file_selector()
                    st.info("You Selected {}".format(filename))

                    # Read Data
                    df = pd.read_csv(filename)
                    # Show Dataset
                    st.markdown("<h3 style='text-align:left ; color: white;'>Click Here For Given Dataset Information</h3>",unsafe_allow_html=True)
                    if st.checkbox("Dataset"):
                        st.markdown( "<h3 style='text-align:left ; color: white;'>Credit Card transcation Dataset </h3>",unsafe_allow_html=True)
                        st.dataframe(df.head(20))
                        st.markdown( "<h3 style='text-align:left ; color: white;'>Credit Card transcation dataset Columns</h3>",unsafe_allow_html=True)
                        st.write(df.columns)

                        st.markdown( "<h3 style='text-align:left ; color: white;'>Shape of Credit Card transcation dataset</h3>",unsafe_allow_html=True)
                        st.write(df.shape)
                        st.markdown( "<h3 style='text-align:left ; color: white;'>Note: Rows = 284807 ,Columns= 31</h3>",unsafe_allow_html=True)

                        st.markdown("<h3 style='text-align:left ; color: white;'>Select Columns from Given Dataset</h3>",unsafe_allow_html=True)
                        all_columns = df.columns.tolist()
                        selected_columns = st.multiselect("Select", all_columns)
                        new_df = df[selected_columns]
                        st.dataframe(new_df)
                        st.markdown("<h3 style='text-align:left ; color: white;'> Values Counts Given Dataset</h3>",unsafe_allow_html=True)
                        st.write(df.iloc[:, -1].value_counts())
                        st.markdown("<h3 style='text-align:left ; color: white;'> Datatype Given Dataset</h3>",unsafe_allow_html=True)
                        st.write(df.dtypes)
                        st.markdown("<h3 style='text-align:left ; color: white;'>Is there any null values in given Dataset</h3>",unsafe_allow_html=True)
                        st.write(df.isna().sum())
                        st.markdown("<h3 style='text-align:left ; color: white;'>Summary given Dataset</h3>",unsafe_allow_html=True)
                        st.write(df.describe().T)

                    st.markdown("<h3 style='text-align:left ; color: white;'>Click here for detail exploration of Credit Card Transcation Dataset</h3>",unsafe_allow_html=True)
                    if st.checkbox("Data Exploration"):
                        st.markdown("<h3 style='text-align:left ; color: white;'>No of Valid cases in credit card dataset</h3>",unsafe_allow_html=True)
                        counts = df.Class.value_counts()
                        normal = counts[0]
                        fraudulent = counts[1]
                        perc_normal = (normal / (normal + fraudulent)) * 100
                        perc_fraudulent = (fraudulent / (normal + fraudulent)) * 100
                        st.write('There were {} non-fraudulent transactions ({:.3f}%)'.format(normal, perc_normal))
                        st.markdown("<h3 style='text-align:left ; color: white;'>No of Fraud cases in credit card dataset </h3>",unsafe_allow_html=True)
                        st.write('There were {} fraudulent transactions ({:.3f}%).'.format(fraudulent, perc_fraudulent))
                        st.markdown("<h3 style='text-align:left ; color: white;'>Amount Details of Fraud transcation </h3>",unsafe_allow_html=True)
                        fraud = df[df['Class'] == 1]
                        st.write(fraud.Amount.describe())
                        st.markdown("<h3 style='text-align:left ; color: white;'>Amount Details of Valid transcation </h3>",unsafe_allow_html=True)
                        valid = df[df['Class'] == 0]
                        st.write(valid.Amount.describe())
                        st.markdown("<h3 style='text-align:left ; color: white;'>Observation from Given dataset</h3>",unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align:left ; color: white;'>1) Except for the transaction and amount we dont know what "
                            "the other columns are (due to privacy reasons).</h3>",unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align:left ; color: white;'>2) The only thing we know, is that those columns that are unknown "
                            "have been scaled already.</h3>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align:left ; color: white;'>3) The transaction amount is relatively"" small. The mean of all the "
                            "mounts made is approximately USD 88 ""have been scaled already.</h3>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align:left ; color: white;'>4) There are no Null values, so we don't have to work on ways to replace values.</h3>",
                            unsafe_allow_html=True)

                elif task == "Data Visualization":
                    #st.subheader("Data Visualization of Credit card Transcation Dataset")
                    image = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/Credit7.png')
                    st.image(image, caption='', use_column_width=True)
                    st.markdown("<h1 style='text-align: left; color: white;'> Data Visualization of Credit card Transcation Dataset </h1>",unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: left; color: white;'>Select file</h3>", unsafe_allow_html=True)
                    def file_selector(folder_path='./dataset'):
                        filenames = os.listdir(folder_path)
                        selected_filename = st.selectbox("Select A file", filenames)
                        return os.path.join(folder_path, selected_filename)
                    filename = file_selector()
                    st.info("You Selected {}".format(filename))
                    # Read Data
                    df = pd.read_csv(filename)
                    #Pie Plot
                    st.markdown("<h3 style='text-align: left; color: white;'>Generating Pie Chart Given Dataset </h3>",unsafe_allow_html=True)
                    if st.checkbox("Pie Plot"):
                        fraud = len(df[df['Class'] == 1])
                        nonfraud = len(df[df['Class'] == 0])
                        # Data to  plot
                        labels = 'Fraud', 'Not Fraud'
                        sizes = [fraud, nonfraud]
                        # Plot
                        fig, ax = plt.subplots(figsize=(10, 8))
                        st.write(plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=0))
                        st.write(plt.title('Ratio Of Fraud Vs Non-Fraud\n', fontsize=20))
                        st.write(sns.set_context("paper", font_scale=2))
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                        st.set_option('deprecation.showPyplotGlobalUse', False)

                    st.markdown("<h3 style='text-align: left; color: white;'>Generating Bar Plot of Given Dataset </h3>",unsafe_allow_html=True)
                    if st.checkbox("Bar Plot"):
                        counts = df.Class.value_counts()
                        fig ,ax = plt.subplots(figsize=(8, 6))
                        st.write(sns.barplot(x=counts.index, y=counts, color='green'))
                        st.write(plt.title('Count of Fraudulent vs. Non-Fraudulent Transactions'))
                        st.write(plt.ylabel('Count'))
                        st.write(plt.xlabel('Class (0:Non-Fraudulent, 1:Fraudulent)'))
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                    st.markdown("<h3 style='text-align: left; color: white;'>Generating distribution of Amount and Time of credit card dataset</h3>",unsafe_allow_html=True)
                    if st.checkbox("Distribution of Amount and Time"):
                        import warnings
                        warnings.simplefilter(action='ignore', category=UserWarning)
                        fig, ax = plt.subplots(figsize=(10, 7))
                        st.write(sns.set_style("whitegrid"),ax=ax)
                        st.write(sns.FacetGrid(df, hue="Class", size=10).map(plt.scatter, "Time", "Amount").add_legend())
                        st.pyplot()
                        st.write(sns.FacetGrid(df, hue="Class", size=10).map(plt.scatter, "Amount", "Time").add_legend())
                        st.pyplot()
                        st.success("You have successfully created distribution of amount and time")
                        st.markdown( "<h3 style='text-align: left; color: white;'> Observation </h3>",unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align:left ; color: white;'>1) From the above two plots it is clearly visible "
                            "that there are frauds only on the transactions which have transaction amount approximately less than 2500. "
                            "Transactions which have transaction amount approximately above 2500 have no fraud.</h3>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: left; color: white;'>2) As per with the time, the frauds in t"
                            "he transactions are evenly distributed throughout time. </h3>",
                            unsafe_allow_html=True)

                    st.markdown("<h3 style='text-align: left; color: white;'>Generating distribution of Class</h3>",unsafe_allow_html=True)
                    if st.checkbox("Distribution of Class atrribute of credit card dataset "):
                        st.write(plt.title('Distribution of "Class" Atrribute'))
                        fig, ax = plt.subplots(figsize=(10, 7))
                        st.write(sns.distplot(df['Class'], color='#00008B',ax= ax))
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()

                    st.markdown("<h3 style='text-align: left; color: white;'>Generating distribution of Amount </h3>",unsafe_allow_html=True)
                    if st.checkbox("Distribution of Amount Atrribute of Credit Card Dataset"):
                        st.write(plt.title('Distribution of "Amount" Atrribute'))
                        fig, ax = plt.subplots(figsize=(10, 7))
                        st.write(sns.distplot(df['Amount'], color='red',ax=ax))
                        st.pyplot()

                    st.markdown("<h3 style='text-align: left; color: white;'>Generating distribution of Time </h3>",unsafe_allow_html=True)
                    if st.checkbox("Distribution of Time Atrribute of Credit Card Dataset"):
                        st.write(plt.title('Distribution of "Time " Atrribute'))
                        fig, ax = plt.subplots(figsize=(10, 7))
                        st.write(sns.distplot(df['Time'], color='green',ax=ax))
                        st.pyplot()

                    st.markdown(
                        "<h3 style='text-align: left; color: white;'>Distribution of Time and Class Atrribute of Credit "
                        "Card Dataset in terms Box plot and Whiskers </h3>", unsafe_allow_html=True)
                    if st.checkbox("Distribution of Time and Class Atrribute of Credit Card Dataset in terms Box plot and Whiskers "):
                        st.write(sns.boxplot(x="Class", y="Time", data=df))
                        st.pyplot()
                        st.markdown("<h3 style='text-align: left; color: white;'> Observation :</h3>", unsafe_allow_html=True)
                        st.markdown("<h4 style='text-align: left; color: white;'> By looking at the above box plot we can say that both fraud & genuine "
                            "transactions occur throughout time and there is no distinction between them. </h4>", unsafe_allow_html=True)
                        st.write(sns.boxplot(x="Class", y="Amount", data=df))
                        st.write(plt.ylim(0, 5000))
                        st.pyplot()
                        st.markdown("<h3 style='text-align: left; color: white;'> Observation :</h3>", unsafe_allow_html=True)
                        st.markdown("<h4 style='text-align: left; color: white;'> From above box plot we can easily infer that there are no fraud transactions occur above the transaction"
                            "amount of 3000. All of the fraud transactions have transaction amount less than 3000.However, "
                            "there are many transactions which have a transaction amount greater than 3000 and all of them are genuine. " "</h4>",
                            unsafe_allow_html=True)
                        creditCard_genuine = df.loc[df["Class"] == 0]
                        creditCard_fraud = df.loc[df["Class"] == 1]
                        st.write(plt.plot(creditCard_genuine["Time"], np.zeros_like(creditCard_genuine["Time"]), "o"))
                        st.write(plt.plot(creditCard_fraud["Time"], np.zeros_like(creditCard_fraud["Time"]), "o"))
                        st.pyplot()
                        # X-axis: Time
                        st.markdown("<h3 style='text-align: left; color: white;'> Observation :</h3>", unsafe_allow_html=True)
                        st.markdown("<h4 style='text-align: left; color: white;'> Fraud and genuine transactions are spread evenly "
                             "thought time and there is no clear distinction. " "</h4>",unsafe_allow_html=True)

                        st.write(plt.plot(creditCard_genuine["Amount"], np.zeros_like(creditCard_genuine["Amount"]), "o"))
                        st.write(plt.plot(creditCard_fraud["Amount"], np.zeros_like(creditCard_fraud["Amount"]), "o"))
                        st.pyplot()

                        st.markdown("<h3 style='text-align: left; color: white;'> Observation :</h3>", unsafe_allow_html=True)
                        st.markdown(
                            "<h4 style='text-align: left; color: white;'>It can clearly be observed from this 1D scatter plot that the fraud transactions are"
                            "there only on the transaction amount less than 2500. " "</h4>", unsafe_allow_html=True)# X-axis: Amount
                        st.markdown("<h3 style='text-align: left; color: white;'> =================================================================</h3>", unsafe_allow_html=True)

                    st.markdown("<h3 style='text-align: left; color: white;'>Generating Correlation between all "
                                "other different attributes(Unbalanced Dataset) </h3>",unsafe_allow_html=True)
                    if st.checkbox("Correlation between all other different attributes"):
                        corrmat = df.corr()
                        fig, ax = plt.subplots(figsize=(15, 10))
                        st.write(plt.title('Correlation between all other different attributes'))
                        st.write(sns.heatmap(corrmat, vmax=1,square=True, cmap='viridis',ax=ax))
                        st.pyplot()
                        st.markdown("<h3 style='text-align: left; color: white;'> Note(Important)</h3>", unsafe_allow_html=True)


                    st.markdown("<h3 style='text-align: left; color: white;'>Correlation among the Time ,Amount and Class representation in term of Data</h3>",unsafe_allow_html=True)
                    if st.checkbox("Correlation among the Time , Amount and Class representation in term of Data"):
                        df1 = df.loc[:, ['Time', 'Amount', 'Class']]  # Selecting data of interest
                        df1.head()
                        st.write(df1.corr())

                    st.markdown(
                        "<h3 style='text-align: left; color: white;'>Correlation among the Time , Amount "
                        "and Class representation in term of plot </h3>", unsafe_allow_html=True)
                    if st.checkbox("Correlation among the Time , Amount and Class representation in term of plot"):
                        fig ,ax = plt.subplots(figsize=(15, 10))
                        st.write(sns.set(font_scale=3))
                        corrmat = df1.corr()
                        st.write(sns.heatmap(corrmat, vmax=1, square=True, annot=True, cmap='viridis',ax=ax))
                        st.write(plt.title('Correlation between different attributes'))
                        st.pyplot()

                    st.markdown("<h3 style='text-align: left; color: white;'>Time and Amount,Class representation in terms Gridspec </h3>", unsafe_allow_html=True)
                    if st.checkbox("Time and Amount , Class representation in terms Gridspec"):
                        fig ,ax = plt.subplots(figsize=(20, 30 * 10))
                        features = df1.iloc[:, 0:30].columns
                        gs = gridspec.GridSpec(30, 1)
                        for i, feature in enumerate(df[features]):
                            ax = plt.subplot(gs[i])
                            st.write(sns.distplot(df1[feature][df1.Class == 1], bins=50 ,ax=ax))
                            st.write(sns.distplot(df1[feature][df1.Class == 0], bins=50,ax=ax))
                            st.write(ax.set_xlabel(''))
                            st.write(ax.set_title('Feature:' + str(feature)))
                        st.pyplot()
                        st.set_option('deprecation.showPyplotGlobalUse', False)

                elif task == "Credit Card Fraud Detctor":
                    image = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/Credit9.png')
                    st.image(image, caption='', use_column_width=True)
                    st.markdown("<h1 style='text-align: left; color: white;'>Credit Card Fraud Detctor</h1>", unsafe_allow_html=True)
                    def file_selectors(folder_paths='./dataset'):
                        filenames1 = os.listdir(folder_paths)
                        selected_filename1 = st.selectbox("Select A file", filenames1)
                        return os.path.join(folder_paths, selected_filename1)

                    df = pd.read_csv('C:/Users/Vikas/PycharmProject/CreditCardDetection/dataset/credit.csv')
                    X = df.drop(['Class'], axis=1)
                    y = df["Class"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                    # concatenate our training data back together
                    X = pd.concat([X_train, y_train], axis=1)
                    st.write("Concatenate our training data back together")
                    X.head()
                    filename1 = file_selectors()
                    st.info("You Selected {}".format(filename1))

                    # Read Data
                    df1 = pd.read_csv(filename1)
                    features = pd.DataFrame(df1, index=[0])
                    features1 = features.drop(['Class'], axis=1)
                    user_input = features1

                    # Set a subheader and display the user input
                    st.subheader("User Input:")
                    st.write(user_input)

                    # random forest model creation
                    from sklearn.ensemble import RandomForestClassifier
                    RandomForestClassifier = RandomForestClassifier()
                    RandomForestClassifier.fit(X_train, y_train)

                    # predictions
                    rfc_pred = RandomForestClassifier.predict(X_test)
                    st.subheader("Model Test Accuracy Score :")
                    st.write('Accuracy(Test score):', accuracy_score(y_test, rfc_pred))
                    prediction = RandomForestClassifier.predict(user_input)

                    # Set the subheader the Classification
                    st.subheader("Classification of given user input :")
                    st.write(prediction)

                    st.markdown("<h2 style='text-align: left; color:white;'>Conculsion</h2>", unsafe_allow_html=True)
                    st.markdown(
                        "<h2 style='text-align: left; color:white;'>Best Classifier for Credit Card Detectiona and "
                        "Preventation is Random Forest Algorithms so for for resolving Business problem we will use this algorithms </h2>",
                        unsafe_allow_html=True)

                elif task == "Profiles":
                    #st.subheader("User Profiles")
                    image = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/Credit10.png')
                    st.image(image, caption='', use_column_width=True)
                    st.markdown("<h1 style='text-align: left; color: white;'>User Profiles</h1>", unsafe_allow_html=True)
                    user_result = view_all_users()
                    clean_db = pd.DataFrame(user_result, columns=["Username", "Password"])
                    st.dataframe(clean_db)
            else:
                st.warning("Incorrect Username/Password")

    elif choice == "SignUp":
        #st.subheader("Create New Account")
        image = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/Credit11.png')
        st.image(image, caption='', use_column_width=True)
        st.markdown("<h3 style='text-align: left; color: white;'>Username</h3>", unsafe_allow_html=True)
        new_user = st.text_input("Username")
        st.markdown("<h3 style='text-align: left; color: white;'>Password</h3>", unsafe_allow_html=True)
        new_password = st.text_input("Password", type='password')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user, make_hashes(new_password))
            st.markdown("<h3 style='text-align: left; color: white;'>You have successfully created a valid Account</h3>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: left; color: white;'>Go to Login Menu to login</h3>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
