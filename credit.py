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
import string
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.utils import resample
from PIL import Image
from sklearn.linear_model import LogisticRegression

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


# DB Management

conn = sqlite3.connect('data.db')
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
# adding color as green
st.markdown(
    """                                                                     
    <style>                                                                 
    .sidebar .sidebar-content {                                             
        background-image: linear-gradient(#84d162,#2c8530);                 
        color: white;                                                       
    }                                                                       
    </style>                                                                
    """, unsafe_allow_html=True,
)


def main():

    st.markdown("<h1 style='text-align: center; color: black;'>  Credit Crack </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: right; color: black;'> - Credit Card Fraud Detection System </h3>", unsafe_allow_html=True)

    menu = ["Home", "Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":

        st.markdown("<h1 style='text-align: left; color: black;'> Home  </h1>",
                    unsafe_allow_html=True)
        # open and display an image
        image = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/creditcrack1.png')
        st.image(image, caption='Credit Card Fraud Detection Project', use_column_width=True)
        st.markdown("<h2 style='text-align: left; color: black;'>Aim </h2>",
                    unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left; color: black;'> The challenge is to recognize fraudulent credit card "
                    "transactions so that the customers of credit card companies are not charged for items that they did not purchase."
                    "We will use a variety of machine learning algorithms that will be able to discern fraudulent from non-fraudulent one. "
                    "By the end of this machine learning project, you will learn how to implement machine learning algorithms to perform classification"
                    ".</h3>",unsafe_allow_html=True)
        image2 = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/creditcrack2.png')
        st.image(image2, caption='', use_column_width=True)
        image3 = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/creditcard3.png')
        st.image(image3, caption='', use_column_width=True)
        image4 = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/creditcard4.png')
        st.image(image4, caption='', use_column_width=True)
        image5 = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/creditcard5.png')
        st.image(image5, caption='', use_column_width=True)
        image7 = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/creditcard6.png')
        st.image(image7, caption='', use_column_width=True)
    elif choice == "Login":
        #st.subheader("Login Section")
        st.markdown("<h1 style='text-align: left; color: black;'> Login Section </h1>", unsafe_allow_html=True)

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username, check_hashes(password, hashed_pswd))
            if result:

                st.success("Logged In as {}".format(username))

                task = st.selectbox("Activities",
                                    ["Dataset Exploration", "Data Visualization", "Predication Using Machine learning",
                                     "CreditCardRisk Assessement", "Profiles"])
                if task == "Dataset Exploration":
                    st.markdown("<h1 style='text-align: left; color: black;'> Credit Card Dataset Transcation "
                                " Exploration""</h1>", unsafe_allow_html=True)
                    def file_selector(folder_path='./dataset'):
                        filenames = os.listdir(folder_path)
                        selected_filename = st.selectbox("Select A file", filenames)
                        return os.path.join(folder_path, selected_filename)

                    filename = file_selector()
                    st.info("You Selected {}".format(filename))

                    # Read Data
                    df = pd.read_csv(filename)
                    # Show Dataset
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click Here For Show the Credit Card Transcation Dataset</h3>",unsafe_allow_html=True)
                    if st.checkbox("Show Dataset"):
                        st.dataframe(df.head(20))
                        st.success("Rows of Given Dataset")
                    # Show Columns
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click Here For Show the Credit Card Dataset Columns</h3>",unsafe_allow_html=True)
                    if st.checkbox("Column Names"):
                        st.write(df.columns)
                        st.success("No Of Columns in the Given Dataset ")
                    # Show Shape
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click Here For Show the Shape of Given Datset of Credit Card Transcation</h3>",unsafe_allow_html=True)
                    if st.checkbox("Shape of Dataset"):
                        st.success("Shape of Given Dataset of Credit Card Transaction")
                        data_dim = st.radio("Show Dimension By ", ("Rows", "Columns"))
                        if data_dim == 'Rows':
                            st.text("Number of Rows")
                            st.write(df.shape[0])
                        elif data_dim == 'Columns':
                            st.text("Number of Columns")
                            st.write(df.shape[1])
                        else:
                            st.write(df.shape)
                    # Select Columns
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click Here for Selecting Columns from Given Datset of Credit Card Transcation</h3>",unsafe_allow_html=True)
                    if st.checkbox("Select Columns To Show"):
                        all_columns = df.columns.tolist()
                        selected_columns = st.multiselect("Select", all_columns)
                        new_df = df[selected_columns]
                        st.dataframe(new_df)
                    # Show Values
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click here for Getting Values of Given Dataset</h3>",unsafe_allow_html=True)
                    if st.checkbox("Value Counts"):
                        st.text("Value Counts By Target/Class")
                        st.write(df.iloc[:, -1].value_counts())
                    # Show Datatypes
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click here for Getting Datatype of Given Credit Card Transcation</h3>",unsafe_allow_html=True)
                    if st.checkbox("Data Types"):
                        st.write(df.dtypes)
                        st.success("Data types of given dataset")
                    #Checking null value
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click here for checking is there any null values of Given Credit Card Transcation Dataset</h3>",unsafe_allow_html=True)
                    if st.checkbox("Checking is there any null values in Dataset"):
                        st.write(df.isna().sum())
                        st.success("No null values in given dataset")

                    # Show Summary
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click here for Summary of Credit Card Transcation Dataset </h3>",unsafe_allow_html=True)
                    if st.checkbox("Summary"):
                        st.success("Summary of Datset")
                        st.write(df.describe().T)
                        st.success("Summary of given dataset ")

                    # No of valid cases in dataset
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click here for no of valid cases in credit card dataset </h3>",unsafe_allow_html=True)
                    if st.checkbox("No Of Valid Cases in Credit Card Dataset"):
                        st.write('Valid Transactions: {}'.format(len(df[df['Class'] == 0])))
                        st.success("Valid cases in given dataset")

                    #No of fraud cases in dataset
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click here for no of Fraud cases in credit card dataset </h3>",unsafe_allow_html=True)
                    if st.checkbox("No Of Fraud Cases in Credit Card Dataset"):
                        st.write('Fraud Cases: {}'.format(len(df[df['Class'] == 1])))
                        st.success("Fraud cases in given dataset")

                    # valid transcation in %
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click here for valid transcation in terms of Percentage(%) </h3>",unsafe_allow_html=True)
                    if st.checkbox(" Valid Transcation in term Percentage"):
                        counts = df.Class.value_counts()
                        normal = counts[0]
                        fraudulent = counts[1]
                        perc_normal = (normal / (normal + fraudulent)) * 100
                        perc_fraudulent = (fraudulent / (normal + fraudulent)) * 100
                        st.write('There were {} non-fraudulent transactions ({:.3f}%)'.format(normal, perc_normal))
                        st.success(" Valid cases in percentage form")

                    # Fraud transcation in %
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click here for fraud transcation in terms of Percentage(%) </h3>",unsafe_allow_html=True)
                    if st.checkbox("Fraud Transcation in term Percentage"):
                        st.write('There were {} fraudulent transactions ({:.3f}%).'.format(fraudulent, perc_fraudulent))
                        st.success("Fraud cases in percentage form")

                     # Amount details in fraud transcation
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click here for Amount Details of Fraud transcation </h3>",unsafe_allow_html=True)
                    if st.checkbox("Amount details of the fraudulent transaction"):
                        st.write("Amount details of the fraudulent transaction")
                        fraud = df[df['Class'] == 1]
                        st.write(fraud.Amount.describe())
                        st.success("Amount details of fraud transaction")

                     # Amount details in valid transcation
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click here for Amount Details of Valid transcation</h3>",unsafe_allow_html=True)
                    if st.checkbox(" Amounts Details of valid transaction"):
                        st.write("Amount details of valid transaction")
                        valid = df[df['Class'] == 0]
                        st.write(valid.Amount.describe())
                        st.success("Amount details of valid transaction")
                    st.markdown("<h3 style='text-align:left ; color: black;'>Click here for Observation from above dataset</h3>",unsafe_allow_html=True)
                    if st.checkbox("Observation"):
                        st.markdown(
                            "<h2 style='text-align:left ; color: black;'>1) Except for the transaction and amount we dont know what "
                            "the other columns are (due to privacy reasons).</h2>",unsafe_allow_html=True)
                        st.markdown(
                            "<h2 style='text-align:left ; color: black;'>2) The only thing we know, is that those columns that are unknown "
                            "have been scaled already.</h3>", unsafe_allow_html=True)
                        st.markdown(
                            "<h2 style='text-align:left ; color: black;'>3) The transaction amount is relatively"" small. The mean of all the "
                            "mounts made is approximately USD 88 ""have been scaled already.</h3>", unsafe_allow_html=True)
                        st.markdown(
                            "<h2 style='text-align:left ; color: black;'>4) There are no Null values, so we don't have to work on ways to replace values.</h2>",
                            unsafe_allow_html=True)
                elif task == "Data Visualization":
                    #st.subheader("Data Visualization of Credit card Transcation Dataset")
                    st.markdown("<h1 style='text-align: left; color: black;'> Data Visualization of Credit card "
                                "Transcation Dataset </h1>",unsafe_allow_html=True)

                    def file_selector(folder_path='./dataset'):
                        filenames = os.listdir(folder_path)
                        selected_filename = st.selectbox("Select A file", filenames)
                        return os.path.join(folder_path, selected_filename)

                    filename = file_selector()
                    st.info("You Selected {}".format(filename))

                    # Read Data
                    df = pd.read_csv(filename)
                    #Pie Plot
                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for generating Pie Chart </h3>",unsafe_allow_html=True)
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
                        st.pyplot()
                        st.success("You have successfully generated pie chart")
                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for generating Bar Plot of given dataset </h3>",unsafe_allow_html=True)
                    if st.checkbox("Bar Plot of Fraud and Legit Transcation"):
                        counts = df.Class.value_counts()
                        fig ,ax = plt.subplots(figsize=(8, 6))
                        st.write(sns.barplot(x=counts.index, y=counts, color='green'))
                        st.write(plt.title('Count of Fraudulent vs. Non-Fraudulent Transactions'))
                        st.write(plt.ylabel('Count'))
                        st.write(plt.xlabel('Class (0:Non-Fraudulent, 1:Fraudulent)'))
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                        st.success("You have successfully generated bar plot")
                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for generating distribution "
                                "of Amount and Time of credit card dataset</h3>",unsafe_allow_html=True)
                    if st.checkbox("Distribution of Amount and Time of credit card dataset"):
                        import warnings
                        warnings.simplefilter(action='ignore', category=UserWarning)
                        fig, ax = plt.subplots(figsize=(10, 7))
                        st.write(sns.set_style("whitegrid"),ax=ax)
                        st.write(sns.FacetGrid(df, hue="Class", size=10).map(plt.scatter, "Time", "Amount").add_legend())
                        st.pyplot()
                        st.write(sns.FacetGrid(df, hue="Class", size=10).map(plt.scatter, "Amount", "Time").add_legend())
                        st.pyplot()
                        st.success("You have successfully created distribution of amount and time")
                        st.markdown( "<h3 style='text-align: left; color: black;'> Observation </h3>",unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align:left ; color: black;'>1) From the above two plots it is clearly visible "
                            "that there are frauds only on the transactions which have transaction amount approximately less than 2500. "
                            "Transactions which have transaction amount approximately above 2500 have no fraud.</h3>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: left; color: black;'>2) As per with the time, the frauds in t"
                            "he transactions are evenly distributed throughout time. </h3>",
                            unsafe_allow_html=True)

                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for generating distribution of Class</h3>",unsafe_allow_html=True)
                    if st.checkbox("Distribution of Class atrribute of credit card dataset "):
                        st.write(plt.title('Distribution of "Class" Atrribute'))
                        fig, ax = plt.subplots(figsize=(10, 7))
                        st.write(sns.distplot(df['Class'], color='#00008B',ax= ax))
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                        st.success("You successfully created distribution of Class attribute")

                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for generating distribution of Amount </h3>",unsafe_allow_html=True)
                    if st.checkbox("Distribution of Amount Atrribute of Credit Card Dataset"):
                        st.write(plt.title('Distribution of "Amount" Atrribute'))
                        fig, ax = plt.subplots(figsize=(10, 7))
                        st.write(sns.distplot(df['Amount'], color='red',ax=ax))
                        st.pyplot()
                        st.success("You successfully created distribution of Amount attribute")

                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for generating distribution of Time </h3>",unsafe_allow_html=True)
                    if st.checkbox("Distribution of Time Atrribute of Credit Card Dataset"):
                        st.write(plt.title('Distribution of "Time " Atrribute'))
                        fig, ax = plt.subplots(figsize=(10, 7))
                        st.write(sns.distplot(df['Time'], color='green',ax=ax))
                        st.pyplot()
                        st.success("You successfully created distribution of time attribute")
                    if st.checkbox("Plot of all attribute of credit card dataset"):
                        st.write("df.hist(figsize = (60, 60),color='red')")
                        st.pyplot()
                        st.success("You successfully created distribution of all attribute")
                    # if st.checkbox("Histrogram of All the Atrributes of Credit Card Dataset"):
                    #     df1 = df.loc[df.index.drop_duplicates()]
                    #
                    #     df = pd.DataFrame(data=np.random.random(size=(50, 30)),
                    #                       columns=[i for i in string.ascii_lowercase[:6]])
                    #     st.write(df.hist(bins=20,figsize=(15,8),layout=(1, 3)))
                    #     st.pyplot()

                    st.markdown(
                        "<h3 style='text-align: left; color: black;'>Click here for Distribution of Time and Class Atrribute of Credit "
                        "Card Dataset in terms Box plot and Whiskers </h3>", unsafe_allow_html=True)
                    if st.checkbox("Distribution of Time and Class Atrribute of Credit Card Dataset in terms Box plot and Whiskers "):
                        st.write(sns.boxplot(x="Class", y="Time", data=df))
                        st.pyplot()
                        st.markdown("<h3 style='text-align: left; color: black;'> Observation :</h3>", unsafe_allow_html=True)
                        st.markdown(
                            "<h4 style='text-align: left; color: black;'> By looking at the above box plot we can say that both fraud & genuine "
                            "transactions occur throughout time and there is no distinction between them. </h4>", unsafe_allow_html=True)
                        st.write(sns.boxplot(x="Class", y="Amount", data=df))
                        st.write(plt.ylim(0, 5000))
                        st.pyplot()
                        st.success("You have succesfully created box plot")
                        st.markdown(
                            "<h3 style='text-align: left; color: black;'> Observation :</h3>", unsafe_allow_html=True)
                        st.markdown(
                            "<h4 style='text-align: left; color: black;'> From above box plot we can easily infer that there are no fraud transactions occur above the transaction"
                            "amount of 3000. All of the fraud transactions have transaction amount less than 3000.However, "
                            "there are many transactions which have a transaction amount greater than 3000 and all of them are genuine. " "</h4>",
                            unsafe_allow_html=True)
                        creditCard_genuine = df.loc[df["Class"] == 0]
                        creditCard_fraud = df.loc[df["Class"] == 1]

                        st.write(plt.plot(creditCard_genuine["Time"], np.zeros_like(creditCard_genuine["Time"]), "o"))
                        st.write(plt.plot(creditCard_fraud["Time"], np.zeros_like(creditCard_fraud["Time"]), "o"))
                        st.pyplot()
                        # X-axis: Time
                        st.markdown("<h3 style='text-align: left; color: black;'> Observation :</h3>", unsafe_allow_html=True)
                        st.markdown(
                            "<h4 style='text-align: left; color: black;'> Fraud and genuine transactions are spread evenly "
                             "thought time and there is no clear distinction. " "</h4>",unsafe_allow_html=True)
                        st.write(plt.plot(creditCard_genuine["Amount"], np.zeros_like(creditCard_genuine["Amount"]), "o"))
                        st.write(plt.plot(creditCard_fraud["Amount"], np.zeros_like(creditCard_fraud["Amount"]), "o"))
                        st.pyplot()
                        st.markdown("<h3 style='text-align: left; color: black;'> Observation :</h3>", unsafe_allow_html=True)
                        st.markdown(
                            "<h4 style='text-align: left; color: black;'>It can clearly be observed from this 1D scatter plot that the fraud transactions are"
                            "there only on the transaction amount less than 2500. " "</h4>", unsafe_allow_html=True)# X-axis: Amount
                        st.markdown(
                            "<h3 style='text-align: left; color: black;'> =================================================================</h3>", unsafe_allow_html=True)

                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for generating Correlation between all "
                                "other different attributes(Unbalanced Dataset) </h3>",unsafe_allow_html=True)
                    if st.checkbox("Correlation between all other different attributes"):
                        corrmat = df.corr()
                        fig, ax = plt.subplots(figsize=(15, 10))
                        st.write(plt.title('Correlation between all other different attributes'))
                        st.write(sns.heatmap(corrmat, vmax=1,square=True, cmap='viridis',ax=ax))
                        st.pyplot()
                        st.success("You have successfully created correlation of between all other different attributes'")
                        st.markdown(
                            "<h3 style='text-align: left; color: black;'> Note(Important)</h3>", unsafe_allow_html=True)

                        st.markdown(
                            "<h3 style='text-align: left; color: black;'>Click here for Correlation among the Time ,"
                            " Amount and Class representation in term of Data</h3>", unsafe_allow_html=True)

                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for Correlation among the Time ,"
                                " Amount and Class representation in term of Data</h3>",unsafe_allow_html=True)
                    if st.checkbox("Correlation among the Time , Amount and Class representation in term of Data"):
                        df1 = df.loc[:, ['Time', 'Amount', 'Class']]  # Selecting data of interest
                        df1.head()
                        st.write(df1.corr())
                        st.success("You successfully generate Correlation among the Time Amount and Class representation in term of Data")

                    st.markdown(
                        "<h3 style='text-align: left; color: black;'>Click here for Correlation among the Time , Amount "
                        "and Class representation in term of plot </h3>", unsafe_allow_html=True)
                    if st.checkbox("Correlation among the Time , Amount and Class representation in term of plot"):
                        fig ,ax = plt.subplots(figsize=(15, 10))
                        st.write(sns.set(font_scale=3))
                        corrmat = df1.corr()
                        st.write(sns.heatmap(corrmat, vmax=1, square=True, annot=True, cmap='viridis',ax=ax))
                        st.write(plt.title('Correlation between different attributes'))
                        st.pyplot()
                        st.success("You successfully correlation among the Time , Amount and Class representation in term of plot ")

                    st.markdown(
                        "<h3 style='text-align: left; color: black;'>Click here for Time and Amount , Class representation in terms Gridspec </h3>", unsafe_allow_html=True)
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

                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for plot of value counts</h3>",unsafe_allow_html=True)
                    if st.checkbox("Plot of Value Counts"):
                        st.text("Value Counts By Target")
                        all_columns_names = df.columns.tolist()
                        primary_col = st.selectbox("Primary Columm to GroupBy", all_columns_names)
                        selected_columns_names = st.multiselect("Select Columns", all_columns_names)
                        if st.button("Plot"):
                            st.text("Generate Plot")
                            if selected_columns_names:
                                vc_plot = df.groupby(primary_col)[selected_columns_names].count()
                            else:
                                vc_plot = df.iloc[:, -1].value_counts()
                            st.write(vc_plot.plot(kind="bar"))
                            st.pyplot()

                    # Customizable Plot
                    st.markdown("<h3 style='text-align: left; color: black;'>Customizable Plot</h3>",unsafe_allow_html=True)

                    all_columns_names = df.columns.tolist()
                    type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
                    selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

                    if st.button("Generate Plot"):
                        st.success(
                            "Generating Customizable Plot of {} for {}".format(type_of_plot, selected_columns_names))

                        # Plot By Streamlit
                        if type_of_plot == 'area':
                            cust_data = df[selected_columns_names]
                            st.area_chart(cust_data)

                        elif type_of_plot == 'bar':
                            cust_data = df[selected_columns_names]
                            st.bar_chart(cust_data)

                        elif type_of_plot == 'line':
                            cust_data = df[selected_columns_names]
                            st.line_chart(cust_data)

                        # Custom Plot
                        elif type_of_plot:
                            cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
                            st.write(cust_plot)
                            st.pyplot()

                elif task =="Predication Using Machine learning":
                    st.markdown("<h2 style='text-align: left; color: black;'>Predicating and Detecting Credit Fraud Transaction</h2>",unsafe_allow_html=True)

                    def file_selector(folder_path='./dataset'):
                        filenames = os.listdir(folder_path)
                        selected_filename = st.selectbox("Select A file", filenames)
                        return os.path.join(folder_path, selected_filename)

                    filename = file_selector()
                    st.info("You Selected {}".format(filename))
                    # Read Data
                    df = pd.read_csv(filename)
                    # Show Dataset
                    st.markdown("<h3 style='text-align: left; color: black;'>Click Here For Show the Credit Card Transcation "
                       "Dataset</h3>",unsafe_allow_html=True)
                    if st.checkbox("Show Dataset"):
                        st.dataframe(df.head(20))
                        st.success("Rows of Given Dataset")

                    st.markdown("<h2 style='text-align: left; color: black;'>Preparing our data for Model Building</h2>",unsafe_allow_html=True)

                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for separating data  into "
                                "dependent and independent variables</h3>",unsafe_allow_html=True)
                    if st.checkbox("Separating our data into Dependent and Independent Variables"):
                        # dividing the X and the Y from the dataset
                        # Separating our data into Dependent and Independent Variables
                        X = df.drop(['Class'], axis=1)
                        y = df["Class"]
                        st.write(X.shape)
                        st.write(y.shape)
                        # getting just the values for the sake of processing
                        # (its a numpy array with no columns)
                        xData = X.values
                        yData = y.values
                    st.markdown("<h3 style='text-align: left; color: black;'>Preparing our data for Model Building</h3>",unsafe_allow_html=True)
                    if st.checkbox("Training and Testing Data Bifurcation"):
                        st.markdown(
                            "<h3 style='text-align: left; color: black;'>We will be dividing the dataset into two main groups. "
                            "One for training the model and the other for Testing our trained model’s performance.</h3>",
                            unsafe_allow_html=True)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                        # concatenate our training data back together
                        X = pd.concat([X_train, y_train], axis=1)
                        st.write("Concatenate our training data back together")
                        st.write(X.head())
                    st.markdown("<h3 style='text-align: left; color: black;'> Click here for building a Random Forest Model For Unbalanced Data</h3>",unsafe_allow_html=True)
                    if st.checkbox("Building a Random Forest Model For Unbalanced Data"):
                        # random forest model creation
                        rfc = RandomForestClassifier().fit(X_train, y_train)
                        # predictions
                        rfc_pred = rfc.predict(X_test)
                        st.markdown("<h3 style='text-align: left; color: black;'>The model used is Random Forest classifier</h3>", unsafe_allow_html=True)
                        # Checking unique labels
                        st.write('Unique predicted labels:', np.unique(rfc_pred))
                        # checking accuracy
                        st.write('Accuracy(Test score):', accuracy_score(y_test, rfc_pred))
                        fraud = df[df['Class'] == 1]
                        valid = df[df['Class'] == 0]
                        n_outliers = len(fraud)
                        n_errors = (rfc_pred != y_test).sum()
                        acc = accuracy_score(y_test, rfc_pred)
                        st.write("The accuracy is {}".format(acc))
                        prec = precision_score(y_test, rfc_pred)
                        st.write("The precision is {}".format(prec))
                        rec = recall_score(y_test, rfc_pred)
                        st.write("The recall is {}".format(rec))
                        f1 = f1_score(y_test, rfc_pred)
                        st.write("The F1-Score is {}".format(f1))
                        MCC = matthews_corrcoef(y_test, rfc_pred)
                        st.write("The Matthews correlation coefficient is{}".format(MCC))
                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for Confusion Matrix Of RandomForestClassifierr</h3>",unsafe_allow_html=True)
                    if st.button("Confusion Matrix Of RandomForestClassifier"):
                        # printing the confusion matrix
                        LABELS = ['Normal', 'Fraud']
                        conf_matrix = confusion_matrix(y_test, rfc_pred)
                        fig ,ax = plt.subplots(figsize=(12, 12))
                        st.write(sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d",ax=ax))
                        st.write(plt.title("Confusion matrix"))
                        st.write(plt.ylabel('True class'))
                        st.write(plt.xlabel('Predicted class'))
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                        st.success("You have successfully bulit  Confusion Matrix Of RandomForestClassifier")
                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for Dummy Classifier to verify the inbalance data output </h3>",unsafe_allow_html=True)
                    if st.checkbox("Dummy Classifier to verify the inbalance data output"):
                        dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
                        dummy_pred = dummy.predict(X_test)
                        # Checking unique labels
                        st.write('Unique predicted labels:', np.unique(dummy_pred))
                        st.write('Test score:', accuracy_score(y_test, dummy_pred))
                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for All kinds of evaluating parameters for dummyclassfier  </h3>",unsafe_allow_html=True)
                    if st.button("All kinds of evaluating parameters"):
                        st.write("The model used is Dummy classifier")
                        acc1 = accuracy_score(y_test, dummy_pred)
                        st.write("The accuracy is {}".format(acc1))
                        prec1 = precision_score(y_test, dummy_pred)
                        st.write("The precision is {}".format(prec1))
                        rec1 = recall_score(y_test, dummy_pred)
                        st.write("The recall is {}".format(rec1))
                        f2 = f1_score(y_test, dummy_pred)
                        st.write("The F1-Score is {}".format(f2))
                        MCC1 = matthews_corrcoef(y_test, dummy_pred)
                        st.write("The Matthews correlation coefficient is{}".format(MCC1))
                        st.success("You have successfully evaluate all the parameters of dummyclassifier")
                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for Confusion Matrix Of DummyClassifier </h3>",unsafe_allow_html=True)
                    if st.button("Confusion Matrix Of DummyClassifier"):
                        # printing the confusion matrix
                        LABELS = ['Normal', 'Fraud']
                        conf_matrix = confusion_matrix(y_test, dummy_pred)
                        fig ,ax = plt.subplots(figsize=(12, 12))
                        st.write(sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", ax=ax))
                        st.write(plt.title("Confusion matrix"))
                        st.write(plt.ylabel('True class'))
                        st.write(plt.xlabel('Predicted class'))
                        st.pyplot()
                        st.success("You have successfully bulit  Confusion Matrix of DummyClassifier  ")

                    st.markdown("<h3 style='text-align: left; color: black;'>Upsampling</h3>",unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for separate minority and majority classes(Upsampling)</h3>",unsafe_allow_html=True)
                    if st.button("separate minority and majority classes(Upsampling)"):
                        not_fraud = X[X.Class == 0]
                        fraud = X[X.Class == 1]
                        st.write(not_fraud)
                        st.write(fraud)
                        # upsample minority
                        fraud_upsampled = resample(fraud,
                                                   replace=True,  # sample with replacement
                                                   n_samples=len(not_fraud),  # match number in majority class
                                                   random_state=27)  # reproducible results
                        # Combine majority and upsampled minority
                        upsampled = pd.concat([not_fraud, fraud_upsampled])
                        # Check new Class counts
                        st.write(upsampled.Class.value_counts())
                    st.markdown("<h3 style='text-align: left; color: black;'>Click here for Logistic Regression with the Balanced Dataset</h3>",unsafe_allow_html=True)
                    if st.checkbox("Logistic Regression with the Balanced Dataset"):
                        y_train = upsampled.Class
                        X_train = upsampled.drop('Class', axis=1)
                        from sklearn.linear_model import LogisticRegression
                        upsampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)
                        upsampled_pred = upsampled.predict(X_test)
                        st.markdown(
                            "<h3 style='text-align: left; color: black;'>Accuracy of Logistic Regression</h3>",
                            unsafe_allow_html=True)

                        st.write(accuracy_score(y_test, upsampled_pred))
                        st.markdown("<h3 style='text-align: left; color: black;'>Logistic Regression Classification Report", unsafe_allow_html=True)
                        print(classification_report(y_test, upsampled_pred))
                        st.markdown("<h3 style='text-align: left; color: black;'>Logistic Regression of Confusion Matrix", unsafe_allow_html=True)
                        # confusion matrix
                        pd.DataFrame(confusion_matrix(y_test, upsampled_pred))
                    st.markdown("<h2 style='text-align: left; color: black;'>Downsampling</h2>",unsafe_allow_html=True)
                    if st.checkbox("Downsampling"):
                        not_fraud = X[X.Class == 0]
                        fraud = X[X.Class == 1]
                    # downsample majority
                    not_fraud_downsampled = resample(not_fraud,
                                                     replace=False,  # sample without replacement
                                                     n_samples=len(fraud),  # match minority n
                                                     random_state=27)  # reproducible results
                    # combine minority and downsampled majority
                    downsampled = pd.concat([not_fraud_downsampled, fraud])
                    # checking counts
                    st.write(downsampled.Class.value_counts())
                    st.markdown("<h2 style='text-align: left; color: black;'>Downsampling</h2>",unsafe_allow_html=True)
                    if st.checkbox("Downsampling"):
                        not_fraud = X[X.Class == 0]
                        fraud = X[X.Class == 1]

                    # Trying Logistic Regression with the Balanced Dataset
                    from sklearn.linear_model import LogisticRegression

                    y_train = downsampled.Class
                    X_train = downsampled.drop('Class', axis=1)

                    undersampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

                    undersampled_pred = undersampled.predict(X_test)
                    st.write(accuracy_score(y_test, undersampled_pred))










                    st.write("Click Here For Show the Credit Card Dataset Columns ")
                    if st.checkbox('Select Multiple Columns'):
                        new_data = st.multiselect("Select your perferred columns(Note:Let Your Target Variable be the "
                                                  "column to be select)", df.columns)
                        df1 = df[new_data]
                        st.dataframe(df1)
                        st.success("No Of Multiple Columns in the Given Dataset ")
                        # Dividing my data into X and Y variables

                        X = df1.iloc[:, 0:-1]
                        y = df1.iloc[:, -1]
                    seed = st.slider("Seed", 1, 200)
                    classifier_name = st.selectbox('Select your Perferred Classifier:',
                                                   ('Support Vector Machine', 'KNeighborsClassifier',
                                                    'LogisticRegression',
                                                    'naive_bays', 'decision tree'))
                    def add_parameter(name_of_clf):
                        params = dict()
                        if name_of_clf == 'Support Vector Machine':
                            C = st.slider('C', 0.01, 15.0)
                            params['C'] = C
                        if name_of_clf == 'KNeighborsClassifier':
                            K = st.slider('K', 1, 15)
                            params['K'] = K
                        if name_of_clf == 'LogisticRegression':
                            L = st.slider('L', 1, 15)
                            params['L'] = L
                        if name_of_clf == 'naive_bays':
                            N = st.slider('N', 1, 15)
                            params['N'] = N
                        if name_of_clf == 'decision tree':
                            D = st.slider('D', 1, 15)
                            params['D'] = D
                            return params
                    params = add_parameter(classifier_name)

                    def get_classifier(name_of_clf, params):
                        clf = None
                        if name_of_clf == 'Support Vector Machine':
                            clf = SVC(C=params['C'])
                        elif name_of_clf == 'KNeighborsClassifier':
                            clf = KNeighborsClassifier(n_neighbors=params['K'])
                        elif name_of_clf == 'LR':
                            clf = LogisticRegression(L=params['L'])
                        elif name_of_clf == 'naive_bays':
                            clf = MultinomialNB(N=params['N'])
                        elif name_of_clf == 'decision tree':
                            clf = RandomForestClassifier(D=params['D'])
                        else:
                            st.warning('Select your choice of algorithms')

                            return clf
                        clf=get_classifier(classifier_name, params)

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
                        clf.fit(X_train, y_train)

                        y_pred =clf.predict(X_test)
                        st.write('Predications :', y_pred)

                        accuracy = accuracy_score(y_test, y_pred)
                        st.write('Name of Classifier:', classifier_name)
                        st.write('Accuracy of Given Classifier:', accuracy)
                elif task == "Profiles":
                    #st.subheader("User Profiles")
                    st.markdown("<h1 style='text-align: left; color: black;'>User Profiles</h1>", unsafe_allow_html=True)
                    user_result = view_all_users()
                    clean_db = pd.DataFrame(user_result, columns=["Username", "Password"])
                    st.dataframe(clean_db)
            else:
                st.warning("Incorrect Username/Password")
    elif choice == "SignUp":
        #st.subheader("Create New Account")
        st.markdown("<h1 style='text-align: left; color: black;'> Create New Account </h1>", unsafe_allow_html=True)

        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user, make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")


if __name__ == '__main__':
    main()
