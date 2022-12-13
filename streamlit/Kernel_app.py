import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib


st.set_page_config(
    page_title="GPU Kernel Performance",
    page_icon="🐸",
    layout="wide",
)

st.title('GPU Kernel Performance')

@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['Data Description', 'Prediction'])
data=pd.read_csv('sgemm_product.csv')

def detect_outliers_zscore(data):
    outliers = []
    thres = 3
    index = 0
    mean = np.mean(data)
    std = np.std(data)
    # print(mean, std)
    for i in data:
        index = index + 1
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            outliers.append(index)
    return outliers# Driver code


#################################################################################################
datatf = data.copy()
datatf['Runtime'] = datatf[['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)']].mean(axis=1)
datatf = datatf.drop(['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'], axis = 1)
sample_outliers_Runtime = detect_outliers_zscore(datatf['Runtime'])
datatf = datatf.drop(sample_outliers_Runtime)

one_hot_encoded_data = pd.get_dummies(datatf, columns = ['SA', 'SB', 'STRM', 'STRN'])
datatf = one_hot_encoded_data.copy()
prework = datatf['NWG']*datatf['MWG']
datatf['prework'] = prework
datatf["MWI"] = datatf["MWG"]*datatf["MDIMC"]
datatf['NWI'] = datatf["NWG"]*datatf["NDIMC"]
datatf['Target']=np.log(datatf.Runtime)
datatf = datatf.drop(columns=["Runtime"])
#################################################################################################

if app_mode=='Data Description':
    st.image("imatges/tune.png")

    st.markdown("#### Attribute Information:")
    st.markdown("Independent variables:")
    st.markdown("1. `MWG`: per-matrix 2D tiling at workgroup level: {16, 32, 64, 128} (integer)")
    st.markdown("2. `NWG`: per-matrix 2D tiling at workgroup level: {16, 32, 64, 128} (integer)")
    st.markdown("3. `KWG`: inner dimension of 2D tiling at workgroup level: {16, 32} (integer)")
    st.markdown("4. `MDIMC`: local workgroup size also defines the amount of work-per-thread in M dimensions: {8, 16, 32} (integer)")
    st.markdown("5. `NDIMC`: local workgroup size also defines the amount of work-per-thread in N dimensions: {8, 16, 32} (integer)")
    st.markdown("6. `MDIMA`: local memory shape: {8, 16, 32} (integer)")
    st.markdown("7. `NDIMB`: local memory shape: {8, 16, 32} (integer)")
    st.markdown("8. `KWI`: kernel loop unrolling factor: {2, 8} (integer)")
    st.markdown("9. `VWM`: per-matrix vector widths for loading and storing: {1, 2, 4, 8} (integer)")
    st.markdown("10. `VWN`: per-matrix vector widths for loading and storing: {1, 2, 4, 8} (integer)")
    st.markdown("11. `STRM`: enable stride for accessing off-chip memory within a single thread: {0, 1} (categorical)")
    st.markdown("12. `STRN`: enable stride for accessing off-chip memory within a single thread: {0, 1} (categorical)")
    st.markdown("13. `SA`: per-matrix manual caching of the 2D workgroup tile: {0, 1} (categorical)")
    st.markdown("14. `SB`: per-matrix manual caching of the 2D workgroup tile: {0, 1} (categorical)")
    st.markdown("Output:")
    st.markdown("15-18. `Run1`, `Run2`, `Run3`, `Run4`: performance times in milliseconds for 4 independent runs using the same parameters. They range between 13.25 and 3397.08.")

    
    st.markdown('Aquest conjunt de dades mesura el temps dexecució dun producte matriu-matriu A · B = C, on totes les matrius tenen una mida de 2048 x 2048, utilitzant un nucli de GPU SGEMM parametrizable amb 241600 combinacions de paràmetres possibles. ')
    st.markdown('Per a cada combinació provada, es van realitzar 4 execucions i els seus resultats es presenten com a les 4 darreres columnes. Tots els temps es mesuren en mil·lisegons. L,experiment es va executar en una estació de treball descriptori amb Ubuntu 16.04 Linux amb un Intel Core i5 (3,5 GHz),16 GB de RAM i una GPU NVidia Geforce GTX 680 4 GB GF580 GTX-1,5 GB.')


    st.markdown('## Dataset :')
    st.write(data.head())
    st.write("#### Nombre de files i columnes")
    st.write(f'Files: {data.shape[0]}')
    st.write(f'Columnes: {data.shape[1]}')



    desfase = (0.1, 0, 0, 0)
    colores = ["#EE6055","#60D394","#AAF683","#FFD97D","#FF9B85"]



    data['Runtime'] = data[['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)']].mean(axis=1)
    data = data.drop(['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'], axis = 1)

    st.sidebar.write('Modificar per visualitzar correctament les gràfiques')
    width = st.sidebar.slider("plot width", 1, 25, 3)
    height = st.sidebar.slider("plot height", 1, 25, 1)
    
    datatf = data.copy()
    sample_outliers_Runtime = detect_outliers_zscore(datatf['Runtime'])
    datatf.drop(sample_outliers_Runtime , inplace = True)


    one_hot_encoded_data = pd.get_dummies(datatf, columns = ['SA', 'SB','STRM', 'STRN'])
    datatf = one_hot_encoded_data.copy()
    prework = datatf['NWG']*datatf['MWG']
    datatf['prework'] = prework
    datatf["MWI"] = datatf["MWG"]*datatf["MDIMC"]
    datatf['NWI'] = datatf["NWG"]*datatf["NDIMC"]

    datatf['Target']=np.log(datatf.Runtime)
    datatf = datatf.drop(columns=["Runtime"])


    st.markdown('### Visualització de les dades:')


    st.write("El nostre objectiu es aconseguir predir el temps d'execució total. Es a dir, la suma de les variables Run. Per tant l'anàlisis realitzat a continuació s'ha fet ja amb la variable Runtime, que equival a la suma de les altres")

      # Select columns to display
    if st.checkbox("Mostrar columnes concretes del dataset"):
        # get the list of columns
        columns = data.columns.tolist()
        st.write("#### Selecciona la columna a visualitzar:")
        selected_cols = st.multiselect("", columns)
        if len(selected_cols) > 0:
            selected_df = data[selected_cols]
            st.dataframe(selected_df)

    if st.checkbox("Correlació de dades"):
        st.write("Heatmap")
        fig, ax = plt.subplots(figsize=(width,height))
        st.write(sns.heatmap(data.corr(), annot=True,linewidths=.5,fmt='.4f',ax=ax, cmap="crest"))
        st.pyplot(fig)

    if st.checkbox("Distribucions de les dades"):
        st.write("#### Selecciona la columna a visualitzar: ")
        columns = data.columns.tolist()
        class_name = "Runtime"
        column_name = st.selectbox("",columns)
        if st.button("Genera"):
            fig, ax = plt.subplots(figsize=(width,height))
            st.write(sns.distplot(data[column_name], kde=True))
            st.pyplot(fig)
        

    if st.checkbox("Proporció de cada valor per variable"):
        st.write("#### Selecciona la columna a visualitzar: ")
        columns = data.columns.tolist()
        columns.remove("Runtime")
        column_name = st.selectbox("",columns)
        if st.button("Genera"):
            fig, ax = plt.subplots(figsize=(width,height))
            plt.pie(data[column_name].value_counts(), labels= data[column_name].unique(), colors=colores, autopct="%0.1f %%", shadow=True, startangle=140)
            st.pyplot(fig)

       


    st.markdown('### Processament de dades:')
    st.write("#### Tractament de Outliers")
    datac = data.copy()
    if st.checkbox("Visualització de outliers:"):
        st.markdown("Les uniques variables amb outliers visibles són: `Runtime`, `MDIMC` i `NDIMC` ")
        fig, ax = plt.subplots(figsize=(width,height))
        st.write(sns.boxplot(data=datac, x='Runtime', palette="blend:#7AB,#EDA"))
        st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(width,height))
        st.write(sns.boxplot(data=datac, x='MDIMC', palette="blend:#7AB,#EDA"))
        st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(width,height))
        st.write(sns.boxplot(data=datac, x='NDIMC', palette="blend:#7AB,#EDA"))
        st.pyplot(fig)

        st.markdown('Aplicant un Z-score test veiem que lúnica variable amb outliers reals és la de Runtime, per tant eliminant-los ens queda el següent plot:')
        sample_outliers_Runtime = detect_outliers_zscore(datac['Runtime'])
        datac.drop(sample_outliers_Runtime , inplace = True)
        fig, ax = plt.subplots(figsize=(width,height))
        st.write(sns.boxplot(data=datac, x='Runtime', palette="blend:#7AB,#EDA"))
        st.pyplot(fig)

    sample_outliers_Runtime = detect_outliers_zscore(data['Runtime'])
    data.drop(sample_outliers_Runtime , inplace = True)
    

    st.markdown('#### Transformació de la variable `Runtime`')
    st.markdown('Per arribar a una millor distribució, aplico una transformació de la variable `Runtime`, ja que millora la distribució i l acosta a una de normal, la qual cosa facilitarà l avaluació en els models i ens donarà millors resultats. Com podem veure la distribució de Runtime, probablement segueix una distribució exponencial.')
    if st.checkbox("Visualitzar gràfiques de distribució"):
        datacc = data.copy()
        st.markdown('Distribució inicial de `Runtime`')
        fig, ax = plt.subplots(figsize=(width,height))
        st.write(sns.distplot(datacc['Runtime']))
        st.pyplot(fig)

        datacc['Target']=np.log(datacc.Runtime)

        st.markdown('Distribució de `Runtime` desprès de la transformació logarítmica')
        fig, ax = plt.subplots(figsize=(width,height))
        st.write(sns.distplot(datacc['Target']))
        st.pyplot(fig)

        datacc = datacc.drop(columns=["Runtime"])
        if st.checkbox("Correlació actual de les dades"):
            fig, ax = plt.subplots(figsize=(width,height))
            st.write(sns.heatmap(datacc.corr()[['Target']].sort_values(by='Target', ascending=False), vmin=-1, vmax=1, annot=True, cmap='crest'))
            st.pyplot(fig)

    data['Target']=np.log(data.Runtime)
    data = data.drop(columns=["Runtime"])

    st.markdown('#### Pre processing')
    st.markdown('L objectiu es crear noves variables que ajudin al model a obtenir bons resultats de predicció')
    st.markdown('Per fer-ho s han creat diverses variables relacionades amb les propietats dels processadors:')
    st.markdown('   - La variable `prework` ha estat creada a partir de la multiplicació de `NWG` i `MWG`.')
    st.markdown('   - A més a més s ha afegit les variables WI (workitem). Aquestes s han creat a partor de la multiplicació respectiva de `MWG` i `NWG` amb `MDIMC` i `NDIMC` ')
    st.markdown('Finalment, s ha realitzat un One-Hot encoder de les variables categòriques. El dataset amb el que treballarem per fer la predicció es el següent:')

    
    one_hot_encoded_data = pd.get_dummies(data, columns = ['SA', 'SB','STRM', 'STRN'])
    data = one_hot_encoded_data.copy()
    prework = data['NWG']*data['MWG']
    data['prework'] = prework
    data["MWI"] = data["MWG"]*data["MDIMC"]
    data['NWI'] = data["NWG"]*data["NDIMC"]
    st.write(data.head())
 
elif app_mode == 'Prediction':
    st.markdown('Emplenar les dades amb les que vols treballar:')
    st.sidebar.header("Informació de la GPU :")
    col1, col2 = st.columns(2)
    with col1:
        MWG = st.selectbox("MGW", options=[16, 32, 64, 128])
        NWG = st.selectbox("NGW", options=[16, 32, 64, 128])
        KWG = st.selectbox("KGW", options=[16, 32])
        MDIMC = st.selectbox("MDIMC", options=[8,16, 32])
        NDIMA =st.selectbox("NDIMA", options=[8,16, 32])
        NDIMB = st.selectbox("NDIMB", options=[8,16, 32])
        NDIMC = st.selectbox("NDIMC", options=[8,16, 32])
    with col2:
        KWI = st.selectbox("KWI", options=[2, 8])
        VWM = st.selectbox("VWM", options=[1, 2, 4, 8])
        VWN = st.selectbox("VWN", options=[1, 2, 4, 8])
        STRM = st.selectbox("STRM", options=[1, 0])
        STRN = st.selectbox("STRN", options=[1, 0])
        SA = st.selectbox("SA", options=[1, 0])
        SB =st.selectbox("SB", options=[1, 0])
    
    data1={
    'MWG':MWG,
    'NWG':NWG,
    'KWG':KWG,
    'MDIMC':MDIMC,
    'NDIMC':NDIMC,
    'NDIMA':NDIMA,
    'NDIMB':NDIMB,
    'KWI':KWI,
    'VWM':VWM,
    'VWN':VWN,
    'SA':SA,
    'SB':SB,
    'STRM':STRM,
    'STRN':STRN
    }
    df = pd.DataFrame([data1])
 
    

    if(df['SA'].all() == 0):
        df['SA_0'] = 1
        df['SA_1'] = 0
    else:
        df['SA_0'] = 0
        df['SA_1'] = 1
        
    if(df['SB'].all() == 0):
        df['SB_0'] = 1
        df['SB_1'] = 0
    else:
        df['SB_0'] = 0
        df['SB_1'] = 1
        
    if(df['STRM'].all() == 0):
        df['STRM_0'] = 1
        df['STRM_1'] = 0
    else:
        df['STRM_0'] = 0
        df['STRM_1'] = 1
        
    if(df['STRN'].all() == 0):
        df['STRN_0'] = 1
        df['STRN_1'] = 0
    else:
        df['STRN_0'] = 0
        df['STRN_1'] = 1
    
    df = df.drop(columns=['STRN', 'STRM', 'SA', 'SB'])
    df2 = df.copy()
    df = df.drop(columns = ['SA_1', 'SB_1', 'STRM_1', 'STRN_1'])

    prework = df['NWG']*df['MWG']
    df['prework'] = prework
    df["MWI"] = df["MWG"]*df["MDIMC"]
    df['NWI'] = df["NWG"]*df["NDIMC"]

    prework = df2['NWG']*df2['MWG']
    df2['prework'] = prework
    df2["MWI"] = df2["MWG"]*df2["MDIMC"]
    df2['NWI'] = df2["NWG"]*df2["NDIMC"]
    st.write(df.head())
   
    st.subheader('Regressió: ')
    st.markdown('El model que usarem per predir les dades es un `Random Forest Regressor`')
    st.markdown('Per veure quant sera la duració de l execució del programa amb les dades entrades clica generar')
    if st.button("Genera regressió"): 
        st.markdown('Calculant resultats.....')
        predd=np.array(df.loc[0]).reshape(1,-1)

        regr = joblib.load("regr.pkl")
        pred = regr.predict(predd) #make prediction on test set

        pred=np.exp(pred)
        st.success('Càlcul realitzat amb éxit')
        st.write(pred)


    st.subheader('Classificació: ')
    
    if st.button("Genera classificació"): 
        clas = joblib.load("class.pkl")
        predd=np.array(df2.loc[0]).reshape(1,-1)
        pred = clas.predict(predd) #make prediction on test set
        st.success('Càlcul realitzat amb éxit')
        st.write(pred)
        st.markdown('Si el resultat és 0 vol dir que el teu temps està per sota la mitjana, si és 1 vol dir que està per sobre')
