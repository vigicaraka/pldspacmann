import pandas as pd
import numpy as np
import joblib

params = {'out_path' : '../output/data_concat.pkl',
          'col_use' : ['Patient Age at Treatment',
                     'Total Number of Previous cycles, Both IVF and DI',
                     'Total Number of Previous treatments, Both IVF and DI at clinic',
                     'Total Number of Previous IVF cycles',
                     'Total Number of Previous DI cycles',
                     'Total number of previous pregnancies, Both IVF and DI',
                     'Total number of IVF pregnancies', 'Total number of DI pregnancies',
                     'Total number of live births - conceived through IVF or DI',
                     'Total number of live births - conceived through IVF',
                     'Total number of live births - conceived through DI',
                     'Type of Infertility - Female Primary',
                     'Type of Infertility - Female Secondary',
                     'Type of Infertility - Male Primary',
                     'Type of Infertility - Male Secondary',
                     'Type of Infertility -Couple Primary',
                     'Type of Infertility -Couple Secondary',
                     'Cause  of Infertility - Tubal disease',
                     'Cause of Infertility - Ovulatory Disorder',
                     'Cause of Infertility - Male Factor',
                     'Cause of Infertility - Patient Unexplained',
                     'Cause of Infertility - Endometriosis',
                     'Stimulation used', 'Type of treatment - IVF or DI',
                     'Specific treatment type', 'Live Birth Occurrence',
                     'Sperm From', 'Number of Live Births',
                     'Number of foetal sacs with fetal pulsation'],
         'patient_age_value' : {'18 - 34':0, '35-37':1, '38-39':2, '40-42':3, '43-44':4, '45-50':5},
          'to_int_feature' : ['Patient Age at Treatment', 
                           'Total Number of Previous cycles, Both IVF and DI',
                           'Total Number of Previous treatments, Both IVF and DI at clinic', 
                           'Total Number of Previous IVF cycles',
                           'Total Number of Previous DI cycles',
                           'Total number of previous pregnancies, Both IVF and DI',
                           'Total number of IVF pregnancies',
                           'Total number of live births - conceived through IVF or DI'],
         'to_dummy_feature' : ['Type of treatment - IVF or DI', 
                             'Specific treatment type',
                             'Sperm From']}

def feature_selection(params):
    df = joblib.load(params['out_path'])
    
    # hanya menggunakan column yang didefinisikan
    df = df[params['col_use']]
    
    # drop missing observation
    df = df.dropna()
    
    return df

def feature_engineering(df, params):
    
    # ubah patient age treatment menjadi ordinal value
    df['Patient Age at Treatment'] = df['Patient Age at Treatment'].replace(params['patient_age_value'])
    
    # ubah data dengan value '>=5' menjadi 6
    for col in params['to_int_feature']:
        df[col] = df[col].replace({'>=5':6})
    
    # ubah fitur dengan type object menjadi int
    for col in params['to_int_feature']:
        df[col] = pd.to_numeric(df[col])
        
    # ubah categorical menjadi dummy feature
    for col in params['to_dummy_feature']:
        dum = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dum], axis=1)
        df.drop(col, axis=1, inplace=True)
    
    joblib.dump(df, '../output/data_fe.pkl')    
    
    return df

if __name__ == "__main__":
    df_fs = feature_selection(params)
    feature_engineering(df_fs, params)
