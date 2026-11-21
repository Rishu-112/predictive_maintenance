"""Streamlit viewer for saved model results, not a real-time industrial system."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Turbofan RUL Lab',page_icon='🔧',layout='wide')
st.title('🔧 Turbofan Predictive Maintenance')
st.caption('NASA C-MAPSS FD001 | Remaining Useful Life (RUL) in engine cycles')
metrics=Path('reports/metrics.json')
predictions=Path('reports/test_predictions.csv')
if not metrics.exists() or not predictions.exists():
    st.warning('Train first: python -m src.train (or --demo for artificial data).')
    st.stop()
data=json.loads(metrics.read_text())
df=pd.read_csv(predictions)
if 'SYNTHETIC' in data['dataset']:
    st.error('SYNTHETIC DEMO — these are NOT real NASA benchmark scores.')
st.write('Dataset:',data['dataset'],' | Selected model:',data['chosen_model'])
mae,rmse,r2=st.columns(3)
mae.metric('Test MAE',data['official_test_last_cycle']['MAE'])
rmse.metric('Test RMSE',data['official_test_last_cycle']['RMSE'])
r2.metric('Test R²',data['official_test_last_cycle']['R2'])
st.subheader('Actual vs predicted RUL')
st.scatter_chart(df,x='actual_rul',y='predicted_rul')
engine=st.selectbox('Inspect an engine',sorted(df.engine_id.unique()))
row=df[df.engine_id==engine].iloc[0]
st.write(f"Engine {engine}: observed cycle {int(row.observed_cycle)} · predicted {row.predicted_rul:.1f} cycles · actual {int(row.actual_rul)} cycles")
st.subheader('Test predictions')
st.dataframe(df,use_container_width=True)
st.download_button('Export predictions as CSV',df.to_csv(index=False),'rul_predictions.csv')
st.info('Educational demonstration. Simulated engines; model estimates are not maintenance or safety instructions.')
