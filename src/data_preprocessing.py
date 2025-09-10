import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Load the dataset:
df = pd.read_csv('data\weather_prediction_dataset.csv')
df_bbq = pd.read_csv('data\weather_prediction_bbq_labels.csv')


# Data Cleaning & Preprocessing:
# - Handle missing values
# - Convert datatypes (like datetime)
# - Feature scaling/encoding if needed

# We can see there's no missing or null data in the dataset
df.isna().sum()

df['DATE'] = pd.to_datetime(df['DATE'], format="%Y%m%d")

required_columns = ['DATE', 'MONTH', 'OSLO_cloud_cover', 'OSLO_wind_speed', 'OSLO_wind_gust', 'OSLO_humidity', 'OSLO_pressure', 'OSLO_global_radiation', 'OSLO_precipitation', 'OSLO_sunshine', 'OSLO_temp_mean', 'OSLO_temp_min', 'OSLO_temp_max']

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print("Missing columns:", missing_columns)
else:
    df_OSLO = df[required_columns]


# Adding target column from another dataset
df_OSLO['BBQ'] = df_bbq['OSLO_BBQ_weather']


# Exploratory Data Analysis (EDA):
# - Correlation heatmap
# - Distribution plots
# - Trends over time

fig, axs = plt.subplots(2,2, figsize = (10,10))
sns.kdeplot(data = df_OSLO, x='OSLO_cloud_cover', hue = 'BBQ', fill = True, ax = axs[0,0])
axs[0,0].set_title('Cloud cover condition for BBQ')
sns.kdeplot(data = df_OSLO, x='OSLO_wind_speed', hue = 'BBQ', fill = True, ax = axs[0,1])
axs[0,1].set_title('Wind speed condition for BBQ')
sns.kdeplot(data = df_OSLO, x='OSLO_wind_gust', hue = 'BBQ', fill = True, ax = axs[1,0])
axs[1,0].set_title('Wind gust condition for BBQ')
sns.kdeplot(data = df_OSLO, x='OSLO_humidity', hue = 'BBQ', fill = True, ax = axs[1,1])
axs[1,1].set_title('Humidity condition for BBQ')
plt.tight_layout()

plt.show()


fig, axs = plt.subplots(2,2, figsize = (10,10))
sns.kdeplot(data = df_OSLO, x='OSLO_sunshine', hue = 'BBQ', fill = True, ax = axs[0,0])
axs[0,0].set_title('Sunshine condition for BBQ')
sns.kdeplot(data = df_OSLO, x='OSLO_temp_mean', hue = 'BBQ', fill = True, ax = axs[0,1])
axs[0,1].set_title('Average temperature condition for BBQ')
sns.kdeplot(data = df_OSLO, x='OSLO_temp_min', hue = 'BBQ', fill = True, ax = axs[1,0])
axs[1,0].set_title('Minimum temperature condition for BBQ')
sns.kdeplot(data = df_OSLO, x='OSLO_temp_max', hue = 'BBQ', fill = True, ax = axs[1,1])
axs[1,1].set_title('Maximum temperature condition for BBQ')
plt.tight_layout()

plt.show()


fig, axs = plt.subplots(2,2, figsize = (10,10))
sns.kdeplot(data = df_OSLO, x='OSLO_pressure', hue = 'BBQ', fill = True, ax = axs[0,0])
axs[0,0].set_title('Pressure condition for BBQ')
sns.kdeplot(data = df_OSLO, x='OSLO_global_radiation', hue = 'BBQ', fill = True, ax = axs[0,1])
axs[0,1].set_title('Global radiation condition for BBQ')
sns.kdeplot(data = df_OSLO, x='OSLO_precipitation', hue = 'BBQ', fill = True, ax = axs[1,0])
axs[1,0].set_title('Precipitation condition for BBQ')
plt.tight_layout()

plt.show()


# Feature Engineering:
# - Extract useful features (month, season, etc.)

# Add Day series
df['Day'] = df['DATE'].dt.day

# Season mapping
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['Season'] = df['MONTH'].apply(get_season)
df_OSLO['Season'] = df_OSLO['MONTH'].apply(get_season)


# Correlation Heatmap for OSLO specific
plt.figure(figsize=(12,8))
sns.heatmap(df_OSLO.select_dtypes(include=['number']).corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap of Weather Features (OSLO)")
plt.show()

X = df_OSLO.drop(['DATE', 'BBQ'], axis = 1)
y = df_OSLO['BBQ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)