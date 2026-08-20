import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# (We can use mean, max etc to fill in missing values that were identified as outliers
# But we have a cool pandas feature called "interpolate" that does that for us)
# --------------------------------------------------------------

for col in predictor_columns:
    df[col] = df[col].interpolate()
    
df.info() 

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df["set"]==50]["acc_y"].plot()
df[df["set"]==25]["acc_y"].plot()

duration = df[df["set"]==1].index[-1] - df[df["set"]==1].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    
    duration = stop -start
    df.loc[(df["set"] == s), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] /5
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --This smoothens out the noise to enhance the ML model (Plot to see the diff.)
# __you can chek out the LowPassFilter class to see the functions and parameters
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000 / 200 # check the LowPassFilter class to see what fs means
cutoff = 1.3    # This cuts off outliers beyond 1.3. We don't want it to be too smooth. It might result to overfitting

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="row data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
# --------------------------------------------------------------
# Principal component analysis PCA
# --PCA reduces the number of features by combining features to produce another that manages
# _to capture most of the information from the features that were combined 
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

## using the elbow method to check for the optimal number of PCA
plt.figure(figsize=(8, 8))
plt.plot(range(1, len(predictor_columns) +1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)
subset = df_pca[df_lowpass["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
#--calculating the magnitudes r (data direction)for the 3 accelerometer data and 3 gyroscope data
#__will cover the information without bias and it can handle re-orientation without issues
## r = sqrt(x^2 + y^2 + z^2)
# --------------------------------------------------------------
df_squared = df_pca.copy()
acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
## next we instantiate the NumericalAbstraction class (ctrl+click to see its features above)
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

## window size = ws (it is a method of trial and error)
ws = int(1000/200) ## to get 1 second of data we would sum five rows

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")
## The loop above will cause data from other sets to be captured in another, which introduces noise
## To correct it we would loop over each set and find the mean and std and concat 
df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy() # to apply it on unique set so that we don't have an overflow
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset  = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)
    
df_temporal = pd.concat(df_temporal_list)

##__quick visualization to see what we have 
df_temporal[df_temporal["set"]==75][["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
df_temporal[df_temporal["set"]==75][["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index() ## we reset the index bcos the function is expecting discrete values
FreqAbs = FourierTransformation() ##__check the class to see its propoerties

fs = int(1000/200)  ## number of samples per second
ws = int(2800/200)  ##--window size is avg length of a repetition

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

## Visualization results
subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[[
    "acc_y_max_freq",
    "acc_y_freq_weighted",
    "acc_y_pse",
    "acc_y_freq_1.429_Hz_ws_14",
    "acc_y_freq_2.5_Hz_ws_14"
    
]].plot()

##create the loop
df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformation to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()

## there is a high correlation between the data and this can cause overfitting
## since we have enough data, we get rid of 50% of the data by skipping 2 steps per row
df_freq = df_freq.iloc[::2] ## it will take rows stepwise ( 2 steps)

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state = 0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_) ## if you check the sklearn library for kmeans, you'll see why we can call kmeans.inertia_
    
### visualize to see the elbow and know the optimal number of centroids k
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("Number of centroids (k)")
plt.ylabel("sum of squared distances")
plt.show()

## seeing from the visualization, 5 would arquably be the optimal amount of centroids
## now we can choose k = 5
kmeans = KMeans(n_clusters = 5, n_init=20, random_state = 0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

## my take from the 2 plots above is that the number of centroids could be 6 to see if it can group all six unique exercises
## using 5 made it group two exercise as one
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")