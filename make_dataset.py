import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")


single_file_gyr = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")



# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
len(files)


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path = "../../data/raw/MetaMotion\\"
f = files[0]
   ## f.split("-")  #--> it splits or seperate the items in the string based on points with - inbetween.
Participant = f.split("-")[0].replace(data_path,"")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("12345").rstrip("_MetaWear_2019") 
   ## you can simply use rstrip("2") or rstrip("123") to remove the digit 2 but not rstrip("1")
   
df = pd.read_csv(f) ## converting the file to a dataframe by reading the csv file of the first file

  ## the code below assigns the following three columns below to the data frame during participant's A bench exercise with heavy metals I think
df["Participant"] = Participant
df["label"] = label
df["category"] = category



# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    Participant = f.split("-")[0].replace(data_path,"")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019") 
    
    df = pd.read_csv(f)
    
    df["Participant"] = Participant
    df["label"] = label
    df["category"] = category
    
    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])
        
    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1       
        gyr_df = pd.concat([gyr_df, df])
        
        

    
    

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.info()

## ___ The code that follows converts unix time to readable human time by using the method to_datetime

pd.to_datetime(df["epoch (ms)"], unit= "ms")

##___ The codes that follow use the converted unix time (epoch (ms)) as index instead of the default 0,1,2 ...

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit = "ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit = "ms")

##__ Next we clean the table by deleting the other columns with time

del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]





# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")

def read_data_from_files(files):
    
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        Participant = f.split("-")[0].replace(data_path,"")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019") 
        
        df = pd.read_csv(f)
        
        df["Participant"] = Participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
            
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1       
            gyr_df = pd.concat([gyr_df, df])
            
        pd.to_datetime(df["epoch (ms)"], unit= "ms")

    ##___ The codes that follow use the converted unix time (epoch (ms)) as index instead of the default 0,1,2 ...

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit = "ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit = "ms")

    ##__ Next we clean the table by deleting the other columns with time

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]   
    
    return acc_df, gyr_df

acc_df, gyr_df = read_data_from_files(files)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis = 1)

##-- The "axis" parameter above indicates we want to join the table column wise. If axis=0 it means row-wise
##-- The iloc is to choose the first 3 columns in the dataframe. check out more resources for explanation online

## -- you can run "data_merged.dropna()" to see the clean table without missing values
##-- the code that follows change the name of the columns for convenience 
data_merged.columns = [ "acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","participant","label","category", "set"]



# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

data_merged.head(1000)

##-- make a dictionary to apply on the merged data to resample it

sampling = {
    "acc_x": "mean", 
    "acc_y": "mean", 
    "acc_z": "mean",
    "gyr_x": "mean", 
    "gyr_y": "mean", 
    "gyr_z": "mean",  
    "participant": "last", 
    "label": "last", 
    "category": "last", 
    "set": "last"}

##-- the sampling with "mean" finds average of a particular defined set of data, in this case after the interval of 200ms
##-- while "last" tells it to take the last vale of the last calculated average and assign to the new row (average calculated)
##-- the statement seems vague. What about if we have participants A, A, A, and B, does it mean it will take B while
##-- most of the data is from participant A?

resampling_data = data_merged[:1000].resample(rule="200ms").apply(sampling)

# Split by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])


data_resampled["set"] = data_resampled["set"].astype("int")
data_resampled.info()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")

 ##-- you can export to excel or csv, your choice. Here pickle file was chosen.