# Depression & Anxiety Detection Using Multimodal Data 

This repository supports the paper: > **A Multimodal Sensor Fusion Approach Using Mobile, Wearable, and
IoT Sensors for Mental Health Detection**

We present a machine learning and deep learning framework that uses multimodal data collected from **mobile**, **wearable deivices**, and **home IoT sensors** to detect signs of **depression and anxiety** in real-world environments.

---

## 🚀 Getting Started

We provide extracted and preprocessed features (in ML models/FEATURES or DL models/FEATURES) to support reproducibility.

🔒 The raw data (from mobile, wearable, and IoT sensors) is not publicly released due to privacy concerns.

## 📋 Data Collection & Analysis

### 🏠 Data Collected from Aqara Sensors
| **Sensor Name**                        | **Data Field**                       |
| -------------------------------------- | ------------------------------------ |
| Aqara Vibration Sensor (DJT11LM)       | 1: Vibrate <br> 2: Tilt <br> 3: Drop |
| Aqara Door & Window Sensor (MCCGQ11LM) | 0: Closed <br> 1: Open               |
| Aqara Motion Sensor (RTCGQ11LM)        | 0: Unoccupied <br> 1: Occupied       |
| Aqara Smart Plug (SP-EUC01)            | Cost energy (Wh)                     |


### 🛏️ Data Collected from Withings Sleep Sensor
| **Data Field**       | **Description**                                                          |
| -------------------- | ------------------------------------------------------------------------ |
| wake up duration     | Time spent awake (in seconds)                                            |
| wake up count        | Number of times the user woke up in bed (not counting out-of-bed events) |
| rem sleep duration   | Duration in REM sleep (in seconds)                                       |
| total time in bed    | Total time spent in bed                                                  |
| total sleep time     | Sum of light, deep, and REM durations                                    |
| sleep efficiency     | Ratio of total sleep time to time in bed                                 |
| sleep latency        | Time spent in bed before falling asleep                                  |
| wake up latency      | Time spent in bed after waking up                                        |
| waso                 | Time awake in bed after initially falling asleep                         |
| nb rem episodes      | Number of REM phases during sleep                                        |
| out of bed count     | Number of times user got out of bed                                      |
| light sleep duration | Duration of light sleep (in seconds)                                     |
| deep sleep duration  | Duration of deep sleep (in seconds)                                      |



### 🎤 Sensor Data Collected from Smart Speaker System
| **Sensor Type**                  | **Data Field**                                        | **Sampling Rate** |
| -------------------------------- | ----------------------------------------------------- | ----------------- |
| Smartphone - Camera              | Number of people                                      | 1 Hz              |
| Smartphone - Microphone          | Noise (dB)                                            | 1 Hz              |
| Smartphone - Light sensor        | Light (lx)                                            | 1 Hz              |
| Smartphone - Microphone (Survey) | User voice                                            | Per survey        |
| Air Quality Sensor (BSP02AIQ)    | Temperature (°C), Humidity (%), CO₂ (ppm), TVOC (ppb) | 1 Hz              |



### 🗣️ Self-Reported Data Collected from Smart Speaker
| **Category**       | **Questions**                                                                                                                                                              | **Answers**                          |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| Depression (PHQ-2) | 1. Over the past 1–2 hours, how often have you been bothered by little interest or pleasure in doing things? <br> 2. How often have you felt down, depressed, or hopeless? | Not at all (0) – Very frequently (3) |
| Anxiety (GAD-2)    | 3. Over the past 1–2 hours, how often have you felt nervous, anxious, or on edge? <br> 4. How often have you been unable to stop or control worrying?                      | Not at all (0) – Very frequently (3) |



### 📱 Mobile Data - Preprocessing 
| **Type**               | **Raw Data**  | **Preprocessing**                                                                                                         |
| ---------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Social interaction** | Call event    | Filter negative and zero durations                                                                                        |
|                        | Message event | Encode the categorical events to 1 as numeric values                                                                      |
| **Physical activity**  | Accelerometer | Calculate magnitude from X, Y, Z axes                                                                                     |
| **Context**            | Location      | Haversine distance between GPS points; POI-based clustering; assign semantic labels (e.g., home, work, Google Map labels) |
| **Phone usage**        | App usage     | Recategorize into predefined types; calculate duration per session                                                        |
|                        | Installed app | Calculate Jaccard similarity between consecutively installed app sets                                                     |
|                        | Screen event  | Calculate duration of screen-on sessions per user                                                                         |
|                        | WiFi events   | Compute cosine, Euclidean, Manhattan distance between RSSI values; Jaccard similarity between consecutive BSSID sets      |
|                        | Media events  | EEncode the categorical video, image, and all types events to 1 as numeric values                                         |


### 📱Mobile - Extracted Features
| **Type**               | **Raw Data**         | **Information Being Aggregated**                            | **Features**                                                         |
| ---------------------- | -------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------- |
| **Social interaction** | Call event           | Call duration and previous times contacted                  | Numeric features, call frequency                                     |
|                        | Message event        | Messages sent, received, and total message events           | Numeric features                                                     |
| **Physical activity**  | Accel.               | X, Y, Z values and magnitude                                | Numeric features                                                     |
|                        | Activity transition  | ENTER\_WALKING, ENTER\_STILL, ENTER\_IN\_VEHICLE, etc.      | Categorical features                                                 |
|                        | Activity event       | Confidence levels: OnFoot, InVehicle, Tilting, etc.         | Numeric features                                                     |
| **Context**            | Location             | Distance traveled, location clusters, and semantic labels   | Numeric features for distance; categorical for clusters and labels   |
| **Phone usage**        | App usage            | App usage types and durations                               | Categorical features (app events), numeric features (usage duration) |
|                        | Installed app        | Jaccard similarity index of consecutively installed apps    | Numeric features                                                     |
|                        | Screen event         | Screen on/off events and duration                           | Categorical (event type), numeric (duration)                         |
|                        | OnOffEvent           | Phone power on/off                                          | Categorical features                                                 |
|                        | Network connectivity | Network connected events                                    | Categorical features                                                 |
|                        | Battery event        | Battery level, temperature, status                          | Numeric (level, temp); categorical (status)                          |
|                        | Data traffic         | Mobile data usage in kB                                     | Numeric features                                                     |
|                        | WiFi events          | Distances between RSSI values; Jaccard similarity of BSSIDs | Numeric features                                                     |
|                        | Media events         | Event counts: image, video, all                             | Numeric features                                                     |
|                        | System events        | Ringer mode, power saving state, charging events            | Categorical features                                                 |


## 🤖 Model Training

### Traditional ML Models
* Location: ML_models/
* Models: Decision Tree, Random Forest, AdaBoost, XGBoost, LDA, kNN, SVM

```bash
cd ML_models/
python main.py
```
More details in ML_models/README.md

### Deep Learning Models
Location: DL_models/
Models: 1D-CNN, attention-based fusion

```bash
cd DL_models/
python main.py
```
More details in DL_models/README.md

## 📊 Evaluation
Includes:
* Accuracy, AUROC
* Generalized vs Personalized model comparison
* Top behavioral features analysis

##  📌 Notes
* Extracted features (after preprocessing) are shared.
* Raw sensor data is not included due to participant privacy constraints.
* The full pipeline can still be run end-to-end using the provided features.

## 📄 Citation
To be added upon publication.