# 🏠 Depression & Anxiety Detection Using Multimodal Data from Mobile, Wearable and Home IoT Sensors

This repository supports the paper: > **A Multimodal Sensor Fusion Approach Using Mobile, Wearable, and
IoT Sensors for Mental Health Detection**

We present a machine learning and deep learning framework that uses multimodal data collected from **mobile**, **wearable deivices**, and **home IoT sensors** to detect signs of **depression and anxiety** in real-world environments.

---

## 🚀 Getting Started

We provide extracted and preprocessed features (in ML models/FEATURES or DL models/FEATURES) to support reproducibility.

🔒 The raw data (from mobile, wearable, and IoT sensors) is not publicly released due to privacy concerns.


## 📋 Mobile Data Preprocessing 
| **Type**               | **Raw Data**                   | **Preprocessing**                                                                                                                                                                                                                   |
| ---------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Social interaction** | Call event                     | Filter negative and 0 duration                                                                                                                                                                                                      |
|                        | Message event                  | Encode the categorical events to 1 as numeric values                                                                                                                                                                                |
| **Physical activity**  | Accelerometer                  | Calculate the magnitude of accelerometer data                                                                                                                                                                                       |
| **Context**            | Location                       | Calculate haversine distance between consecutive GPS recordings; cluster the GPS data for each user using POI clustering; label clusters with semantic labels (home, work, Google Map API labels, and none)                    |
|                        | UltraViolet                    | Calculate UV exposure between consecutive UV recordings                                                                                                                                                                             |
| **Phone usage**        | App usage                      | Recategorize apps into predefined categories, calculate app usage duration for each app usage session                                                                                                                         |
|                        | Installed app                  | Calculate Jaccard similarity index between consecutively installed app names                                                                                                                                                        |
|                        | Screen event                   | Calculate screen-on duration for each screen-on event session for each user                                                                                                                                                         |
|                        | WiFi events                    | Calculate cosine, Euclidean, and Manhattan distance between consecutive WiFi RSSI; calculate Jaccard similarity index between consecutive WiFi BSSID                                                                                |
|                        | Media events                   | Encode the categorical video, image, and all types of events to 1 as numeric values                                                                                                                                                 |


## Mobile Extracted Features

| **Type**               | **Raw Data**            | **Information being aggregated into features**                                               | **Features**                                                                                                   |
| ---------------------- | ----------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Social interaction** | Call event              | Call duration and previous times contacted of the contact person                             | Numeric features, call frequency                                                                               |
|                        | Message event           | Message sent, received, and all events including both sent and received events               | Numeric features                                                                                               |
| **Physical activity**  | Accel.                  | X, Y, Z values, and magnitude                                                                | Numeric features                                                                                               |
|                        | Activity transition     | ENTER\_WALKING, ENTER\_STILL, ENTER\_IN\_VEHICLE, ENTER\_ON\_BICYCLE, ENTER\_RUNNING events  | Categorical features                                                                                           |
|                        | Physical activity event | Confidence of unknown, OnFoot, Walking, InVehicle, OnBicycle, Running, and Tilting           | Numeric features                                                                                               |
| **Context**            | Location                | Distance traveled, location cluster, and location cluster semantic label                     | Numeric features for distance, categorical for location cluster and cluster label, number of locations visited |
| **Time Info**          | Label timestamp         | Day of week, weekend or not, hour name                                                       | Categorical features                                                                                           |
|                        | UltraViolet             | UV exposure, intensity                                                                       | Numeric features                                                                                               |
|                        | Ambient light           | Ambient light brightness                                                                     | Numeric features                                                                                               |
| **Phone usage**        | App usage               | Different types of app usage events and their duration                                       | Categorical features for app events and numeric features for usage duration                                    |
|                        | Installed app           | Jaccard similarity index between consecutively installed app name                            | Numeric features                                                                                               |
|                        | Screen event            | Screen events and screen on duration                                                         | Categorical for screen events and numeric for screen on duration                                               |
|                        | OnOffEvent              | Phone power on/off events                                                                    | Categorical features                                                                                           |
|                        | Network connectivity    | Network connected events                                                                     | Categorical features                                                                                           |
|                        | Battery event           | Battery level, status, and temperature                                                       | Numeric features for battery level and temperature, categorical features for battery status                    |
|                        | Data traffic            | Received and sent data in kbytes                                                             | Numeric features                                                                                               |
|                        | WiFi events             | Three types of distances between consecutive RSSI and Jaccard similarity index between BSSID | Numeric features                                                                                               |
|                        | Media events            | Video, image, and all types of events                                                        | Numeric features                                                                                               |
|                        | System events           | Ringer mode types, power save event types, and mobile charge event types                     | Categorical features                                                                                           |


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