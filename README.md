# Heat Stress / Heat Index Prediction System

A machine learning–based system to predict human heat stress (heat index) using essential weather parameters such as temperature, humidity, and wind speed.

---

## 📌 Overview

This project predicts **heat stress / heat index** by analyzing the combined effect of atmospheric conditions on human comfort and health.  
The system currently accepts **manual user inputs** and is designed to be scalable for **real-time, location-based automation** in the future.

---

## 🎯 Objectives

- Predict heat stress using machine learning
- Understand the impact of temperature, humidity, and wind speed
- Enable manual weather parameter input
- Design a foundation for GPS-based automation
- Support future alert and notification systems for extreme heat

---

## 🧑‍💻 User Inputs

The user must provide the following parameters:

| Parameter | Description | Unit |
|----------|------------|------|
| Air Temperature | Ambient atmospheric temperature | °C |
| Relative Humidity | Moisture content in air | % |
| Wind Speed | Speed of air movement | km/h |

---

## 📊 Features / Columns Used

| Feature Name | Description | Unit |
|-------------|------------|------|
| Air Temperature | Ambient temperature | °C |
| Relative Humidity | Atmospheric moisture | % |
| Wind Speed | Horizontal wind velocity | m/s |
| Wet Bulb Temperature | Combined heat & humidity metric | °C |
| Dew Point Temperature | Moisture saturation temperature | °C |
| Heat Index (Target) | Perceived temperature felt by humans | °C |
| Date / Time | Observation timestamp | — |

---

## 🧠 Machine Learning Workflow

1. Data preprocessing and cleaning  
2. Feature selection and scaling  
3. Model training using supervised learning  
4. Model evaluation and error analysis  
5. Prediction on new user-provided inputs  

---

## 🛠️ Technologies Used

- **Python 3**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **Jupyter Notebook**
- **VS Code**
- **Git & GitHub**

---

## 🔮 Future Enhancements

- Automatic weather data fetching using GPS
- Integration with real-time weather APIs
- Heat alert notifications (SMS / system alerts)
- Web or mobile application deployment
- Multi-location model generalization

---


## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.  
Predictions should not be used as a substitute for official meteorological or medical advisories.

---

## 👤 Author

**Shubhra Kanti Banerjee**  
Engineering Student  
Machine Learning & Web Development
