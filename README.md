#  Used Car Price Predictor

A powerful machine learning web app that predicts used car prices based on real-world listing data.  
Built with Python, CatBoost, and Streamlit.

---

## Description

This project uses a CatBoost regression model trained on over **400,000 used car listings** to estimate resale value.  
It incorporates real-world features like mileage, car age, condition, fuel type, transmission, and more.

You can access the app through a browser via Streamlit, enter the details of a used car, and receive an instant price prediction.

---
##  Working App

Check out the **live demo** of the Used Car Price Estimator here:

👉 [🟢 Launch App](https://usedcarspriceestimator.streamlit.app/)  
*(hosted with Streamlit Cloud)*

---

##  Features

-  Predict used car prices with **92.7% R² score**
-  Built with **CatBoost**, optimized with log-transformed targets
-  Fast predictions on live user input
-  Interactive and clean **Streamlit UI**
-  Modular, reproducible structure for training and inference

---

##  Model Performance

| Metric       | Value            |
|--------------|------------------|
| **R² Score** | 0.927            |
| **Model**    | CatBoostRegressor |
| **Data Size**| ~400,000 samples |

---

###  Steps to run locally


```git clone https://github.com/yourusername/used-car-price-predictor.git```

```cd used-car-price-predictor```

```python -m venv venv```

```source venv/bin/activate```      
#### On Windows:  ```venv\Scripts\activate```

```pip install -r requirements.txt```

```streamlit run app.py```

##  Author

**Diana Shadibaeva**  
🎓Data Science @ Michigan Technological University  
💡 Passionate about ML, AI, and deploying smart tools  
🧊 Also into skating, music, and building creative apps  

- 🔗 [LinkedIn](www.linkedin.com/in/diana-shadibaeva-1956b829a)

---

##  License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute it with attribution.


