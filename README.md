# 🚚 Predictive Delivery Optimizer - Streamlit App

This project analyzes historical delivery data to train a machine learning model capable of predicting the risk of delay for new orders. It includes a Streamlit web application that allows users to input new order details, get delay predictions with confidence scores, receive corrective suggestions, and view an interactive dashboard summarizing key analysis findings.

---

## ✨ Features

* **Predict Delay Risk:** Enter details for a new order (priority, carrier, route, value, etc.) and predict if it's likely to be delayed.
* **Confidence Score:** Provides the model's confidence percentage for the delay/on-time prediction.
* **Corrective Suggestions:** Offers actionable advice if an order is flagged as high risk (e.g., change carrier, notify customer, review specific factors).
* **Live Fleet View:** Displays available vehicles by type at the selected origin location (using `vehicle_fleet.csv`).
* **Analysis Dashboard:** Presents interactive charts summarizing key insights derived from the historical data, including:
    * Overall On-Time vs. Delayed Performance
    * Delay Rate Breakdown by Carrier
    * Delay Rate Breakdown by Order Priority
    * Delay Rate Breakdown by Special Handling Type
    * Delay Rate Breakdown by Customer Segment
    * Delay Rate Breakdown by Product Category
    * Delay Rate Breakdown by Order Value Bins
* **Data Exploration:** Includes a Jupyter Notebook (`DeliveryDelayDataAnalysis.ipynb`) detailing the exploratory data analysis (EDA) process.
* **Model Training:** A script (`MLmodel.py`) to preprocess data, engineer features, and train the Random Forest model.

---

## 📊 Analysis Highlights

The underlying analysis (`DeliveryDelayDataAnalysis.ipynb`) revealed several key factors influencing delivery delays:

* **Overall Performance:** Roughly **46.7%** of historical orders were delayed.
* **Carrier Impact:** Carrier choice is crucial. `GlobalTransit` (64%) and `ReliableExpress` (53%) showed the highest delay rates, while `QuickShip` (26%) was the most reliable.
* **Priority Paradox:** "Express" orders had the highest delay rate (50%), indicating potential issues with meeting expedited promises.
* **Handling Challenges:** "Hazmat" orders had a very high delay rate (>70%).
* **Inventory Levels:** Surprisingly, orders where the product stock was *below* the reorder level had a *lower* delay rate (39%) compared to orders with sufficient stock (49%) in this dataset. This might warrant further investigation.
* **Traffic Hotspots:** Certain routes consistently experience higher average traffic delays (e.g., Mumbai-Chennai, Pune-Mumbai, Hyderabad-Chennai).

---

## 🤖 Model Details

* **Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`) with 100 estimators and a max depth of 10.
* **Target Variable:** `is_delayed` (Binary: 1 if `Actual_Delivery_Days > Promised_Delivery_Days`, 0 otherwise).
* **Features Used:**
    * *Categorical:* `Priority`, `Carrier`, `Special_Handling`, `Weather_Impact`, `Origin`, `Destination`, `Customer_Segment`, `Product_Category`
    * *Numerical:* `Distance_KM`, `Traffic_Delay_Minutes`, `Order_Value_INR`
    * *Engineered:* `Is_Below_Reorder` (Binary: 1 if `Current_Stock_Units < Reorder_Level`, 0 otherwise).
* **Preprocessing:** Categorical features are One-Hot Encoded. Numerical features are passed through.
* **Performance:** Achieved approximately **83.33% accuracy** on the test set during the training run documented in `MLmodel.py`.
* **Persistence:** The trained pipeline (preprocessing + model) is saved to `delivery_optimizer_model.joblib`.

---

## 🛠️ Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add Data Files:**
    ❗ **Important:** This repository **does not** include the data CSV files. You must place the following CSV files in the **root directory** of the project:
    * `orders.csv`
    * `routes_distance.csv`
    * `delivery_performance.csv`
    * `vehicle_fleet.csv`
    * `warehouse_inventory.csv`
    * *(Optional, used in notebook)* `cost_breakdown.csv`
    * *(Optional, used in notebook)* `customer_feedback.csv`

---

## 🚀 Usage

1.  **Train the Model (Optional):**
    * If you want to retrain the model (e.g., after updating data), run:
        ```bash
        python MLmodel.py
        ```
    * This will create/overwrite the `delivery_optimizer_model.joblib` file.

2.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    * Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Explore the Analysis (Optional):**
    * Launch Jupyter Notebook or JupyterLab:
        ```bash
        jupyter notebook
        # or
        jupyter lab
        ```
    * Open and run the cells in `DeliveryDelayDataAnalysis.ipynb`.

---

