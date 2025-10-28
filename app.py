import streamlit as st
import pandas as pd
import joblib
import altair as alt
import warnings

# no warnings
warnings.filterwarnings('ignore')

# Page Config
# This sets the browser tab title, icon, and makes the page wide layout
st.set_page_config(
    page_title="NexGen Predictive Optimizer",
    page_icon="📦",
    layout="wide"
)

# Caching Functions
# We cache these so they only run once when the app starts, not on every interaction.

@st.cache_data  # Caching
def load_model():
    """Loads the pre-trained model pipeline."""
    try:
        pipeline = joblib.load('delivery_optimizer_model.joblib')
        return pipeline
    except FileNotFoundError:
        st.error("Error: 'delivery_optimizer_model.joblib' not found. Please run MLmodel.py first to create it.")
        return None
    except Exception as e:
        st.error(f"Something went wrong loading the model: {e}")
        return None

@st.cache_data
def load_data():
    """Loads and preps all the CSV data we need for the app."""
    try:
        # Load. Remember keep_default_na False for the files with "None" strings.
        orders = pd.read_csv("orders.csv", keep_default_na=False)
        routes = pd.read_csv("routes_distance.csv", keep_default_na=False)
        deliveries = pd.read_csv("delivery_performance.csv")
        fleet = pd.read_csv("vehicle_fleet.csv")  # The new fleet data
        
        # Quick Cleaning
        orders['Special_Handling'] = orders['Special_Handling'].replace('None', 'Normal Handling')
        routes['Weather_Impact'] = routes['Weather_Impact'].replace('None', 'Normal Weather')
        
        # Pre-calculate route averages
        # This is for the "live" prediction. We'll look up new orders against these averages.
        route_averages = routes.groupby('Route')[['Distance_KM', 'Traffic_Delay_Minutes']].mean().reset_index()
        
        # Merge data for the dashboard charts
        viz_data = pd.merge(orders, deliveries, on="Order_ID", how="inner")
        viz_data = pd.merge(viz_data, routes, on="Order_ID", how="inner")
        
        # Feature Engineering for Charts
        # We need our 'is_delayed' target for the dashboard
        viz_data['is_delayed'] = (viz_data['Actual_Delivery_Days'] > viz_data['Promised_Delivery_Days']).astype(int)
        
        # Bin Order Value for the new chart
        bins = [0, 500, 2000, 5000, 20000, float('inf')]
        labels = ['0-500', '501-2k', '2k-5k', '5k-20k', '20k+']
        viz_data['Order_Value_Bin'] = pd.cut(viz_data['Order_Value_INR'], bins=bins, labels=labels, right=False)
        
        # Return all the dataframes
        return orders, viz_data, route_averages, routes, fleet
    
    except FileNotFoundError as e:
        st.error(f"Error: Missing CSV file. Make sure all 7 CSVs are in the folder. Can't find: {e.filename}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None, None, None, None, None

# Load everything up
pipeline = load_model()
orders, viz_data, route_averages, routes_fallback, fleet = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Predictive Optimizer", "Analysis Dashboard"])


#  PAGE 1: PREDICTIVE OPTIMIZER 

if page == "Predictive Optimizer" and pipeline is not None and orders is not None:
    st.title("📦 Predictive Delivery Optimizer")
    st.markdown("Enter the details of a new order to predict its delay risk and get corrective suggestions.")

    # Create Input Form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Order Details
        with col1:
            st.subheader("Order Details")
            # Grab unique values from our loaded data for the dropdowns
            priority_list = orders['Priority'].unique()
            segment_list = orders['Customer_Segment'].unique()
            category_list = orders['Product_Category'].unique()
            
            in_priority = st.selectbox("Order Priority", priority_list)
            in_segment = st.selectbox("Customer Segment", segment_list)
            in_category = st.selectbox("Product Category", category_list)
            in_order_value = st.number_input("Order Value (INR)", min_value=0.0, value=1000.0, step=100.0)

        # Column 2: Logistics Details
        with col2:
            st.subheader("Logistics Details")
            # Get unique values from the dataframe for dropdowns
            carrier_list = viz_data['Carrier'].unique() # Use viz_data since it's merged
            origin_list = orders['Origin'].unique()
            dest_list = orders['Destination'].unique()
            
            in_carrier = st.selectbox("Assigned Carrier", carrier_list)
            in_origin = st.selectbox("Origin", origin_list)
            in_dest = st.selectbox("Destination", dest_list)

            # NEW: Live Vehicle Chart
            if fleet is not None and in_origin:
                st.markdown("---")
                st.subheader(f"Available Fleet at {in_origin}")
                
                # Filter fleet by selected origin and 'Available' status
                available_fleet = fleet[
                    (fleet['Current_Location'] == in_origin) &
                    (fleet['Status'] == 'Available')
                ]
                
                if available_fleet.empty:
                    st.warning("No vehicles currently available at this location.")
                else:
                    # Tally up the vehicle types
                    available_fleet_chart_df = available_fleet.groupby('Vehicle_Type').size().reset_index(name='Count')
                    
                    # Make a simple bar chart
                    fleet_chart = alt.Chart(available_fleet_chart_df).mark_bar().encode(
                        x=alt.X('Vehicle_Type'),
                        y=alt.Y('Count', title='Available Count'),
                        tooltip=['Vehicle_Type', 'Count']
                    ).interactive()
                    
                    st.altair_chart(fleet_chart, use_container_width=True)


        # Column 3: Conditions
        with col3:
            st.subheader("External Conditions")
            # Get unique values from the dataframe for dropdowns
            handling_list = viz_data['Special_Handling'].unique()
            weather_list = viz_data['Weather_Impact'].unique()
            
            in_handling = st.selectbox("Special Handling", handling_list)
            in_weather = st.selectbox("Weather Impact", weather_list)
            # This is our engineered feature. User just picks Yes/No.
            in_stock = st.selectbox("Is stock below reorder level?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        # Submit Button
        submit_button = st.form_submit_button(label='🔮 Predict Delay Risk')

    # Show Prediction
    if submit_button:
        # Automated Feature Calculation
        # 1. Figure out the route name from the inputs
        route_name = f"{in_origin}-{in_dest}"
        
        # 2. Look this route up in our pre-calculated averages
        route_data = route_averages[route_averages['Route'] == route_name]
        
        if not route_data.empty:
            # 3a. Use the averages for this specific route.
            auto_distance = route_data['Distance_KM'].iloc[0]
            auto_traffic = route_data['Traffic_Delay_Minutes'].iloc[0]
            st.info(f"Route '{route_name}' found. Using average distance: {auto_distance:.0f}km and traffic: {auto_traffic:.0f} mins.")
        else:
            # 3b. No data for this route. Use the overall dataset average as a fallback.
            auto_distance = routes_fallback['Distance_KM'].mean()
            auto_traffic = routes_fallback['Traffic_Delay_Minutes'].mean()
            st.warning(f"Route '{route_name}' not in historical data. Using overall average distance: {auto_distance:.0f}km and traffic: {auto_traffic:.0f} mins.")

        # Create the input DataFrame for the model
        # This has to match the order from the training script!
        
        features_dict = {
            'Priority': [in_priority],
            'Carrier': [in_carrier],
            'Special_Handling': [in_handling],
            'Weather_Impact': [in_weather],
            'Origin': [in_origin],
            'Destination': [in_dest],
            'Customer_Segment': [in_segment],
            'Product_Category': [in_category],
            'Distance_KM': [auto_distance],
            'Traffic_Delay_Minutes': [auto_traffic],
            'Order_Value_INR': [in_order_value],
            'Is_Below_Reorder': [in_stock]
        }
        
        input_df = pd.DataFrame(features_dict)
        
        # Get the prediction.
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0]
        
        st.header("--- Prediction Result ---")
        
        if prediction == 1:
            delay_prob = probability[1] * 100
            st.error(f"**Status: HIGH RISK OF DELAY**")
            st.metric(label="Confidence in Delay", value=f"{delay_prob:.1f}%")
            
            # Corrective Suggestions
            st.subheader("💡 Corrective Suggestions")
            if in_carrier == "GlobalTransit" or in_carrier == "ReliableExpress":
                st.warning(f"**Action:** Re-assign carrier. '{in_carrier}' has a high delay rate. **Suggest 'QuickShip'** for this route.")
            elif in_priority == "Express":
                st.warning(f"**Action:** This 'Express' order is at high risk. Add a 1-day buffer to the customer promise time and notify them immediately.")
            elif in_handling == "Hazmat":
                 st.warning(f"**Action:** 'Hazmat' orders have a >70% failure rate. Flag this order for manual review by a specialist.")
            else:
                st.info(f"**Action:** Review route and traffic. Consider adding a buffer day to the customer's promised time.")
                
        else:
            on_time_prob = probability[0] * 100
            st.success(f"**Status: LIKELY ON-TIME**")
            st.metric(label="Confidence in On-Time", value=f"{on_time_prob:.1f}%")
            st.info("No corrective action needed. Proceed as planned.")

# --- PAGE 2: ANALYSIS DASHBOARD ---

elif page == "Analysis Dashboard" and viz_data is not None:
    st.title("📊 Analysis Dashboard")
    st.markdown("This dashboard shows the key insights from my analysis that power the predictive model.")

    col1, col2 = st.columns(2)
    
    with col1:
        # Overall Delivery Performance
        st.subheader("Overall Delivery Performance")
        status_df = viz_data.groupby('Delivery_Status').count()['Order_ID'].reset_index()
        status_df = status_df.rename(columns={'Order_ID': 'Count'}) 
        
        pie_base = alt.Chart(status_df).encode(
            theta=alt.Theta("Count:Q", stack=True)
        )
        
        pie_chart = pie_base.mark_arc(outerRadius=120).encode(
            color=alt.Color("Delivery_Status:N"),
            order=alt.Order("Count:Q", sort="descending"),
            tooltip=["Delivery_Status", "Count", alt.Tooltip("Count:Q", format=".1%")]
        )
        # The text labels
        text = pie_base.mark_text(radius=140).encode(
            text=alt.Text("Count:Q", format=".1%"),
            order=alt.Order("Count:Q", sort="descending"),
            color=alt.value("black")  # Make text black
        )
        
        final_pie = (pie_chart + text).properties(
            title='Nearly Half of All Orders are Delayed'
        )
        st.altair_chart(final_pie, use_container_width=True)

        # Delay Rate by Carrier
        st.subheader("Delay Rate by Carrier")
        carrier_delay_df = viz_data.groupby('Carrier')['is_delayed'].mean().reset_index()
        carrier_delay_df = carrier_delay_df.rename(columns={'is_delayed': 'Delay_Rate'})
        carrier_delay_df['Delay_Rate_Percent'] = carrier_delay_df['Delay_Rate'] * 100

        carrier_chart = alt.Chart(carrier_delay_df).mark_bar().encode(
            x=alt.X('Carrier', sort='-y'),
            y=alt.Y('Delay_Rate_Percent', title='Delay Rate (%)'),
            tooltip=['Carrier', alt.Tooltip('Delay_Rate_Percent', format='.1f')]
        ).properties(
            title='Carrier Performance is a Key Driver'
        ).interactive()
        st.altair_chart(carrier_chart, use_container_width=True)

        # Delay Rate by Customer Segment
        st.subheader("Delay Rate by Customer Segment")
        segment_delay_df = viz_data.groupby('Customer_Segment')['is_delayed'].mean().reset_index()
        segment_delay_df = segment_delay_df.rename(columns={'is_delayed': 'Delay_Rate'})
        segment_delay_df['Delay_Rate_Percent'] = segment_delay_df['Delay_Rate'] * 100

        segment_chart = alt.Chart(segment_delay_df).mark_bar().encode(
            x=alt.X('Customer_Segment', sort='-y'),
            y=alt.Y('Delay_Rate_Percent', title='Delay Rate (%)'),
            tooltip=['Customer_Segment', alt.Tooltip('Delay_Rate_Percent', format='.1f')]
        ).properties(
            title='Delay Rate for Customer Segments'
        ).interactive()
        st.altair_chart(segment_chart, use_container_width=True)
        
        # Delay Rate by Order Value
        st.subheader("Delay Rate by Order Value (INR)")
        # Drop any NaNs from the binning just in case
        value_delay_df = viz_data.dropna(subset=['Order_Value_Bin']).groupby('Order_Value_Bin')['is_delayed'].mean().reset_index()
        value_delay_df = value_delay_df.rename(columns={'is_delayed': 'Delay_Rate'})
        value_delay_df['Delay_Rate_Percent'] = value_delay_df['Delay_Rate'] * 100

        value_chart = alt.Chart(value_delay_df).mark_bar().encode(
            x=alt.X('Order_Value_Bin', sort=None), # Keep bin order
            y=alt.Y('Delay_Rate_Percent', title='Delay Rate (%)'),
            tooltip=['Order_Value_Bin', alt.Tooltip('Delay_Rate_Percent', format='.1f')]
        ).properties(
            title='Delay Rate by Order Value Bins'
        ).interactive()
        st.altair_chart(value_chart, use_container_width=True)

    with col2:
        # Delay Rate by Priority
        st.subheader("Delay Rate by Priority")
        priority_delay_df = viz_data.groupby('Priority')['is_delayed'].mean().reset_index()
        priority_delay_df = priority_delay_df.rename(columns={'is_delayed': 'Delay_Rate'})
        priority_delay_df['Delay_Rate_Percent'] = priority_delay_df['Delay_Rate'] * 100

        priority_chart = alt.Chart(priority_delay_df).mark_bar().encode(
            x=alt.X('Priority', sort='-y'),
            y=alt.Y('Delay_Rate_Percent', title='Delay Rate (%)'),
            tooltip=['Priority', alt.Tooltip('Delay_Rate_Percent', format='.1f')]
        ).properties(
            title='"Express" Priority is the Least Reliable'
        ).interactive()
        st.altair_chart(priority_chart, use_container_width=True)

        # Delay Rate by Special Handling
        st.subheader("Delay Rate by Special Handling")
        handling_delay_df = viz_data.groupby('Special_Handling')['is_delayed'].mean().reset_index()
        handling_delay_df = handling_delay_df.rename(columns={'is_delayed': 'Delay_Rate'})
        handling_delay_df['Delay_Rate_Percent'] = handling_delay_df['Delay_Rate'] * 100

        handling_chart = alt.Chart(handling_delay_df).mark_bar().encode(
            x=alt.X('Special_Handling', sort='-y'),
            y=alt.Y('Delay_Rate_Percent', title='Delay Rate (%)'),
            tooltip=['Special_Handling', alt.Tooltip('Delay_Rate_Percent', format='.1f')]
        ).properties(
            title='"Hazmat" Orders Fail >70% of the Time'
        ).interactive()
        st.altair_chart(handling_chart, use_container_width=True)

        # Delay Rate by Product Category
        st.subheader("Delay Rate by Product Category")
        category_delay_df = viz_data.groupby('Product_Category')['is_delayed'].mean().reset_index()
        category_delay_df = category_delay_df.rename(columns={'is_delayed': 'Delay_Rate'})
        category_delay_df['Delay_Rate_Percent'] = category_delay_df['Delay_Rate'] * 100

        category_chart = alt.Chart(category_delay_df).mark_bar().encode(
            x=alt.X('Product_Category', sort='-y'),
            y=alt.Y('Delay_Rate_Percent', title='Delay Rate (%)'),
            tooltip=['Product_Category', alt.Tooltip('Delay_Rate_Percent', format='.1f')]
        ).properties(
            title='Delay Rate by Product Category'
        ).interactive()
        st.altair_chart(category_chart, use_container_width=True)


# if the data or model failed to load
elif pipeline is None or orders is None or fleet is None:
    st.error("Application cannot start. Please check the console for errors. Did you run the training script?")