import streamlit as st
import requests
import joblib

from app import (
    recommend_by_location_input,
    recommend_by_facilities_input,
    get_unique_locations,
    get_unique_facilities,
    recommend_by_location_input1,
    recommend_by_price_input,
    get_unique_locations1,
    get_unique_price1,
    recommend_by_location_input2,
    recommend_by_price_input2,
    get_unique_locations2,
    get_unique_price2,
    recommend_by_location_input3,
    recommend_by_price_input3,
    get_unique_locations3,
    get_unique_price3
)

# Together AI API setup
API_KEY = "tgp_v1_OkQDxCrt3mIrkdUovITSSSPQ2Hv5NinS7dyUdzHJaIo"
url = "https://api.together.xyz/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

st.set_page_config(page_title="Real Estate Recommender", layout="wide")
st.title("🏡 Real Estate Property Recommendation System")

# Sidebar selections
city = st.sidebar.selectbox("Choose City", ("Gurgaon", "Bangalore", "Pune", "Delhi"))
rec_type = st.sidebar.radio("Recommendation Type", ("Location Based", "Price/Facilities Based", "Price Prediction"))

st.markdown("---")

# Function to handle different city-based logic
def get_inputs_and_recommend(city, rec_type):
    st.subheader(f"Recommendations for {city} - {rec_type}")

    if city == "Gurgaon":
        if rec_type == "Location Based":
            option = st.selectbox("Select Nearby Location", get_unique_locations())
            if st.button("Recommend"):
                st.write(recommend_by_location_input(option))
        else:
            option = st.selectbox("Select Facility", get_unique_facilities())
            if st.button("Recommend"):
                st.write(recommend_by_facilities_input(option))

    elif city == "Bangalore":
        if rec_type == "Location Based":
            option = st.selectbox("Select Location", get_unique_locations1())
            if st.button("Recommend"):
                st.dataframe(recommend_by_location_input1(option))

        elif rec_type == "Price/Facilities Based":
            option = st.selectbox("Select Price (as string)", get_unique_price1())
            if st.button("Recommend"):
                st.dataframe(recommend_by_price_input(option))

        elif rec_type == "Price Prediction":
            st.subheader("📊 Predict House Price")

            total_sqft = st.number_input("Enter total_sqft", value=1000)
            bath = st.number_input("Enter number of bathrooms", value=2)
            balcony = st.number_input("Enter number of balconies", value=1)

            if st.button("Predict Price"):
                try:
                    model = joblib.load('bangalore_rf_model.pkl')
                    input_data = [[total_sqft, bath, balcony]]
                    predicted_price = model.predict(input_data)[0]
                    st.success(f"🏷️ Estimated Price: ₹ {round(predicted_price, 2)} Lakhs")
                except FileNotFoundError:
                    st.error("Model file not found. Please ensure 'bangalore_rf_model.pkl' is in your directory.")

    elif city == "Pune":
        if rec_type == "Location Based":
            option = st.selectbox("Select Location", get_unique_locations2())
            if st.button("Recommend"):
                st.dataframe(recommend_by_location_input2(option))
        else:
            option = st.selectbox("Select Price (as string)", get_unique_price2())
            if st.button("Recommend"):
                st.dataframe(recommend_by_price_input2(option))

    elif city == "Delhi":
        if rec_type == "Location Based":
            option = st.selectbox("Select Location", get_unique_locations3())
            if st.button("Recommend"):
                st.dataframe(recommend_by_location_input3(option))
        else:
            option = st.selectbox("Select Price (as string)", get_unique_price3())
            if st.button("Recommend"):
                st.dataframe(recommend_by_price_input3(option))

get_inputs_and_recommend(city, rec_type)

# Together AI: Chat section
st.markdown("---")
st.subheader("🤖 Ask Anything About Real Estate")

user_query = st.text_area("Ask a question like: 'best cities to invest in real estate?, comparison'")

if st.button("Get Answer "):
    if user_query.strip():
        with st.spinner("wait..."):
            try:
                payload = {
                    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "messages": [
                        {"role": "system", "content": "You are a helpful real estate assistant."},
                        {"role": "user", "content": user_query}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 200
                }

                response = requests.post(url, headers=headers, json=payload)

                if response.status_code == 200:
                    answer = response.json()["choices"][0]["message"]["content"]
                    st.write(answer)
                else:
                    st.error(f"Error: {response.status_code}, {response.text}")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question first.")
