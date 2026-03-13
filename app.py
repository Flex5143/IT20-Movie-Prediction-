import streamlit as st
import pandas as pd
import joblib

class MovieSuccessModel:
    def predict(self, X):
        gross = X["gross_millions"].values
        votes = X["votes"].values
        score = (gross * 0.6) + (votes / 100000 * 0.4)
        return [1 if s > 50 else 0 for s in score]

    def predict_proba(self, X):
        gross = X["gross_millions"].values
        votes = X["votes"].values
        score = (gross * 0.6) + (votes / 100000 * 0.4)
        prob = [min(max(s/100,0),1) for s in score]
        return [[1-p, p] for p in prob]

# Load the model
model = joblib.load("random_forest_model.pkl")

# Page configuration
st.set_page_config(page_title="Movie Success Predictor", page_icon="🎬")

# Main title
st.title("🎬 Movie Success Predictor")
st.write("Enter movie details below to predict if it will be successful")

# Create a clean form layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        runtime = st.number_input("Runtime (minutes)", 
                                  min_value=60, 
                                  max_value=180, 
                                  value=120,
                                  help="Length of the movie in minutes")
        
        # Changed to 1-digit slider input
        votes = st.slider("Number of Votes (in 100,000s)", 
                         min_value=0, 
                         max_value=10, 
                         value=1,
                         step=1,
                         help="Number of votes in hundreds of thousands (1 = 100,000 votes)")
        
        # Convert back to actual votes for the model
        actual_votes = votes * 100000
    
    with col2:
        gross = st.number_input("Gross Revenue (Millions $)", 
                                min_value=0.0, 
                                max_value=500.0, 
                                value=50.0,
                                help="Expected box office revenue in millions")
        
        year = st.number_input("Release Year", 
                               min_value=1980, 
                               max_value=2035, 
                               value=2025,
                               help="Year the movie will be released")
    
    # Submit button
    submitted = st.form_submit_button("🎯 Predict Success", type="primary")

# Make prediction when form is submitted
if submitted:
    # Create dataframe for prediction (using actual votes)
    data = pd.DataFrame({
        "runtime_minutes": [runtime],
        "votes": [actual_votes],
        "gross_millions": [gross],
        "year": [year]
    })
    
    # Get prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    # Display results in a clean format
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.success("✅ SUCCESSFUL")
        else:
            st.error("❌ NOT SUCCESSFUL")
    
    with col2:
        # Show probability as percentage
        st.metric("Success Probability", f"{probability:.1%}")
    
    with col3:
        # Show confidence level
        confidence = abs(probability - 0.5) * 2
        if confidence > 0.8:
            st.info("🔮 High Confidence")
        elif confidence > 0.5:
            st.info("📊 Medium Confidence")
        else:
            st.info("📉 Low Confidence")
    
    # Show probability bar
    st.progress(float(probability))
    
    # Show input summary
    st.markdown("---")
    st.subheader("📝 Input Summary")
    
    summary_data = {
        "Feature": ["Runtime", "Votes", "Gross Revenue", "Release Year"],
        "Value": [f"{runtime} min", f"{actual_votes:,}", f"${gross}M", year]
    }
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)
    
    # Show the simplified votes value for clarity
    st.caption(f"ℹ️ Votes slider value: {votes} (representing {votes}00,000 votes)")

