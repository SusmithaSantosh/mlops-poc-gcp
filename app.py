from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

# Define the input data schema using Pydantic
class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

# Initialize the FastAPI app
app = FastAPI()

# Load the model and preprocessing pipeline
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

    
# Define the prediction endpoint
@app.post("/predict")
def predict(passenger: PassengerData):
    try:
        # Convert input data into a DataFrame
        input_data = passenger.dict()

        # Preprocess the input data to match training data format
        input_data['Sex'] = 1 if input_data['Sex'] == 'female' else 0
        input_data['Embarked'] = {'S': 0, 'C': 1, 'Q': 2}.get(input_data['Embarked'], 0)

        input_df = pd.DataFrame([input_data])

        # Make a prediction
        prediction = model.predict(input_df)

        # Return the prediction as a JSON response
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
