from pydantic import BaseModel
from fastapi import FastAPI
import joblib

# Create FastAPI instance
app = FastAPI()

# Create a class to validate request body
class RequestBody(BaseModel):
  study_hours: float

# Load the model
pontuation_model = joblib.load('./pontuation_model.pkl')

@app.post('/predict')
def predict(data: RequestBody):
  input_feature = [[data.study_hours]]

  y_pred = pontuation_model.predict(input_feature)[0].astype(int)

  return {'pontuation': y_pred.tolist()}

# to run:
  # uvicorn api_regression_model:app --reload