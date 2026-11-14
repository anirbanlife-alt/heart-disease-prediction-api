from pydantic import BaseModel, Field


class HeartData(BaseModel):
    """The 13 details the model needs to make a prediction."""

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="0 = female, 1 = male")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type")
    trestbps: int = Field(..., gt=0, le=300, description="Resting blood pressure")
    chol: int = Field(..., ge=0, le=1000, description="Cholesterol level")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120: 1 = yes")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result")
    thalach: int = Field(..., gt=0, le=300, description="Maximum heart rate")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina: 1 = yes")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia category")
