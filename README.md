# Heart Disease Prediction API

This is my student project for the **FastAPI + Docker + Deployment** assignment.
I trained a machine-learning model using heart patient information, then made a
small API so another application can send patient details and receive a prediction.

> **Important:** This project is only for learning and demonstration. It is not a
> medical tool and must not be used to diagnose or make treatment decisions.

## 1. What is included?

- `app/` – the FastAPI application and the input format.
- `train_model.py` – reads the dataset, trains the model, checks its result, and saves it.
- `model/` – where the trained model is saved.
- `data/` – where `heart.csv` can be placed.
- `Dockerfile` – instructions for building the application image.
- `docker-compose.yml` – an easy way to run the app with Docker Compose.
- `requirements.txt` – the Python packages needed by this project.

## 2. Dataset and model

The assignment uses the Heart Disease dataset from Kaggle:

https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Download `heart.csv` and put it inside the `data` folder. The file should be:

`data/heart.csv`

If you do not put the CSV there, the training script downloads a public copy
with the same common 14-column layout. For submission, it is best to use the
dataset file provided by the assignment.

The model uses these 13 input columns:

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

The `target` column is the answer the model learns to predict:
- `0` means no heart disease in the dataset.
- `1` means heart disease is present in the dataset.

I used a Random Forest classifier. The script splits the data into training
and testing parts, trains the model, prints a simple evaluation, and saves
the trained pipeline as `model/heart_model.joblib`.

## 3. Run it on your computer

You need Python 3.11 or Docker installed.

### Option A: Run with Python

Open a terminal in the project folder and run:

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open this address in your browser:

http://localhost:8000/docs

### Option B: Run with Docker Compose (the assignment requirement)

Make sure Docker Desktop is running. From the project folder, run:

```bash
docker-compose build
docker-compose up
```

If your Docker installation uses the newer command, this also works:

```bash
docker compose build
docker compose up
```

Open the API documentation:

http://localhost:8000/docs

To stop the app, press `Ctrl + C` in the terminal.

## 4. API endpoints

### GET `/health`

Checks whether the API is running and whether the model was loaded.

### GET `/info`

Shows the model name, input features, and target labels.

### POST `/predict`

Sends patient information and receives a prediction.

Example JSON body:

```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```

The response includes `heart_disease` as `true` or `false`.
The values above are only an example request, not a real patient's details.

You can also test the endpoints using Postman or curl. For example:

```bash
curl http://localhost:8000/health
```

## 5. Deploy on Render

1. Make sure the project runs locally with Docker Compose.
2. Create a GitHub repository and upload all project files.
3. Open Render and choose **New → Web Service**.
4. Connect your GitHub repository.
5. Choose **Docker** as the environment.
6. Keep the build context as the project root (the folder containing `Dockerfile`).
7. Create the service and wait for the build and deployment to finish.
8. Open the Render URL and add `/docs` to test the API.

Render builds the Docker image. During the build, `train_model.py` downloads
the dataset if it is not already in `data/`, trains the model, and saves it
inside the image. No separate start command is needed because the Dockerfile
already starts Uvicorn.

After deployment, test:
- `https://YOUR-RENDER-URL/health`
- `https://YOUR-RENDER-URL/info`
- `https://YOUR-RENDER-URL/docs`

Replace `YOUR-RENDER-URL` with the URL Render gives you.

## 6. Before submitting

Your GitHub repository should contain:

- `app/`
- `model/` (the trained model will be created after running `train_model.py`)
- `data/` (optional; you can keep the dataset out of GitHub if required)
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`
- `train_model.py`
- `README.md`

Submit these two things to your teacher:
1. Your GitHub repository link.
2. Your live Render deployment URL.

## A small note about the result

The accuracy printed by the training script is based on the test split from this
dataset. A good-looking score does not mean the model is reliable for real
medical use. This project is mainly to practise training a model, building an
API, using Docker, and deploying an application.
