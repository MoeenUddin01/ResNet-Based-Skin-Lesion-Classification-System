from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse

from app.middleware import setup_middleware
from app.dependencies import get_model
from app.predict import predict_image
from app.schemas import PredictionResponse

def create_app() -> FastAPI:
    """Application factory for the FastAPI backend."""
    app = FastAPI(
        title="Skin Lesion Classification API",
        description="API and UI for interacting with the ResNet-152 model.",
        version="1.0.0",
    )
    
    setup_middleware(app)
    
    @app.get("/", response_class=HTMLResponse, summary="Serve the React UI")
    def serve_frontend():
        """Returns the main index.html which contains the React UI."""
        ui_path = Path("app/templates/index.html")
        if not ui_path.exists():
            return HTMLResponse("<h1>UI template missing.</h1>", status_code=404)
        return HTMLResponse(content=ui_path.read_text())
        
    @app.get("/health", summary="Health Check")
    def health_check():
        """Returns the status of the API."""
        return {"status": "ok"}
        
    @app.post("/api/v1/predict", response_model=PredictionResponse, summary="Predict Skin Lesion Class")
    def predict_endpoint(
        file: UploadFile = File(...),
        model=Depends(get_model),
    ):
        """Accepts an image file and returns the model prediction."""
        try:
            contents = file.file.read()
            return predict_image(contents, model)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
            
    return app

app = create_app()
