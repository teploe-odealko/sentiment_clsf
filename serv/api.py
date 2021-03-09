import uvicorn
from fastapi import FastAPI
from kedro.runner import SequentialRunner
from pydantic import BaseModel
from kedro.io import DataCatalog, MemoryDataSet
import sys
sys.path.insert(1, '/home/kedro/src')
from sentiment_clsf.pipelines import api_prediction
from kedro.extras.datasets.pickle import PickleDataSet


app = FastAPI()


class Review(BaseModel):
    text: str


@app.post("/review_predict/")
async def review_predict(review: Review):
    io = DataCatalog(
        {
            "model_logreg": PickleDataSet(filepath="data/06_models/model.pkl"),
            "bow_vectorizer": PickleDataSet(filepath="data/06_models/bow_vectorizer.pkl"),
            "tfidf_vectorizer": PickleDataSet(filepath="data/06_models/tfidf_vectorizer.pkl"),
            "w2v_model": PickleDataSet(filepath="data/06_models/w2v_model.pkl"),
            "best_features_selector": PickleDataSet(filepath="data/06_models/best_features_selector.pkl"),
            "review_from_user": MemoryDataSet()
        }
    )
    io.save("review_from_user", review.text)
    pipeline = api_prediction.create_pipeline()
    api_prediction_res = SequentialRunner().run(pipeline, catalog=io)
    return {'text': review.text, 'res': str(api_prediction_res['prediction'])}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)

