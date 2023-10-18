""" main module """
import os
import numpy as np
import pandas as pd
from datetime import datetime

from settings import Settings
settings = Settings()

from clients.google import Gcp, Embedder
gcp = Gcp(settings.EMBBEDINGS_BUCKET)
embedder = Embedder()

from utils.texts import Cleaner
cleaner = Cleaner()

if __name__ == "__main__":
    datasets = ['train','test']
    timestamp = int(datetime.utcnow().timestamp())
    for dataset in datasets:
        df = pd.read_csv(f"src/data/{dataset}.csv")
        df.columns = ['serial', 'id'] + [f"tag_{n}" for n in range(1,9)] + ['review']
        df['y'] = df.apply(lambda k: k.tolist()[2:-1], axis=1)
        df['cleaned_review'] = df['review'].apply(lambda x : cleaner.run(x))
        df['embedded_review'] = [list(embedder.get_vector(x).numpy()[0]) for x in df['cleaned_review']]
        np.savez_compressed(
            "reviews",
            index=df.id,
            embedded_review=df.embedded_review,
            y=df.y
        )
        cloud_file = f"{settings.EMBBEDINGS_BUCKET_USEM_DIRECTORY}/{timestamp}_embedded_reviews_{dataset}.npz"
        print(f"file name saved {cloud_file}")
        gcp.write_file_gcs(
                "reviews.npz",
                cloud_file,
            )
        os.remove("reviews.npz")
