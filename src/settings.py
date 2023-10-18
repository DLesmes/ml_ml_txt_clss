import os
from dotenv import load_dotenv
load_dotenv()


class Settings:
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    EMBBEDINGS_BUCKET=os.getenv("EMBBEDINGS_BUCKET")
    EMBBEDINGS_BUCKET_USEM_DIRECTORY=os.getenv("EMBBEDINGS_BUCKET_USEM_DIRECTORY")
    GOOGLE_APPLICATION_CREDENTIALS=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    USEM="https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
