"""  GCP module """

import numpy as np
from google.cloud import storage
import tensorflow_text
import tensorflow_hub as hub

from settings import Settings
settings = Settings()


class Gcp:
    """
    A class for interacting with Google Cloud Storage (GCS) to read and write files.

    Attributes:
    -----------
    GCS_BUCKET_NAME : str
        The name of the GCS bucket.
    client : google.cloud.storage.Client
        The GCS client.
    bucket : google.cloud.storage.Bucket
        The GCS bucket.

    Methods:
    --------
    read_np_from_gcs(GCS_PATH_FILE: str, allow_pickle: bool = False) -> numpy.ndarray:
        Read a .npz file saved in Google Cloud Storage.

    write_file_gcs(LOCAL_PATH_FILE: str, GCS_PATH_FILE: str, verbose: bool = False) -> None:
        Write a local file to Google Cloud Storage.

    Usage:
    ------
    >>> my_gcp = Gcp("my-gcs-bucket")
    >>> data = my_gcp.read_np_from_gcs("path/to/my/file.npz")
    >>> my_gcp.write_file_gcs("local_file.txt", "path/in/gcs/local_file.txt")
    """

    def __init__(self, GCS_BUCKET_NAME: str):
        """
        Initialize the Gcp instance with the GCS bucket name.

        Parameters:
        -----------
        GCS_BUCKET_NAME : str
            The name of the GCS bucket.
        """
        self.GCS_BUCKET_NAME = GCS_BUCKET_NAME
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(self.GCS_BUCKET_NAME)

    def read_np_from_gcs(self, GCS_PATH_FILE: str, allow_pickle: bool = False) -> np.ndarray:
        """
        Read a .npz file saved in Google Cloud Storage.

        Parameters:
        -----------
        GCS_PATH_FILE : str
            Path to the file in the GCS bucket.
        allow_pickle : bool, optional
            Allow loading pickled objects, default is False.

        Returns:
        --------
        numpy.ndarray
            Loaded numpy object uncompressed or None if unsuccessful.
        """
        try:
            blob = self.bucket.get_blob(GCS_PATH_FILE)
            blob.download_to_filename("file.npz")
            data = np.load("file.npz", allow_pickle=allow_pickle)
            return data
        except Exception as e:
            print(f"Error: {e}")
            return None

    def write_file_gcs(self, LOCAL_PATH_FILE: str, GCS_PATH_FILE: str, verbose: bool = False) -> None:
        """
        Write a local file to Google Cloud Storage.

        Parameters:
        -----------
        LOCAL_PATH_FILE : str
            Local path of the file to be uploaded.
        GCS_PATH_FILE : str
            Path to the file in the GCS bucket.
        verbose : bool, optional
            Whether to print a success message, default is False.

        Returns:
        --------
        None
        """
        try:
            blob = self.bucket.blob(GCS_PATH_FILE)
            blob.upload_from_filename(LOCAL_PATH_FILE)
            if verbose:
                print(
                    f"File '{LOCAL_PATH_FILE}' loaded successfully into {self.GCS_BUCKET_NAME}/{GCS_PATH_FILE}"
                )
        except Exception as e:
            print(f"Error: {e}")


class Embedder:
    """A class for text embedding using various models.

    Args:
        embedding_model (str): The path or name of the embedding model.

    Attributes:
        embed (object): The loaded embedding model.

    Methods:
        get_vector(text: str) -> list: Get the embedding vector for a given text.

    Example:
        # Specify the embedding model path or name (adjust based on your use case)
        embedding_model_path = settings.USEM
        embedder = Embedder(embedding_model_path)

        # Get the vector for a specific text
        text_to_embed = "Hello, world!"
        embedding_vector = embedder.get_vector(text_to_embed)
        print(embedding_vector)
    """

    def __init__(self) -> None:
        """Initialize the Embedder with a specific embedding model."""
        self.embed = hub.load(settings.USEM)

    def get_vector(self, text: str) -> list:
        """Get the embedding vector for a given text.

        Args:
            text (str): The input text to be embedded.

        Returns:
            list: The embedding vector for the input text.
        """
        vector = self.embed(text)
        return vector
