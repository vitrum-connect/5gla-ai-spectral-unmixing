import os
from xxsubtype import bench

from minio import Minio
from minio.error import S3Error
import logging
from io import BytesIO


class PersistentStorageIntegrationService:
    def __init__(self):
        self.endpoint = os.getenv("APP_S3_ENDPOINT")
        self.access_key = os.getenv("APP_S3_ACCESS_KEY")
        self.secret_key = os.getenv("APP_S3_SECRET_KEY")
        self.bucket_name_for_images = os.getenv("APP_S3_PRE_CONFIGURED_BUCKET_NAME_FOR_IMAGES")
        self.bucket_name_for_stationary_images = os.getenv(
            "APP_S3_PRE_CONFIGURED_BUCKET_NAME_FOR_STATIONARY_IMAGES")

        self.client = Minio(self.endpoint, access_key=self.access_key, secret_key=self.secret_key, secure=False)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def store_image(self, transaction_id, image):
        filename = self._get_full_filename(image.tenant, transaction_id, image.filename)
        self._store_image(image.raw_data, self.bucket_name_for_images, filename)

    def store_stationary_image(self, stationary_image):
        filename = self._get_full_filename(stationary_image.tenant, stationary_image.filename)
        self._store_image(stationary_image.raw_data, self.bucket_name_for_stationary_images, filename)

    def _store_image(self, image_data, bucket_name, filename):
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                self.logger.info(f"Created bucket {bucket_name}")

            self.client.put_object(
                bucket_name,
                filename,
                data=BytesIO(image_data),
                length=len(image_data),
                content_type="application/octet-stream"
            )
            self.logger.info(f"Image stored successfully: {filename}")
        except S3Error as e:
            self.logger.error("Could not store image on S3.", exc_info=True)
            raise RuntimeError("Could not store image on S3. Check log for details.") from e

    def get_unprocessed_stationary_data(self):
        yield r"cache\stationary_images\IMG_0000"

    def _list_stationary_files(self):
        try:
            objects = self.client.list_objects(self.bucket_name_for_stationary_images, recursive=True)
            return objects
        except S3Error as e:
            self.logger.error(f"Error listing files in bucket {self.bucket_name_for_stationary_images}: {e}")
            return []

    def store_stationary_files(self, cache_folder="cache"):
        """
        Downloads all listed files in the stationary images bucket to the local cache folder,
        organizing them into subfolders based on their NUMBER identifier.

        :param cache_folder: The root directory for storing downloaded files.
        """
        files = self._list_stationary_files()

        for obj in files:
            file_name = obj.object_name
            try:
                # Extract the NUMBER part from the file name, e.g., "IMG_0000_1.tif" -> "0000"
                number_part = file_name.split('_')[1]
                target_folder = os.path.join(cache_folder, number_part)
                os.makedirs(target_folder, exist_ok=True)

                # Define the local file path
                local_file_path = os.path.join(target_folder, os.path.basename(file_name))

                # Download the file and save it to the local file path
                self.client.fget_object(self.bucket_name_for_stationary_images, file_name, local_file_path)
                self.logger.info(f"Downloaded {file_name} to {local_file_path}")

            except Exception as e:
                self.logger.error(f"Error downloading {file_name}: {e}")


    def get_result_file(self, tenant, transaction_id):
        filename = self._get_filename_for_result_file(tenant, transaction_id)
        try:
            response = self.client.get_object(self.bucket_name_for_images, filename)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            self.logger.error("Could not retrieve result file from S3.", exc_info=True)
            return None

    def _get_full_filename(self, tenant, transaction_id, filename):
        return f"{tenant}/{transaction_id}/{filename}"

    def _get_filename_for_result_file(self, tenant, transaction_id):
        return f"{tenant}/{transaction_id}/result.zip"


if __name__ == "__main__":
    storage_service = PersistentStorageIntegrationService()
    files = list(storage_service._list_stationary_files())
    breakpoint()
