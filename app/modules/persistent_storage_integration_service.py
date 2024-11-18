import os
from xxsubtype import bench

import numpy as np
from minio import Minio
from minio.error import S3Error
import logging
from io import BytesIO

from app.paths_handler import PathsManager


class PersistentStorageIntegrationService:
    def __init__(self):
        self.endpoint = os.getenv("APP_S3_ENDPOINT")
        self.access_key = os.getenv("APP_S3_ACCESS_KEY")
        self.secret_key = os.getenv("APP_S3_SECRET_KEY")

        self.client = Minio(self.endpoint, access_key=self.access_key, secret_key=self.secret_key, secure=False)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.bucket_name_for_images = os.getenv("S3_PRE_CONFIGURED_BUCKET_NAME_FOR_IMAGES")
        self.bucket_name_for_stationary_images = os.getenv(
            "S3_PRE_CONFIGURED_BUCKET_NAME_FOR_STATIONARY_IMAGES")

        # output buckets
        self.bucket_name_for_ai_results = os.getenv("S3_BUCKET_NAME_FOR_AI_RESULTS")
        self.bucket_name_for_registered = os.getenv("S3_BUCKET_NAME_FOR_REGISTERED")
        self.bucket_name_for_unmixed = os.getenv("S3_BUCKET_NAME_FOR_UNMIXED")
        self._ensure_bucket_exists(self.bucket_name_for_images)
        self._ensure_bucket_exists(self.bucket_name_for_registered)
        self._ensure_bucket_exists(self.bucket_name_for_unmixed)


    def _ensure_bucket_exists(self, bucket_name):
        """
        Ensures that the specified bucket exists. If it doesn't, create it.

        :param bucket_name: The name of the bucket to check/create.
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                self.logger.info(f"Bucket '{bucket_name}' created successfully.")
            else:
                self.logger.info(f"Bucket '{bucket_name}' already exists.")
        except S3Error as e:
            self.logger.error(f"Error ensuring bucket '{bucket_name}' exists: {e}")
            raise RuntimeError(f"Could not ensure bucket '{bucket_name}' exists.") from e

    def upload_image_registered(self, image_data, pm: PathsManager):
        bucket_name = self.bucket_name_for_registered
        self._upload_image(image_data, bucket_name, pm)

    def upload_image_unmixed(self, image_data, pm: PathsManager):
        bucket_name = self.bucket_name_for_unmixed
        self._upload_image(image_data, bucket_name, pm)

    def _upload_image(self, image_data, bucket_name, pm: PathsManager):
        try:
            # Ensure image data is C-contiguous
            if not image_data.flags['C_CONTIGUOUS']:
                image_data = np.ascontiguousarray(image_data)

            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                self.logger.info(f"Created bucket {bucket_name}")

            # Convert the image data to bytes if needed
            image_bytes = BytesIO(image_data.tobytes())

            self.client.put_object(
                bucket_name,
                pm.file_path_registered,
                data=image_bytes,
                length=len(image_data.tobytes()),
                content_type="application/octet-stream"
            )
            self.logger.info(f"Image stored successfully: {pm.file_path_registered}")
        except S3Error as e:
            self.logger.error("Could not store image on S3.", exc_info=True)
            raise RuntimeError("Could not store image on S3. Check log for details.") from e

    def _list_files_in_bucket(self, bucket_name):
        try:
            objects = self.client.list_objects(bucket_name, recursive=True)
            return objects
        except S3Error as e:
            self.logger.error(f"Error listing files in bucket {bucket_name}: {e}")
            return []

    def iter_unprocessed(self):
        unprocessed = self._identify_unprocessed()
        for pm in self._store_files(unprocessed):
            yield pm
            for file in os.listdir(pm.cache_folder):
                os.remove(os.path.join(pm.cache_folder, file))
            os.rmdir(pm.cache_folder)


    def _identify_unprocessed(self):
        files_stationary = self._list_files_in_bucket(self.bucket_name_for_stationary_images)
        files_registered = self._list_files_in_bucket(self.bucket_name_for_registered)

        meta_names_registered = []
        for file in files_registered:
            try:
                meta_names_registered.append(PathsManager(file).file_path_registered)
            except AssertionError as e:
                continue

        meta_names_stationary = []
        files_stationary_filtered = []
        for file in files_stationary:
            try:
                path_registered = PathsManager(file).file_path_registered
                if path_registered in meta_names_stationary:
                    continue
                meta_names_stationary.append(path_registered)
            except AssertionError as e:
                continue
            files_stationary_filtered.append(file)


        unprocessed = [f for f, meta_name in zip(files_stationary_filtered, meta_names_stationary)
                       if meta_name not in meta_names_registered]
        return unprocessed


    def _store_files(self, files):
        """
        Downloads given image files to the local cache folder,
        organizing them into subfolders based on their NUMBER identifier.
        """
        processed = set()
        for obj in files:
            try:
                pm = PathsManager(obj)
                if pm.cache_folder in processed: # dont process same imagefile multiple times
                    continue
                processed.add(pm.cache_folder)
                os.makedirs(pm.cache_folder, exist_ok=True)
                for local_file_path, assumed_path_name_minio in zip(pm.file_paths_cache, pm.file_paths_stationary):
                    self.client.fget_object(obj.bucket_name, assumed_path_name_minio, local_file_path)
                yield pm

            except Exception as e:
                self.logger.error(f"Error downloading {obj.object_name}: {e}")


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
    storage_service.identify_unprocessed()
    files = list(storage_service._list_stationary_files())
    breakpoint()
