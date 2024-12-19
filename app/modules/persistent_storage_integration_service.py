import os
from minio import Minio
from minio.error import S3Error
import logging
from io import BytesIO

import tifffile
from tifffile import TiffWriter

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
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                self.logger.info(f"Bucket '{bucket_name}' created successfully.")
            else:
                self.logger.info(f"Bucket '{bucket_name}' already exists.")
        except S3Error as e:
            self.logger.error(f"Error ensuring bucket '{bucket_name}' exists: {e}")
            raise RuntimeError(f"Could not ensure bucket '{bucket_name}' exists.") from e

    def upload_image_registered(self, image_data, pm: PathsManager, name_appendix=""):
        bucket_name = self.bucket_name_for_registered
        self._upload_image(image_data, bucket_name, pm, name_appendix)

    def upload_image_unmixed(self, image_data, pm: PathsManager, name_appendix=""):
        bucket_name = self.bucket_name_for_unmixed
        self._upload_image(image_data, bucket_name, pm, name_appendix)

    def _upload_image(self, image_data, bucket_name, pm: PathsManager, name_appendix=""):
        file_path = pm.file_path_registered
        file_path_metadata = pm.file_paths_cache[0]  # Path to the metadata TIFF file

        if name_appendix:
            splitted = file_path.split(".")
            splitted[-2] += name_appendix
            file_path = ".".join(splitted)

        # Extract metadata from the source TIFF file
        if file_path_metadata:
            relevant_tags = ["Make", "Model", "Software", "DateTime", "ExifTag", "GPSTag|OlympusSIS2",
                             "PlanarConfiguration"]
            try:
                with tifffile.TiffFile(file_path_metadata) as tif:
                    tags = tif.pages[0].tags
                    tag_tuples = [(v.code, v.dtype, v.valuebytecount, v.value, False) for tags, v in tags.items()
                                  if v.name in relevant_tags]
            except Exception as e:
                self.logger.error(f"Failed to load metadata from {file_path_metadata}: {e}")
                raise

        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                self.logger.info(f"Created bucket {bucket_name}")

            # Create buffer to hold the TIFF image
            tiff_buffer = BytesIO()
            with TiffWriter(tiff_buffer) as tiff_writer:
                tiff_writer.write(
                    image_data,
                    photometric='minisblack',  # Replace with your image settings
                    planarconfig='contig',
                    extratags=tag_tuples
                )

            tiff_buffer.seek(0)

            # Upload TIFF file to S3
            self.client.put_object(
                bucket_name,
                file_path,
                data=tiff_buffer,
                length=tiff_buffer.getbuffer().nbytes,
                content_type="image/tiff"
            )
            self.logger.info(f"Image stored successfully: {file_path}")

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

    def get_time_of_capture(self, bucket_name, object_name):
        # Download object into memory
        try:
            response = self.client.get_object(bucket_name, object_name)
            with BytesIO(response.data) as tiff_data:
                with tifffile.TiffFile(tiff_data) as tif:
                    # Metadata is often stored in the TIFF tags
                    tags = tif.pages[0].tags
                    time_of_capture = tags.get('DateTime', None)
                    return time_of_capture.value if time_of_capture else None
        except Exception as e:
            print(f"Error extracting metadata from {object_name}: {e}")
            return None

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
