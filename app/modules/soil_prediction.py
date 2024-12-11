import os
from datetime import datetime

import pandas as pd

from app.modules.persistent_storage_integration_service import PersistentStorageIntegrationService
from app.paths_handler import PathsManager

psis = PersistentStorageIntegrationService()
# files = psis._list_files_in_bucket(psis.bucket_name_for_stationary_images)
# for file in files:
#     capture_date = psis.get_time_of_capture(psis.bucket_name_for_stationary_images, file.object_name)
#     print(capture_date)
#     break
# Read CSV into a DataFrame
df = pd.read_csv(r'../../data/Bodenfeuchte/Ostfalia011.csv')
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%Y/%m/%d %H:%M:%S')


# capture_date = datetime.strptime(capture_date, "%Y:%m:%d %H:%M:%S")  # Adjust format as needed
# for date in df['Date Time']:
#     # Find the row with the minimum time difference
#     closest_row = df.loc[(date - capture_date).abs().idxmin()]


from datetime import datetime, timedelta

def get_dates_to_objects(psis, bucket_name):
    dates_to_objects = {}
    for obj in psis._list_files_in_bucket(bucket_name):
        if "KI-Use-Case" not in obj.object_name:
            continue
        try:
            capture_date = psis.get_time_of_capture(bucket_name, obj.object_name)
            capture_date = datetime.strptime(capture_date, "%Y:%m:%d %H:%M:%S")
            dates_to_objects[capture_date] = obj
        except Exception as e:
            print(f"Error processing {obj.object_name}: {e}")
            continue
    return dates_to_objects

def download_closest_image(psis, bucket_name, given_date, dates_to_objects):
    # Ensure `given_date` is a datetime object
    # given_date = datetime.strptime(given_date, "%Y/%m/%d %H:%M:%S")  # Adjust format as needed
    dates = list(dates_to_objects.keys())
    date_closest = min(dates, key=lambda d: abs(d - given_date))
    closest_object = dates_to_objects[date_closest]

    # Download the closest image
    if closest_object:
        pm = PathsManager(closest_object)
        for fname in pm.file_paths_stationary:
            try:
                response = psis.client.get_object(bucket_name, fname)

                path = "/".join(fname.split("/")[0:-1])
                os.makedirs(path, exist_ok=True)
                with open(fname, "wb") as f:
                    for chunk in response.stream(32 * 1024):  # Stream in chunks of 32 KB
                        f.write(chunk)

                print(f"Closest image downloaded to: {fname}")
            except Exception as e:
                print(f"Error downloading the image {fname}: {e}")
    else:
        print("No images found.")

# Example usage
dates_to_objects = get_dates_to_objects(psis, "cluster-odm")
for ref_date in df['Date Time']:
    download_closest_image(psis, "cluster-odm", ref_date, dates_to_objects)


# Display the first few rows
print(df.head())