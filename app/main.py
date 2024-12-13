import os
import time
import logging
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from app.modules import image_registration, unmixing, plant_indices
from app.modules.persistent_storage_integration_service import PersistentStorageIntegrationService

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbose logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file named app.log
        logging.StreamHandler()         # Also log to the console
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
psis = PersistentStorageIntegrationService()


@app.route('/')
def home():
    logger.info("Home route accessed")
    return "Hello, World!"


@app.route('/run')
def run():
    logger.info("Run endpoint triggered")
    for pm in psis.iter_unprocessed():# path_channels, path_parts, number_part
        try:
            logger.info(f"Processing unprocessed data for: {pm.file_paths_stationary}")

            captured, im_aligned_reflectance, im_aligned_reflectance_norm, registered_images_folder = image_registration.run(pm.cache_folder)
            logger.info(f"Image registration completed")

            psis.upload_image_registered(im_aligned_reflectance_norm, pm)
            logger.info(f"Uploaded registered image")

            reconstructed, abundances, endmembers_plant, endmembers_non_plant = unmixing.run(im_aligned_reflectance)
            logger.info(f"Spectral unmixing completed")

            psis.upload_image_unmixed(reconstructed, pm, name_appendix="_reconstructed")
            logger.info(f"Uploaded unmixed image (reconstructed)")

            psis.upload_image_unmixed(abundances, pm, name_appendix="_abundances")
            logger.info(f"Uploaded unmixed image (abundances)")

            savi, savi_norm = plant_indices.savi(abundances, im_aligned_reflectance, endmembers_plant.shape[0])
            logger.info(f"savi index complete")

            psis.upload_image_unmixed(savi_norm, pm, name_appendix="_savi")
            logger.info(f"Uploaded savi")

        except Exception as e:
            logger.exception(f"Error during processing of {pm.file_paths_stationary}: {str(e)}")
            continue
            # return f"Error during processing of {pm.file_paths_stationary}: {str(e)}", 500

    logger.info("Run completed successfully")
    return "Run completed"


def delayed_scheduled_run():
    while True:
        logger.info("Scheduled run started")
        start_time = time.time()

        try:
            with app.app_context():  # Ensure the app context is available for scheduled tasks
                run()

        except Exception as e:
            logger.exception("Error in scheduled run")

        run_duration = time.time() - start_time
        interval = int(os.getenv("INTERVAL_TIME_AI_SECONDS", 600))  # 600 seconds = 10 minutes
        sleep_time = max(0, interval - run_duration)

        logger.info(f"Scheduled run finished. Duration: {run_duration:.2f} seconds. Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)


if __name__ == '__main__':
    logger.info("Starting application")

    try:
        # Set up the scheduler to run `delayed_scheduled_run` as a single job
        scheduler = BackgroundScheduler()
        scheduler.add_job(delayed_scheduled_run, 'date')  # Run only once; looping will handle repetition
        scheduler.start()
        logger.info("Scheduler started")

        # Start the Flask app
        app.run(host='0.0.0.0', port=8080)
    except Exception as e:
        logger.exception("Application encountered an error")
    finally:
        scheduler.shutdown()
        logger.info("Scheduler shut down")
