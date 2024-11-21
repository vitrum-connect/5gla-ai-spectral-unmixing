import os
import time
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from app.modules import image_registration, unmixing
from app.modules.persistent_storage_integration_service import PersistentStorageIntegrationService

app = Flask(__name__)
psis = PersistentStorageIntegrationService()


@app.route('/')
def home():
    return "Hello, World!"


@app.route('/run')
def run():
    for pm in psis.iter_unprocessed(): #path_channels, path_parts, number_part
        captured, im_aligned, registered_images_folder = image_registration.run(pm.cache_folder)
        psis.upload_image_registered(im_aligned, pm)
        unmixed, abundances = unmixing.run(registered_images_folder)
        psis.upload_image_unmixed(unmixed, pm, name_appendix="reconstructed")
        psis.upload_image_unmixed(abundances, pm, name_appendix="_abundances")
    return "Run completed"


def delayed_scheduled_run():
    while True:
        start_time = time.time()

        with app.app_context():  # Ensure the app context is available for scheduled tasks
            run()

        run_duration = time.time() - start_time
        interval = int(os.getenv("INTERVAL_TIME_AI_SECONDS", 600))  # 600 seconds = 10 minutes
        sleep_time = max(0, interval - run_duration)
        time.sleep(sleep_time)


if __name__ == '__main__':
    # Set up the scheduler to run `delayed_scheduled_run` as a single job
    scheduler = BackgroundScheduler()
    scheduler.add_job(delayed_scheduled_run, 'date')  # Run only once; looping will handle repetition
    scheduler.start()

    # Start the Flask app
    app.run(host='0.0.0.0', port=8080)
