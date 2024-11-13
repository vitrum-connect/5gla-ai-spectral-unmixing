from flask import Flask

from app.modules import image_registration, unmixing
from app.modules.persistent_storage_integration_service import PersistentStorageIntegrationService

app = Flask(__name__)
psis = PersistentStorageIntegrationService()

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/run')
def run():
    psis.store_stationary_files()
    for image_path in psis.get_unprocessed_stationary_data():
        captured, im_aligned, registered_images_path = image_registration.run(image_path)
        # unmixing.run(registered_images_path)


if __name__ == '__main__':
    run()
    # app.run(host='0.0.0.0', port=8080)
