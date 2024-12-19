# 5gla-ai

This repository contains an application designed for processing multispectral image data through a pipeline involving image registration, spectral unmixing, and plant index calculations. The application integrates with S3-compatible storage for handling input and output data.

## Features

- **Image Registration:** Aligns multispectral image channels.
- **Spectral Unmixing:** Decomposes spectral images into component abundances and reconstructs spectral data.
- **Plant Index Calculation:** Calculates the SAVI (Soil Adjusted Vegetation Index).
- **Persistent Storage Integration:** Uses S3-compatible storage for input and output data management.
- **Flask API:** Provides endpoints for triggering the processing pipeline.
- **Background Scheduler:** Automates periodic processing tasks.

## Installation

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)
- An S3-compatible storage service (e.g., MinIO)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export APP_S3_ENDPOINT=<s3_endpoint>
   export APP_S3_ACCESS_KEY=<s3_access_key>
   export APP_S3_SECRET_KEY=<s3_secret_key>
   export S3_PRE_CONFIGURED_BUCKET_NAME_FOR_IMAGES=<bucket_for_images>
   export S3_PRE_CONFIGURED_BUCKET_NAME_FOR_STATIONARY_IMAGES=<bucket_for_stationary_images>
   export S3_BUCKET_NAME_FOR_AI_RESULTS=<bucket_for_ai_results>
   export S3_BUCKET_NAME_FOR_REGISTERED=<bucket_for_registered>
   export S3_BUCKET_NAME_FOR_UNMIXED=<bucket_for_unmixed>
   export INTERVAL_TIME_AI_SECONDS=600  # Interval for scheduled tasks in seconds
   ```

4. Run the application:
   ```bash
   python main.py
   ```

## Usage

### Flask Endpoints

#### Home
- **URL:** `/`
- **Method:** GET
- **Description:** Health check for the application.

#### Run Pipeline
- **URL:** `/run`
- **Method:** GET
- **Description:** Manually trigger the processing pipeline.

### Automated Processing
The application uses a background scheduler to automatically process data at regular intervals. The interval is configurable through the `INTERVAL_TIME_AI_SECONDS` environment variable.

## Persistent Storage Integration
The `PersistentStorageIntegrationService` class manages S3-compatible storage:

- Ensures required buckets exist.
- Provides methods to upload processed data, such as registered images, unmixed images, and plant indices.

## Processing Pipeline
1. **Image Registration:**
   - Aligns input multispectral images.
   - Outputs normalized reflectance images.

2. **Spectral Unmixing:**
   - Decomposes images into abundances and endmembers.
   - Outputs reconstructed images and abundance maps.

3. **Plant Index Calculation:**
   - Computes the SAVI index.
   - Outputs normalized SAVI images.

4  **Moisture prediction:**
   - Computes predicted soil moisture for 9 depths.
   - based on multispectral images.
   - training script included

5**Data Upload:**
   - Uploads all outputs to appropriate S3 buckets.

## Logging
- Logs are written to both `app.log` and the console.
- Use `INFO` level for standard operations and `DEBUG` level for detailed troubleshooting.

## Deployment

### Docker
1. Build the Docker image:
   ```bash
   docker build -t ai-spectral-unmixing .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8080:8080 --env-file .env ai-spectral-unmixing
   ```

### Kubernetes
1. Create a ConfigMap for environment variables:
   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: ai-spectral-unmixing-config
   data:
     APP_S3_ENDPOINT: <s3_endpoint>
     APP_S3_ACCESS_KEY: <s3_access_key>
     APP_S3_SECRET_KEY: <s3_secret_key>
     S3_PRE_CONFIGURED_BUCKET_NAME_FOR_IMAGES: <bucket_for_images>
     S3_PRE_CONFIGURED_BUCKET_NAME_FOR_STATIONARY_IMAGES: <bucket_for_stationary_images>
     S3_BUCKET_NAME_FOR_AI_RESULTS: <bucket_for_ai_results>
     S3_BUCKET_NAME_FOR_REGISTERED: <bucket_for_registered>
     S3_BUCKET_NAME_FOR_UNMIXED: <bucket_for_unmixed>
     INTERVAL_TIME_AI_SECONDS: "600"
   ```

2. Deploy the application as a Pod or Deployment referencing the ConfigMap.

## Error Handling
- All exceptions are logged with detailed stack traces.
- Errors in processing specific data entries do not interrupt the overall pipeline.

## Development
### Adding New Features
- Add modules under `app/modules`.
- Update the pipeline logic in `main.py` as necessary.

