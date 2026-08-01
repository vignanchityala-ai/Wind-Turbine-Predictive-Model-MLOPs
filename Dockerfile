# Wind Turbine Early Fault Detection API
#
# NOTE: this image was written and reviewed carefully but NOT build-tested —
# this sandbox has no Docker daemon available. Run `docker build` locally
# before deploying anywhere; the most likely thing to need adjusting is
# pinned versions in requirements.txt if a transitive dependency has moved.
#
# Build:
#   docker build -t wind-turbine-api .
# Run (mount your trained models if you didn't bake them in — see below):
#   docker run -p 8000:8000 -e API_KEY=your-secret-here wind-turbine-api
#
# This bakes models/ into the image, which is simplest to start with: every
# deploy is a self-contained, reproducible artifact. The natural next step
# once this is running is to externalize model loading (mount a volume, or
# pull from S3/Azure Blob/artifact registry at container startup) so
# retraining doesn't require a full image rebuild — worth doing once you
# have more than a couple of turbines' models to manage.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# No API_KEY baked in on purpose — pass it at `docker run` time with -e,
# or via your platform's secrets manager. Running with no API_KEY set
# disables auth, which is fine for local testing but not for anything
# reachable outside your own machine.
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
