# Wind Turbine Predictive Maintenance - Project Q&A

This document serves as a knowledge base and FAQ, capturing important architectural decisions, MLOps processes, and data science strategies discussed during the development of this project. It is ideal for interview preparation or presentations.

---

### 1. Why does uploading a full historical CSV (like `comma_0.csv`) result in so many anomalies (e.g., 69 events) and a noisy graph?
**Answer:** 
The uploaded file contains over a year of historical training data (~54,000 readings spanning 13 months). Squeezing 13 months of data into a single chart makes it look extremely noisy. Furthermore, finding 69 distinct anomaly periods over the course of a year is entirely expected for a real turbine experiencing minor issues (sensor dropouts, temporary icing, maintenance mode). In a true production environment, the system only scores small rolling windows (e.g., the last 6 to 24 hours), so the dashboard would look smooth, clean, and highly readable. To test the system properly, you should evaluate just the "Prediction Window" (the last few weeks leading up to a failure).

### 2. How is data prepared for training, and what columns are selected?
**Answer:** 
We use a strict, physics-informed feature engineering and selection pipeline:
1. **Filtering:** We strictly filter for `train_test == 'train'` and `status_type_id == 0` so the model learns from a pristine, healthy baseline.
2. **Sensor Normalization:** Each turbine's raw sensor values are Z-score normalized against *its own* historical mean and standard deviation. 
3. **Physics Engineering:** We calculate the expected power (Power Curve) and create the `power_residual` (expected minus actual).
4. **Rolling Stats:** We compute causal 1h, 3h, and 6h rolling windows. *Crucially*, for angle sensors (like nacelle direction), we use **Circular Statistics** (Sine/Cosine) to avoid wrap-around bugs (e.g., 359° and 1° averaging to 180°).
5. **Aggressive Feature Selection:** `src/feature_selection.py` drops engineered columns that are >50% `NaN`, near-constant (zero-variance), or highly correlated (Pearson > 0.95) to reduce dimensionality and memory footprint.

### 3. Why is `asset_id` explicitly removed from the training data?
**Answer:** 
If we left `asset_id` (e.g., "Turbine WT03") in the training data, the model would cheat. Instead of learning the physical signs of degradation, it would memorize behaviors tied to specific turbine IDs. Furthermore, if you deployed that model to a brand-new wind farm, it would break because it wouldn't recognize the new turbine IDs. By removing it and relying entirely on turbine-specific Z-score normalization, we force the model to be completely "turbine-agnostic" and highly generalizable.

### 4. How does live data flow into the system for predictions?
**Answer:** 
The FastAPI server is completely **stateless**. It does not connect directly to the SCADA system, and it does not remember past data. Instead, an integration layer makes an HTTP POST request to the API every 10 minutes. 
Because the model relies heavily on historical rolling statistics, every API call must send a JSON payload containing a **Rolling Window** of the last 6 hours of data (~36 readings). The API instantly calculates the rolling averages and physics features, scores the single *most recent* reading, and returns a lightweight JSON response. This stateless design allows us to easily scale the API behind a load balancer without dealing with synchronized memory.

### 5. Does the API calculate data drift (PSI) on every single prediction call?
**Answer:** 
No, doing so is an MLOps anti-pattern. Population Stability Index (PSI) measures the shift between two statistical distributions. You cannot calculate a reliable distribution from a 36-reading API payload; it would cause massive mathematical noise and severely increase API latency. 
Instead, data drift monitoring (`src/monitoring/drift.py`) is designed as an asynchronous background scanner (e.g., a weekly Apache Airflow job). It pulls large batches of aggregated production data to accurately measure statistical shifts against the training baseline.

### 6. Where and how do we store the API-processed data for that drift check?
**Answer:** 
In the current MVP phase, the API is ephemeral—it scores the payload and throws the data away. 
However, in a production Cloud Architecture (Phase 10), we implement **Inference Logging** (Data Capture). The FastAPI `predict` endpoint uses a Background Task to asynchronously write the incoming JSON features and the outgoing anomaly score to a streaming service (like **Azure Event Hubs** or **Kafka**). That stream automatically dumps the logs into a data lake (like **Azure Blob Storage**) in compressed Parquet format. The weekly drift scanner then pulls from this data lake.

### 7. How does the system handle Continuous Training (CT) and Continuous Deployment (CD) when new data arrives?
**Answer:** 
We built an automated, closed-loop MLOps pipeline:
1. **DVC Tracking:** New raw CSVs are tracked with Data Version Control (DVC). Pushing the new `dvc.lock` file to GitHub triggers the pipeline.
2. **Automated Retraining:** GitHub Actions (CI/CD) spins up a runner, pulls the new data, and runs `dvc repro`. This automatically executes the data ingestion, cleaning, feature engineering, and model training scripts in sequence.
3. **MLflow Registry:** During training, MLflow silently logs the hyperparameters and evaluation metrics (CARE scores) and registers the newly minted `.joblib` model.
4. **Docker Deployment:** Once tests pass, the CI pipeline builds a fresh Docker image containing the new model. The production server pulls this image and performs a rolling restart of the FastAPI container, serving the new model seamlessly.
