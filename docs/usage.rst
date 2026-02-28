=====
Usage
=====

Web Application
---------------

To run the interactive web interface for pneumonia detection:

.. code-block:: console

    $ python app.py

Then visit ``http://localhost:8000`` in your browser.

Training Pipeline
-----------------

To run the end-to-end training pipeline (Ingestion -> Transformation -> Training -> Evaluation -> Pusher):

.. code-block:: console

    $ python pneumonia_classifier/pipeline/training_pipeline.py

Experiment Tracking
-------------------

To view experiment logs and metrics in MLflow:

.. code-block:: console

    $ mlflow server --backend-store-uri file:///notebooks/mlruns --port 5001

