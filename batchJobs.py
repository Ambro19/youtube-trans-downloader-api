# backend/BatchJobs.py
"""
Drop-in router alias for batch endpoints.

Why this file? Some projects prefer importing `BatchJobs` instead of `batch`.
This module simply re-exports the existing FastAPI router from `batch.py` so
`main.py` can do either of the following:

    from BatchJobs import router as batch_router
    # or
    from batch import router as batch_router

Both give you the same endpoints:
  POST   /batch/submit
  GET    /batch/jobs
  GET    /batch/jobs/{job_id}
  POST   /batch/jobs/{job_id}/retry_failed
  DELETE /batch/jobs/{job_id}
"""
from batch import router as router

__all__ = ["router"]


#------------End bachjJobs Module-----------