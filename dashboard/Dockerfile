FROM python:3.12-slim

# Non-root user
RUN groupadd -r dashboard && useradd -r -g dashboard -d /app -s /sbin/nologin dashboard

WORKDIR /app

# Install Python dependencies only
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas \
    pyarrow \
    numpy \
    scikit-learn \
    xgboost \
    econml \
    slowapi \
    matplotlib \
    reportlab

# Copy application code
COPY dashboard/ /app/dashboard/
COPY column_maps.py causal_core.py build_causal_forest.py build_attribution_q.py /app/

# Copy data files into the container (not mounted — fully isolated)
COPY event_index.json /app/data/event_index.json
COPY event_well_links.parquet /app/data/event_well_links.parquet
COPY well_day_panel.parquet /app/data/well_day_panel.parquet
COPY tmle_dose_response.csv /app/data/tmle_dose_response_365d_20260412_203226.csv
COPY tmle_shift.csv /app/data/tmle_shift_365d_20260412_203226.csv
COPY tmle_mediation.csv /app/data/tmle_mediation_365d_20260412_203226.csv
COPY cf_cate_*.pkl /app/data/
COPY q_attribution_*.pkl /app/data/

# Startup wrapper that patches paths then launches uvicorn
COPY entrypoint.py /app/entrypoint.py

# Own everything by dashboard user
RUN chown -R dashboard:dashboard /app

USER dashboard
EXPOSE 8765

# Read-only filesystem except /tmp for uvicorn
CMD ["python", "/app/entrypoint.py"]
