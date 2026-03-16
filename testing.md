# AgentML Testing Guide

This document covers how to run, test, and verify all components of the AgentML platform.

**Last Updated:** December 2025

---

## Quick Start

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | **3.11.x** (required) | AutoGluon does NOT support Python 3.13 yet |
| Node.js | 20+ | For frontend |
| Docker Desktop | Latest | For PostgreSQL/Redis |
| PostgreSQL | 15+ | Running on port 5432 |

> **⚠️ IMPORTANT: Python 3.11 is REQUIRED**
>
> AutoGluon (our ML framework) has dependencies that do not work with Python 3.13.
> If you have Python 3.13, you must install Python 3.11 separately.

### Python 3.11 Setup (Windows)

If you have Python 3.13 installed, follow these steps to set up Python 3.11:

```powershell
# 1. Download and install Python 3.11.9 from:
#    https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
#    (Don't uninstall Python 3.13 - you can have multiple versions)

# 2. Create a Python 3.11 virtual environment
cd backend
py -3.11 -m venv .venv311

# 3. Activate the new environment
.venv311\Scripts\activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install all dependencies (this takes 5-10 minutes)
pip install -r requirements.txt

# 6. Verify installation
python -c "import autogluon.tabular; print('AutoGluon tabular OK')"
python -c "import autogluon.timeseries; print('AutoGluon timeseries OK')"
python -c "import autogluon.multimodal; print('AutoGluon multimodal OK')"
```

### Switching Virtual Environments

```powershell
# Exit current venv (if any)
deactivate

# Activate the Python 3.11 venv
cd backend
.venv311\Scripts\activate

# Verify you're using Python 3.11
python --version
# Should show: Python 3.11.x
```

### Start Everything

You need **4 services** running for full functionality:

```bash
# 1. Start PostgreSQL and Redis via Docker
docker-compose up -d postgres redis

# Or start Redis standalone if not using docker-compose:
docker run -d -p 6379:6379 --name redis redis:alpine

# 2. Backend API (Terminal 1)
cd backend
.venv311\Scripts\activate    # Windows - Python 3.11 venv
# source .venv311/bin/activate   # Linux/Mac
alembic upgrade head         # Apply database migrations
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 3. Celery Worker (Terminal 2) - REQUIRED for background tasks
cd backend
.venv311\Scripts\activate
celery -A app.core.celery_app worker --loglevel=info --pool=solo

# 4. Frontend (Terminal 3)
cd frontend
npm run dev
```

### Quick Start Script (Windows)

Use the provided batch script to start both backend services at once:

```powershell
cd backend
.\start_services.bat
```

This starts:
- Celery worker in a new window
- FastAPI server in the current window

### Access Points
| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Docs (ReDoc) | http://localhost:8000/redoc |

---

## Running Tests

### Backend Tests (pytest)

```bash
cd backend
.venv311\Scripts\activate   # Use Python 3.11 venv

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_projects.py -v

# Run specific test class
pytest tests/test_projects.py::TestProjectsCRUD -v

# Run specific test
pytest tests/test_projects.py::TestProjectsCRUD::test_create_project -v

# Run with coverage report
pytest --cov=app --cov-report=html

# Run tests matching a pattern
pytest -k "project" -v
```

### Frontend Tests (lint only - no unit tests yet)

```bash
cd frontend

# Run ESLint
npm run lint

# Build check (ensures TypeScript compiles)
npm run build
```

---

# Phase 0: Project Setup ✅

## What Was Done

Phase 0 established the foundational infrastructure for the entire platform:

### Backend Setup
- **FastAPI application** with CORS middleware
- **Health check endpoint** (`/health`) for monitoring
- **Root endpoint** (`/`) returning API info
- **Configuration management** via Pydantic settings (`.env` support)
- **Database setup** with SQLAlchemy ORM and PostgreSQL
- **Celery configuration** for future async tasks
- **Docker Compose** for PostgreSQL, Redis, and all services

### Frontend Setup
- **React 19 + TypeScript** application with Vite
- **React Router** for page navigation
- **Basic layout** with navigation header
- **Home page** that checks backend health status
- **Placeholder pages** for Projects, Experiments, Models

### Developer Experience
- **pytest** configured for backend testing
- **ESLint** configured for frontend linting
- **Black, Ruff, MyPy** for Python code quality
- **Docker Compose** for one-command environment setup

## How It Connects

```
Phase 0 (Setup) ──────► Phase 1 (Data Model)
     │                       │
     │ Provides:             │ Uses:
     │ - FastAPI app         │ - Add routes to app
     │ - Database connection │ - Create ORM models
     │ - Test framework      │ - Write API tests
     │ - Docker environment  │ - Run against Postgres
```

## Test Files
- `tests/test_health.py` - Health check and root endpoint tests

## Running Phase 0 Tests
```bash
cd backend
pytest tests/test_health.py -v
```

---

# Phase 1: Core Data Model & Basic API ✅

## What Was Done

Phase 1 built the complete data model and REST API for all core entities:

### Database Models (SQLAlchemy ORM)

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `Project` | ML project container | name, description, task_type, status |
| `DataSource` | Data connection | type (file/db/s3/api), config_json, schema_summary |
| `DatasetSpec` | Dataset definition | target_column, feature_columns, filters |
| `Experiment` | ML experiment | status, primary_metric, metric_direction |
| `Trial` | Single experiment run | variant_name, automl_config, metrics_json |
| `ModelVersion` | Trained model | model_type, artifact_location, status |
| `RetrainingPolicy` | Auto-retrain rules | policy_type, schedule_cron, thresholds |

### Database Migrations (Alembic)
- Initial migration creating all 7 tables
- Proper foreign keys with CASCADE delete
- Indexes on frequently queried columns
- PostgreSQL-specific types (UUID, JSONB) with SQLite fallbacks for testing

### REST API Endpoints

**Full CRUD for all entities:**
- Projects: create, list, get, update, delete
- Data Sources: create, list, get, update, delete (scoped to project)
- Dataset Specs: create, list, get, update, delete (scoped to project)
- Experiments: create, list, get, update, delete (scoped to project)
- Trials: create, list, get (scoped to experiment)
- Models: create, list, get, delete, **promote** (scoped to project)

**Special Features:**
- Model promotion workflow (trained → candidate → shadow → production)
- Auto-retire old production model when new one promoted
- Filter models by status

### Pydantic Schemas
- Request validation schemas (Create, Update)
- Response schemas with proper serialization
- Cross-database compatible (PostgreSQL + SQLite for tests)

## How It Connects

```
Phase 0 (Setup) ──► Phase 1 (Data Model) ──► Phase 2 (Data Connection)
                         │                        │
                         │ Provides:              │ Will Use:
                         │ - Project entity       │ - Create projects
                         │ - DataSource entity    │ - Upload files to data sources
                         │ - DatasetSpec entity   │ - Auto-generate dataset specs
                         │ - Experiment entity    │ - Create experiments from specs
                         │ - Trial entity         │ - Run trials with AutoML
                         │ - ModelVersion entity  │ - Store trained models
```

**Phase 1 is the backbone** - every future phase builds on these entities:
- Phase 2 will populate `DataSource.schema_summary` with analyzed schemas
- Phase 3 will update `Experiment.status` and `Trial.metrics_json` during training
- Phase 4 will use `ModelVersion.status` for deployment decisions

## Test Files & Descriptions

### `tests/test_projects.py` - Project CRUD (8 tests)

| Test | Description |
|------|-------------|
| `test_create_project` | Creates project with name, description, task_type; verifies all fields saved and UUID assigned |
| `test_create_project_minimal` | Creates project with only required `name` field; verifies optional fields default correctly |
| `test_list_projects_empty` | Lists projects on empty database; verifies returns empty array `[]` |
| `test_list_projects` | Creates 2 projects, lists all; verifies both returned |
| `test_get_project` | Creates project, retrieves by ID; verifies all fields match |
| `test_get_project_not_found` | Requests non-existent UUID; expects 404 error |
| `test_update_project` | Creates project, updates name/description; verifies changes persisted |
| `test_delete_project` | Creates project, deletes it; verifies subsequent GET returns 404 |

### `tests/test_data_sources.py` - Data Source CRUD (6 tests)

| Test | Description |
|------|-------------|
| `test_create_data_source` | Creates data source with name, type, config_json under a project; verifies project_id linkage |
| `test_create_data_source_invalid_project` | Creates data source with fake project UUID; expects 404 error |
| `test_list_data_sources` | Creates 2 data sources for project; verifies both returned in list |
| `test_get_data_source` | Creates data source, retrieves by ID; verifies all fields match |
| `test_update_data_source` | Creates data source, updates name and config_json; verifies changes persisted |
| `test_delete_data_source` | Creates data source, deletes it; verifies subsequent GET returns 404 |

### `tests/test_dataset_specs.py` - Dataset Spec CRUD (6 tests)

| Test | Description |
|------|-------------|
| `test_create_dataset_spec` | Creates spec with target_column, feature_columns, data_sources_json; verifies all fields saved |
| `test_create_dataset_spec_minimal` | Creates spec with only `name`; verifies target_column=None, feature_columns=[] |
| `test_list_dataset_specs` | Creates 2 specs for project; verifies both returned in list |
| `test_get_dataset_spec` | Creates spec, retrieves by ID; verifies all fields match |
| `test_update_dataset_spec` | Creates spec, updates name/target/features; verifies changes persisted |
| `test_delete_dataset_spec` | Creates spec, deletes it; verifies subsequent GET returns 404 |

### `tests/test_experiments.py` - Experiment & Trial CRUD (9 tests)

| Test | Description |
|------|-------------|
| `test_create_experiment` | Creates experiment with name, primary_metric, metric_direction; verifies status defaults to "draft" |
| `test_create_experiment_minimal` | Creates experiment with only `name`; verifies optional fields default correctly |
| `test_list_experiments` | Creates 2 experiments for project; verifies both returned in list |
| `test_get_experiment` | Creates experiment, retrieves by ID; verifies all fields match |
| `test_update_experiment` | Creates experiment, updates status to "running"; verifies change persisted |
| `test_delete_experiment` | Creates experiment, deletes it; verifies subsequent GET returns 404 |
| `test_create_trial` | Creates trial under experiment with variant_name, automl_config; verifies experiment_id linkage |
| `test_list_trials` | Creates 2 trials for experiment; verifies both returned in list |
| `test_get_trial` | Creates trial, retrieves by ID; verifies all fields match |

### `tests/test_models.py` - Model Version CRUD & Promotion (9 tests)

| Test | Description |
|------|-------------|
| `test_create_model_version` | Creates model with trial_id, model_type, metrics_json, artifact_location; verifies status defaults to "trained" |
| `test_create_model_version_minimal` | Creates model with only required fields; verifies optional fields default correctly |
| `test_list_model_versions` | Creates 2 models for project; verifies both returned in list |
| `test_get_model_version` | Creates model, retrieves by ID; verifies all fields match |
| `test_promote_model_to_candidate` | Creates model, promotes to "candidate"; verifies status changed |
| `test_promote_model_to_production` | Creates model, promotes to "production"; verifies status changed |
| `test_promote_new_model_retires_old_production` | Promotes model A to production, then promotes model B to production; verifies A auto-retired to "retired" |
| `test_filter_models_by_status` | Creates models with different statuses, filters by status; verifies only matching models returned |
| `test_delete_model_version` | Creates model, deletes it; verifies subsequent GET returns 404 |

## Coverage Summary

| Entity | Create | Read | Update | Delete | Special |
|--------|--------|------|--------|--------|---------|
| Project | ✅ | ✅ | ✅ | ✅ | - |
| DataSource | ✅ | ✅ | ✅ | ✅ | - |
| DatasetSpec | ✅ | ✅ | ✅ | ✅ | - |
| Experiment | ✅ | ✅ | ✅ | ✅ | - |
| Trial | ✅ | ✅ | - | - | - |
| ModelVersion | ✅ | ✅ | - | ✅ | Promote, Filter by status, Auto-retire |

**Total Phase 1 Tests:** 40

## Running Phase 1 Tests
```bash
cd backend

# All Phase 1 tests
pytest tests/test_projects.py tests/test_data_sources.py tests/test_dataset_specs.py tests/test_experiments.py tests/test_models.py -v

# Just one entity
pytest tests/test_projects.py -v

# Test model promotion specifically
pytest tests/test_models.py::TestModelVersionsCRUD::test_promote_new_model_retires_old_production -v
```

---

# Phase 2: Data Source Handling & Dataset Building ✅

## What Was Done

Phase 2 adds file upload, schema analysis, and dataset building capabilities:

### File Upload (Multi-File Type Support)
- [x] `POST /projects/{id}/data-sources/upload` - Upload files (multiple types supported)
- [x] Store files in configurable upload directory
- [x] Update `DataSource.config_json` with file path, delimiter, size
- [x] Automatic schema analysis on upload

#### Supported File Types

| Extension | Format | Notes |
|-----------|--------|-------|
| `.csv` | CSV | Comma, semicolon, or tab delimited |
| `.xlsx`, `.xls` | Excel | Reads first sheet by default |
| `.json` | JSON | Array of objects or object with data array |
| `.parquet` | Parquet | Columnar format for large datasets |
| `.txt` | Text | Tab-delimited by default |
| `.docx` | Word | Extracts tables from document |

### Schema Analysis Service (`app/services/schema_analyzer.py`)
- [x] `SchemaAnalyzer` class for analyzing files and DataFrames
- [x] Support for all file types: CSV, Excel, JSON, Parquet, Text, Word
- [x] Extract column names, pandas dtypes
- [x] Infer semantic types: numeric, categorical, datetime, text, boolean
- [x] Calculate null counts and percentages
- [x] Generate statistics per type:
  - **Numeric:** min, max, mean, median, std
  - **Categorical:** top values, mode
  - **Datetime:** min, max dates
- [x] Results stored in `DataSource.schema_summary`

### Dataset Builder Service (`app/services/dataset_builder.py`)
- [x] `DatasetBuilder` class for constructing DataFrames from DatasetSpecs
- [x] `build_dataset_from_spec(dataset_spec_id)` - Build DataFrame from spec
- [x] Load data from file_upload data sources
- [x] Support multiple data sources (concatenation)
- [x] Column selection (target + features)
- [x] Filter support:
  - Range filters: `{"column": {"min": X, "max": Y}}`
  - Value list: `{"column": {"in": [values]}}`
  - Exact match: `{"column": "value"}`
- [x] `get_dataset_info()` - Get metadata without loading data

### Configuration Updates
- [x] `upload_dir` setting for file storage location (default: `./uploads`)
- [x] `max_upload_size_mb` setting (default: 100MB)

## How It Connects

```
Phase 1 (Data Model) ──► Phase 2 (Data Connection) ──► Phase 3 (Experiments)
       │                        │                           │
       │ Provides:              │ Provides:                 │ Will Use:
       │ - DataSource model     │ - File upload endpoint    │ - DatasetBuilder.build_dataset_from_spec()
       │ - DatasetSpec model    │ - Schema analysis         │ - Feature columns from schema
       │                        │ - Dataset building        │ - Target column
       │                        │ - Column statistics       │ - AutoML on built DataFrame
```

**Phase 2 enables real data flow** - future phases can now:
- Phase 3 will call `build_dataset_from_spec()` to get training data
- Phase 3 will use `schema_summary` to inform LLM about column types
- Phase 3 will pass built DataFrame to AutoML for training

## Test Files & Descriptions

### `tests/test_file_upload.py` - File Upload Endpoint (11 tests)

| Test | Description |
|------|-------------|
| `test_upload_csv_file` | Uploads a CSV file and verifies: data source created with correct name/type, config_json contains file path and delimiter, schema_summary populated with row/column counts |
| `test_upload_csv_with_custom_name` | Uploads with optional `name` parameter, verifies custom name is used instead of filename while original filename preserved in config |
| `test_upload_unsupported_file_fails` | Attempts to upload an unsupported file type (.xyz), expects 400 error with "Unsupported file type" message |
| `test_upload_to_nonexistent_project_fails` | Uploads to fake UUID project, expects 404 error |
| `test_upload_schema_analysis_columns` | Uploads CSV with mixed types (id, name, price, is_active, created_at), verifies schema correctly identifies numeric columns and calculates min/max stats |
| `test_upload_with_semicolon_delimiter` | Uploads semicolon-delimited CSV with `delimiter=";"`, verifies columns parsed correctly (not treating semicolons as data) |
| `test_upload_excel_file` | Uploads .xlsx Excel file, verifies schema analysis works correctly for Excel format |
| `test_upload_json_file` | Uploads .json file with array of objects, verifies schema analysis detects columns correctly |
| `test_upload_parquet_file` | Uploads .parquet columnar file, verifies schema analysis works for Parquet format |
| `test_upload_txt_file` | Uploads .txt tab-delimited file, verifies parsing with tab delimiter |
| `test_upload_word_file` | Uploads .docx Word file containing tables, verifies table extraction and schema analysis |

### `tests/test_schema_analyzer.py` - Schema Analysis Service (24 tests)

| Test | Description |
|------|-------------|
| `test_analyze_dataframe_basic` | Analyzes 5-row DataFrame, verifies row_count=5, column_count=6, sample_rows=5, columns array has 6 entries |
| `test_analyze_dataframe_column_names` | Verifies all column names (id, name, age, salary, department, is_active) captured in analysis output |
| `test_analyze_numeric_column` | Checks numeric column "age" has inferred_type="numeric" and stats: min=25, max=35, mean, median, std |
| `test_analyze_categorical_column` | Checks "department" column detected as categorical with top_values dict, mode value, unique_count=3 |
| `test_analyze_boolean_column` | Checks "is_active" column detected as boolean type with unique_count=2 |
| `test_analyze_csv_file` | Reads actual CSV file from disk (not DataFrame), verifies row_count, column_count, columns array match expected |
| `test_analyze_csv_with_delimiter` | Creates semicolon-delimited CSV file, analyzes with delimiter=";", verifies 3 columns detected correctly |
| `test_analyze_null_values` | Creates DataFrame with nulls, verifies null_count=0 for complete column, null_count=2 and null_percentage=40.0 for column with nulls |
| `test_analyze_datetime_column` | Creates column with date strings "2024-01-01", verifies inferred_type="datetime" and min/max date stats present |
| `test_analyze_text_column` | Creates 100 unique description strings (high cardinality), verifies inferred_type="text" (not categorical) |
| `test_analyze_file_not_found` | Calls analyze_csv with non-existent path, expects FileNotFoundError raised |
| `test_empty_dataframe` | Analyzes DataFrame with 0 rows, verifies row_count=0, column_count=2 (columns still detected) |
| `test_unique_count` | Tests unique_count accuracy: column with all same values → 1, column with all different → 5 |

#### Multi-File Type Tests (`TestSchemaAnalyzerFileTypes`)

| Test | Description |
|------|-------------|
| `test_analyze_excel_file` | Creates .xlsx Excel file, verifies schema analysis extracts correct row/column counts and column names |
| `test_analyze_excel_xls_file` | Creates .xls (legacy Excel) file, verifies backward compatibility with older Excel format |
| `test_analyze_json_array_file` | Creates JSON file with array of objects `[{...}, {...}]`, verifies correct parsing and column detection |
| `test_analyze_json_with_data_key` | Creates JSON file with structure `{"data": [...]}`, verifies nested data array extraction |
| `test_analyze_parquet_file` | Creates .parquet columnar file using pandas, verifies efficient reading and schema detection |
| `test_analyze_txt_file` | Creates tab-delimited .txt file, verifies default tab delimiter handling |
| `test_analyze_txt_with_custom_delimiter` | Creates pipe-delimited .txt file, verifies custom delimiter parameter works |
| `test_analyze_word_file_with_table` | Creates .docx Word document with embedded table, verifies table extraction to DataFrame |
| `test_analyze_word_file_no_table` | Creates .docx Word document without tables, expects ValueError with "No tables found" message |
| `test_analyze_unsupported_extension` | Attempts to analyze .xyz file, expects ValueError with "Unsupported file type" message |
| `test_analyze_file_detects_extension` | Verifies `analyze_file()` correctly auto-detects file type from extension and routes to appropriate parser |

### `tests/test_dataset_builder.py` - Dataset Builder Service (10 tests)

| Test | Description |
|------|-------------|
| `test_build_dataset_from_spec` | Creates DatasetSpec with target="salary", features=["age","department"], builds DataFrame, verifies only selected columns present (id/name excluded) |
| `test_build_dataset_all_columns` | Creates DatasetSpec with no column selection, verifies all 5 original columns returned |
| `test_build_dataset_with_filter_range` | Applies filter `{"age": {"min": 28, "max": 33}}`, verifies only 3 rows returned where 28 ≤ age ≤ 33 |
| `test_build_dataset_with_filter_in_list` | Applies filter `{"department": {"in": ["Engineering", "Sales"]}}`, verifies only matching rows returned |
| `test_build_dataset_with_exact_value_filter` | Applies filter `{"department": "Engineering"}`, verifies only Engineering rows returned (2 rows) |
| `test_build_dataset_spec_not_found` | Calls with fake UUID, expects ValueError with "not found" message |
| `test_build_dataset_no_sources` | Creates spec with empty data_sources_json, expects ValueError with "no data sources" message |
| `test_build_dataset_source_not_found` | Creates spec referencing non-existent data source UUID, expects ValueError with "DataSource.*not found" |
| `test_get_dataset_info` | Calls get_dataset_info(), verifies returns metadata (spec id, name, target, features, sources) without loading actual data |
| `test_build_dataset_multiple_sources` | Creates 2 CSV files (5 + 2 rows), 2 data sources, spec referencing both, verifies concatenated result has 7 rows |

**Total Phase 2 Tests:** 45
**Total Tests (All Phases):** 85

## Running Phase 2 Tests
```bash
cd backend

# All Phase 2 tests
pytest tests/test_file_upload.py tests/test_schema_analyzer.py tests/test_dataset_builder.py -v

# Just file upload
pytest tests/test_file_upload.py -v

# Just schema analysis
pytest tests/test_schema_analyzer.py -v

# Just dataset building
pytest tests/test_dataset_builder.py -v

# All tests (Phase 0 + 1 + 2)
pytest tests/ -v
```

## API Endpoint Added

### File Upload
```
POST /projects/{id}/data-sources/upload   # Upload data file
     - Form data: file (required), name (optional), delimiter (optional, default ",")
     - Supported formats: CSV, Excel (.xlsx/.xls), JSON, Parquet, Text (.txt), Word (.docx)
     - Returns: DataSourceResponse with schema_summary populated
```

## Example Usage

### Upload Files via curl

```bash
# CSV file
curl -X POST http://localhost:8000/projects/{project_id}/data-sources/upload \
  -F "file=@sales_data.csv" \
  -F "name=Sales Data 2024" \
  -F "delimiter=,"

# Excel file
curl -X POST http://localhost:8000/projects/{project_id}/data-sources/upload \
  -F "file=@report.xlsx" \
  -F "name=Quarterly Report"

# JSON file
curl -X POST http://localhost:8000/projects/{project_id}/data-sources/upload \
  -F "file=@customers.json" \
  -F "name=Customer Data"

# Parquet file
curl -X POST http://localhost:8000/projects/{project_id}/data-sources/upload \
  -F "file=@large_dataset.parquet" \
  -F "name=Large Dataset"
```

### Upload via Swagger UI
1. Open http://localhost:8000/docs
2. Find `POST /projects/{project_id}/data-sources/upload`
3. Click "Try it out"
4. Enter project_id
5. Click "Choose File" and select a supported file (CSV, Excel, JSON, Parquet, TXT, DOCX)
6. Optionally set name and delimiter (delimiter only applies to CSV/TXT)
7. Click "Execute"

### Build Dataset Programmatically
```python
from app.services import DatasetBuilder
from app.core.database import get_db

db = next(get_db())
builder = DatasetBuilder(db)

# Build DataFrame from a DatasetSpec
df = builder.build_dataset_from_spec(dataset_spec_id)

# Get info without loading data
info = builder.get_dataset_info(dataset_spec_id)
```

---

# Phase 3: AutoML Integration (MVP) ✅

## What Was Done

Phase 3 enables automated ML experiment execution using the full AutoGluon suite:

### AutoML Library (Full Task Type Support)
- [x] AutoGluon integrated with all available modules:
  - `autogluon.tabular` - Tabular data (classification, regression, quantile)
  - `autogluon.timeseries` - Time series forecasting
  - `autogluon.multimodal` - Mixed data (text + tabular + images)
- [x] Runner classes in `app/services/automl_runner.py`:
  - `TabularRunner` - Binary/multiclass classification, regression, quantile regression
  - `TimeSeriesRunner` - Time series forecasting
  - `MultiModalRunner` - Multimodal classification and regression
- [x] `AutoMLResult` dataclass for structured results
- [x] `get_runner_for_task()` factory function for automatic runner selection
- [x] Metric mapping (rmse, accuracy, auc, f1, mase, pinball_loss, etc.)

### Supported Task Types

| Task Type | Runner | Description | Default Metric |
|-----------|--------|-------------|----------------|
| `binary` | TabularRunner | Binary classification | roc_auc |
| `multiclass` | TabularRunner | Multi-class classification | accuracy |
| `regression` | TabularRunner | Standard regression | rmse |
| `quantile` | TabularRunner | Quantile regression (predict percentiles) | pinball_loss |
| `timeseries_forecast` | TimeSeriesRunner | Time series forecasting | MASE |
| `multimodal_classification` | MultiModalRunner | Text + tabular + images classification | accuracy |
| `multimodal_regression` | MultiModalRunner | Text + tabular + images regression | rmse |
| `classification` | TabularRunner | Legacy alias (maps to binary) | roc_auc |

### How Runner Selection Works

When you run an experiment (`POST /experiments/{id}/run`), the system:

1. **Reads the project's `task_type`** from the database
2. **Maps it to a runner** using `get_runner_for_task()` in [automl_runner.py](backend/app/services/automl_runner.py)
3. **Executes training** with the appropriate AutoGluon module

```
Project.task_type  →  get_runner_for_task()  →  Runner Class  →  AutoGluon Module
─────────────────────────────────────────────────────────────────────────────────
binary/multiclass  →  TabularRunner          →  autogluon.tabular
regression/quantile →  TabularRunner          →  autogluon.tabular
timeseries_forecast →  TimeSeriesRunner       →  autogluon.timeseries
multimodal_*        →  MultiModalRunner       →  autogluon.multimodal
```

### Data File Requirements by Task Type

#### Tabular Tasks (binary, multiclass, regression, quantile)

**Required columns:**
- Target column (specified in DatasetSpec)
- Feature columns (numeric, categorical, or mixed)

**Example CSV:**
```csv
id,age,income,department,is_churned
1,25,50000,Sales,0
2,35,75000,Engineering,1
3,28,60000,Marketing,0
```

**No special configuration needed** - AutoGluon auto-detects column types.

---

#### Time Series Forecasting (timeseries_forecast)

**Required columns:**
- **Timestamp column** - dates/datetimes
- **Target column** - the value to forecast
- **ID column** (optional) - for multiple time series

**Example CSV (single series):**
```csv
date,sales
2024-01-01,100
2024-01-02,120
2024-01-03,95
2024-01-04,130
```

**Example CSV (multiple series):**
```csv
store_id,date,sales
store_A,2024-01-01,100
store_A,2024-01-02,120
store_B,2024-01-01,200
store_B,2024-01-02,180
```

**Required configuration in `experiment_plan_json`:**
```json
{
  "automl_config": {
    "prediction_length": 7,
    "time_column": "date",
    "id_column": "store_id",
    "freq": "D"
  }
}
```

| Config | Required | Description |
|--------|----------|-------------|
| `prediction_length` | Yes | Number of future steps to forecast |
| `time_column` | Yes | Name of the timestamp column |
| `id_column` | No | Name of series ID column (for multiple series) |
| `freq` | No | Time frequency: "D" (daily), "H" (hourly), "W" (weekly), etc. |

---

#### Multimodal Tasks (multimodal_classification, multimodal_regression)

**Supported column types (auto-detected):**
- **Text columns** - long strings, descriptions, reviews
- **Numeric columns** - integers, floats
- **Categorical columns** - limited set of values
- **Image columns** - file paths to images

**Example CSV (sentiment classification):**
```csv
review_text,rating,category,sentiment
"This product is amazing! Fast shipping.",5,electronics,positive
"Terrible quality, broke after one day.",1,electronics,negative
"Decent for the price, nothing special.",3,clothing,neutral
```

**Example CSV (with images):**
```csv
image_path,description,price,category
/images/product1.jpg,"Blue wireless headphones",49.99,electronics
/images/product2.jpg,"Cotton t-shirt size M",19.99,clothing
```

**No special configuration needed** - AutoGluon automatically:
- Detects text columns and applies NLP models
- Detects numeric/categorical columns for tabular processing
- Combines all modalities into an ensemble

---

### Installing AutoGluon Modules

The full AutoGluon suite is large. Install only what you need:

```bash
cd backend
.\venv\Scripts\activate

# Tabular only (smallest, ~1GB)
pip install autogluon.tabular

# Add time series (~500MB additional)
pip install autogluon.timeseries

# Add multimodal (~2-3GB additional, includes PyTorch)
pip install autogluon.multimodal
```

**Note:** Tests use mocks and don't require AutoGluon installed.

### Background Task Execution (Celery)
- [x] Celery worker for async task execution
- [x] Redis as message broker and result backend
- [x] `run_automl_experiment_task` in `app/tasks/automl.py` - runs AutoML training
- [x] `generate_synthetic_dataset_task` in `app/tasks/synthetic_data.py` - generates synthetic data via LLM
- [x] Automatic status updates (pending → running → completed/failed)
- [x] Progress tracking via Celery task state updates
- [x] Error handling with experiment status set to FAILED on exceptions

### Experiment Workflow
- [x] `POST /experiments/{id}/run` - Queue experiment for execution
- [x] `POST /experiments/{id}/cancel` - Cancel running experiment
- [x] `GET /experiments/{id}` - Enhanced with trial count, best model, and metrics

### MVP Implementation
- [x] Single hard-coded split strategy: random 80/20 train/test
- [x] Default AutoML config: 5 minute time limit, medium_quality preset
- [x] Creates Trial record with metrics
- [x] Creates ModelVersion for best model with artifact path
- [x] Feature importances captured if available

### Configuration Settings
- [x] `artifacts_dir` - Directory for model artifacts (default: `./artifacts`)
- [x] `automl_time_limit` - Training time budget in seconds (default: 300)
- [x] `automl_presets` - AutoGluon quality preset (default: `medium_quality`)

## How It Connects

```
Phase 2 (Data) ──► Phase 3 (Experiments) ──► Phase 4 (Deployment)
      │                   │                        │
      │ Provides:         │ Provides:              │ Will Use:
      │ - Parsed data     │ - Trained models       │ - Best model selection
      │ - Feature info    │ - Performance metrics  │ - Model artifacts
      │                   │ - Model artifacts      │ - Serving config
```

**Phase 3 enables real ML training** - future phases can now:
- Phase 4 will use `artifact_location` to load models for prediction
- Phase 4 will use `metrics_json` to compare models
- Phase 4 will enable model promotion workflow

## Test Files & Descriptions

### `tests/test_automl.py` - AutoML Integration (29 tests)

#### TestAutoMLRunner - Basic AutoML functionality (4 tests)
| Test | Description |
|------|-------------|
| `test_automl_result_dataclass` | Tests AutoMLResult dataclass initialization with all fields |
| `test_run_experiment_binary` | Mocks AutoGluon to test binary classification experiment flow |
| `test_run_experiment_regression` | Mocks AutoGluon to test regression experiment flow |
| `test_metric_mapping` | Verifies metric name mapping (rmse → root_mean_squared_error, etc.) |

#### TestExperimentRunAPI - API endpoint tests (6 tests)
| Test | Description |
|------|-------------|
| `test_run_experiment_queues_task` | Tests that `/run` endpoint queues Celery task and returns task_id |
| `test_run_experiment_without_dataset_spec_fails` | Expects 400 error when experiment has no dataset_spec_id |
| `test_run_already_running_experiment_fails` | Expects 400 error when trying to run non-pending experiment |
| `test_cancel_pending_experiment` | Tests canceling a pending experiment updates status |
| `test_cancel_completed_experiment_fails` | Expects 400 error when canceling completed experiment |
| `test_run_experiment_not_found` | Expects 404 error for non-existent experiment |

#### TestExperimentDetailResponse - Enhanced responses (2 tests)
| Test | Description |
|------|-------------|
| `test_get_experiment_includes_trial_count` | Verifies enhanced GET response includes trial count |
| `test_get_experiment_includes_best_model` | Verifies completed experiment includes best model info |

#### TestExperimentTaskIntegration - End-to-end (1 test)
| Test | Description |
|------|-------------|
| `test_run_experiment_task_creates_records` | Integration test: verifies task creates Trial and ModelVersion |

#### TestTabularRunner - Tabular task types (2 tests)
| Test | Description |
|------|-------------|
| `test_run_quantile_experiment` | Tests quantile regression experiment with custom quantile levels |
| `test_tabular_runner_default_metrics` | Verifies default metric selection for each tabular task type |

#### TestTimeSeriesRunner - Time series forecasting (3 tests)
| Test | Description |
|------|-------------|
| `test_run_timeseries_experiment` | Tests time series forecasting experiment with prediction length |
| `test_timeseries_default_metric` | Verifies default metric (MASE) for time series |
| `test_timeseries_single_series_handling` | Tests automatic series ID assignment for single series |

#### TestMultiModalRunner - Multimodal tasks (3 tests)
| Test | Description |
|------|-------------|
| `test_run_multimodal_classification` | Tests multimodal classification with text + tabular data |
| `test_run_multimodal_regression` | Tests multimodal regression with mixed data types |
| `test_multimodal_binary_detection` | Tests binary vs multiclass detection for classification |

#### TestGetRunnerForTask - Factory function (4 tests)
| Test | Description |
|------|-------------|
| `test_get_tabular_runners` | Verifies TabularRunner returned for all tabular task types |
| `test_get_timeseries_runner` | Verifies TimeSeriesRunner returned for timeseries_forecast |
| `test_get_multimodal_runners` | Verifies MultiModalRunner returned for multimodal tasks |
| `test_unknown_task_type_raises` | Expects ValueError for unknown task types |

#### TestTaskTypeEnum - Task type definitions (2 tests)
| Test | Description |
|------|-------------|
| `test_all_task_types_defined` | Verifies all 8 task types are defined in TaskType enum |
| `test_task_type_string_values` | Verifies enum values are lowercase strings |

#### TestExperimentTaskRouting - Task routing (2 tests)
| Test | Description |
|------|-------------|
| `test_task_type_mapping` | Verifies all task types map correctly in experiment tasks |
| `test_legacy_classification_maps_to_binary` | Verifies legacy 'classification' maps to 'binary' |

**Total Phase 3 Tests:** 29
**Total Tests (All Phases):** 114

## Running Phase 3 Tests
```bash
cd backend

# All Phase 3 tests
pytest tests/test_automl.py -v

# Just AutoML runner tests
pytest tests/test_automl.py::TestAutoMLRunner -v

# Just API tests
pytest tests/test_automl.py::TestExperimentRunAPI -v

# All tests (Phase 0 + 1 + 2 + 3)
pytest tests/ -v
```

## API Endpoints Added

### Experiment Execution
```
POST /experiments/{id}/run     # Queue experiment for background execution
     - Returns: ExperimentRunResponse with task_id
     - Requires: dataset_spec_id must be configured

POST /experiments/{id}/cancel  # Cancel pending/running experiment
     - Returns: ExperimentRunResponse with cancellation status

GET  /experiments/{id}         # Enhanced with summary info
     - Returns: ExperimentDetailResponse with trial_count, best_model, best_metrics
```

## Example Usage

### Run an Experiment (curl)
```bash
# 1. Create project with appropriate task type
# Options: binary, multiclass, regression, quantile, timeseries_forecast,
#          multimodal_classification, multimodal_regression
curl -X POST http://localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Price Prediction", "task_type": "regression"}'

# 2. Upload data file
curl -X POST http://localhost:8000/projects/{project_id}/data-sources/upload \
  -F "file=@housing.csv"

# 3. Create dataset spec
curl -X POST http://localhost:8000/projects/{project_id}/dataset-specs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Housing Spec",
    "target_column": "price",
    "feature_columns": ["sqft", "bedrooms", "bathrooms"],
    "data_sources_json": {"sources": ["{data_source_id}"]}
  }'

# 4. Create experiment
curl -X POST http://localhost:8000/projects/{project_id}/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Housing Price Experiment",
    "dataset_spec_id": "{dataset_spec_id}",
    "primary_metric": "rmse",
    "metric_direction": "minimize"
  }'

# 5. Run the experiment
curl -X POST http://localhost:8000/experiments/{experiment_id}/run

# 6. Check status (poll until completed)
curl http://localhost:8000/experiments/{experiment_id}
```

### Task-Specific Examples

#### Quantile Regression
```bash
# Predict price at 10th, 50th, and 90th percentiles
curl -X POST http://localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Price Intervals", "task_type": "quantile"}'

# In experiment_plan_json, specify quantile_levels:
# "experiment_plan_json": {"automl_config": {"quantile_levels": [0.1, 0.5, 0.9]}}
```

#### Time Series Forecasting
```bash
# Forecast next 7 days
curl -X POST http://localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Sales Forecast", "task_type": "timeseries_forecast"}'

# Data must have: timestamp column, value column
# Config: prediction_length, time_column, id_column (for multiple series)
# "experiment_plan_json": {"automl_config": {"prediction_length": 7, "time_column": "date"}}
```

#### Multimodal Classification
```bash
# Classify using text + tabular features
curl -X POST http://localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Review Sentiment", "task_type": "multimodal_classification"}'

# Data can have: text columns, numeric columns, categorical columns
# AutoGluon automatically detects column types
```

### Run via Swagger UI
1. Open http://localhost:8000/docs
2. Create project, upload data, create dataset spec, create experiment
3. Find `POST /experiments/{experiment_id}/run`
4. Click "Try it out", enter experiment_id, click "Execute"
5. Use `GET /experiments/{experiment_id}` to check status

### Running the Celery Worker
```bash
cd backend
.venv311\Scripts\activate    # Use Python 3.11 venv!

# Start Celery worker (Windows) - note: app.core.celery_app not app.tasks.celery_app
celery -A app.core.celery_app worker --loglevel=info --pool=solo

# Start Celery worker (Linux/Mac)
celery -A app.core.celery_app worker --loglevel=info
```

**Note:** The worker must be running for experiments to execute. Without it, tasks will be queued but not processed.

### Celery Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI       │────▶│     Redis       │────▶│  Celery Worker  │
│   (API Server)  │     │  (Task Queue)   │     │  (Task Executor)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                                               │
        │ POST /experiments/{id}/run                    │
        │ - Validates experiment                        │
        │ - Queues task via .delay()                    │
        │                                               │
        │                                               ▼
        │                                       ┌───────────────┐
        │                                       │ AutoML Runner │
        │                                       │  (AutoGluon)  │
        │                                       └───────────────┘
        │                                               │
        │                                               ▼
        │                                       ┌───────────────┐
        └──────────────────────────────────────│   Database    │
          Poll GET /experiments/{id}            │  (PostgreSQL) │
          to check status                       └───────────────┘
```

### Task Flow: Running an Experiment

1. **API Request**: `POST /experiments/{id}/run`
2. **Validation**: Check experiment exists, has dataset_spec, is in runnable state
3. **Queue Task**: `run_automl_experiment_task.delay(experiment_id)`
4. **Celery Worker** picks up task from Redis queue
5. **Load Data**: `load_dataset_from_spec()` loads DataFrame from data sources
6. **Select Runner**: `get_runner_for_task()` picks TabularRunner/TimeSeriesRunner/MultiModalRunner
7. **Train Models**: AutoGluon trains multiple models within time_limit
8. **Save Results**: Create ModelVersion record with metrics and artifact path
9. **Update Status**: Experiment status → COMPLETED (or FAILED on error)

---

## Running Real AutoGluon Experiments

To run actual ML training (not just tests with mocks), you need all services running:

### Prerequisites Checklist

| Service | Command | Purpose |
|---------|---------|---------|
| PostgreSQL | `docker-compose up -d postgres` | Database |
| Redis | `docker-compose up -d redis` | Celery task queue |
| Backend API | `uvicorn app.main:app --reload` | REST API |
| Celery Worker | `celery -A app.tasks.celery_app worker --loglevel=info --pool=solo` | Executes AutoML tasks |

### Step-by-Step: Start All Services

**Terminal 1: Docker services**
```powershell
cd AgentML
docker-compose up -d postgres redis
```

**Terminal 2: Backend API**
```powershell
cd backend
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
alembic upgrade head    # First time only, or after schema changes
uvicorn app.main:app --reload
```

**Terminal 3: Celery Worker**
```powershell
cd backend
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
celery -A app.tasks.celery_app worker --loglevel=info --pool=solo
```

### Verify AutoGluon Installation

Before running experiments, verify AutoGluon is installed correctly:

```powershell
cd backend
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Quick import test
python -c "import autogluon.tabular; print('tabular OK')"
python -c "import autogluon.timeseries; print('timeseries OK')"
python -c "import autogluon.multimodal; print('multimodal OK')"

# Run the test suite (uses mocks, fast)
python -m pytest tests/test_automl.py -v
```

### Test Real AutoGluon Training (Quick Test)

This actually trains a model (takes 10-30 seconds):

```powershell
cd backend
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -c "
from autogluon.tabular import TabularPredictor
import pandas as pd

# Tiny test dataset
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'target': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
})

# Train for 10 seconds max
predictor = TabularPredictor(label='target').fit(df, time_limit=10)
print('Training complete!')
print(f'Best model: {predictor.model_best}')
"
```

### What Happens When You Run an Experiment

1. **API Call**: `POST /experiments/{id}/run` → Queues Celery task
2. **Celery Worker**: Picks up task from Redis queue
3. **Dataset Building**: `DatasetBuilder.build_dataset_from_spec()` loads data
4. **Runner Selection**: `get_runner_for_task()` picks correct AutoGluon module
5. **Training**: AutoGluon trains multiple models (default: 5 minutes)
6. **Results**: Trial and ModelVersion records created in database

### Watching Experiment Progress

**In Celery worker terminal**, you'll see:
```
[INFO] Starting experiment: My Experiment (uuid...)
[INFO] Created trial: (trial_uuid...)
[INFO] Trial completed: best_model=LightGBM
[INFO] Created model version: (model_uuid...)
[INFO] Experiment completed: My Experiment
```

**Via API**, poll the experiment:
```bash
curl http://localhost:8000/experiments/{experiment_id}
```

Status will change: `pending` → `running` → `completed` (or `failed`)

### Training Time Expectations

| Dataset Size | Time Limit | Actual Time |
|--------------|------------|-------------|
| < 1,000 rows | 60 seconds | ~1-2 minutes |
| 1,000-10,000 rows | 300 seconds (default) | ~5-10 minutes |
| > 10,000 rows | 300+ seconds | 10+ minutes |

**Tip:** For testing, use small datasets and short time limits:
```json
{
  "experiment_plan_json": {
    "automl_config": {
      "time_limit": 60,
      "presets": "medium_quality"
    }
  }
}
```

---

## Synthetic Data Generation (LLM-Powered)

The platform can generate synthetic datasets using LLMs (OpenAI/Gemini) for testing and prototyping.

### How It Works

1. **Configure API Key**: Store your OpenAI or Gemini API key via the Settings page
2. **Request Generation**: `POST /synthetic-datasets/generate` with dataset parameters
3. **Celery Task**: `generate_synthetic_dataset_task` runs in background
4. **LLM Generation**: Prompts LLM to generate realistic CSV data
5. **Import as Data Source**: Generated data is automatically created as a DataSource

### API Endpoints

```
POST /api-keys                          # Store LLM API key
GET  /api-keys                          # List stored API keys
POST /synthetic-datasets/generate       # Start generation
GET  /synthetic-datasets/{id}           # Check generation status
GET  /synthetic-datasets/{id}/download  # Download generated CSV
```

### Generation Request Example

```bash
curl -X POST http://localhost:8000/synthetic-datasets/generate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Customer Churn Data",
    "description": "Synthetic customer data for churn prediction",
    "dataset_type": "binary_classification",
    "provider": "openai",
    "target_training_minutes": 5,
    "num_features": 10
  }'
```

### Dataset Types

| Type | Description | Target Column |
|------|-------------|---------------|
| `binary_classification` | Two-class classification | 0/1 or True/False |
| `multiclass_classification` | Multi-class classification | Category labels |
| `regression` | Continuous value prediction | Numeric values |
| `timeseries` | Time series data | Timestamp + values |

### Requirements

- **Redis running** for Celery task queue
- **Celery worker running** to process generation tasks
- **Valid API key** stored for the selected provider (OpenAI or Gemini)

---

# Phase 3.5: Multi-User Authentication & Sharing

## What Was Done

Phase 3.5 adds multi-user support with authentication, data isolation, and collaboration features.

### User Authentication

- [x] **User Model** - Email/password authentication with optional Google OAuth
- [x] **JWT Tokens** - Secure token-based authentication
- [x] **Password Hashing** - bcrypt for secure password storage
- [x] **Google OAuth** - Sign in with Google support

### API Endpoints

```
POST /api/v1/auth/signup           # Create new account
POST /api/v1/auth/login            # Email/password login
POST /api/v1/auth/google           # Google OAuth login
GET  /api/v1/auth/me               # Get current user
PUT  /api/v1/auth/me               # Update profile
POST /api/v1/auth/change-password  # Change password
POST /api/v1/auth/logout           # Logout (client clears token)
```

### Data Isolation

All existing endpoints now filter data by owner:
- **Projects** - Users only see their own projects (or shared)
- **Experiments** - Scoped to user-owned projects
- **Synthetic Datasets** - Users only see their own datasets (or shared)
- **Conversations** - Chat history scoped per user

**Backward Compatibility:**
- Resources without an `owner_id` (legacy data) remain accessible to all users
- Unauthenticated requests work for read-only operations (optional auth)

### Project Sharing

```
GET  /api/v1/sharing/projects/{id}/shares        # List shares
POST /api/v1/sharing/projects/{id}/shares        # Share with user
PUT  /api/v1/sharing/projects/{id}/shares/{sid}  # Update role
DELETE /api/v1/sharing/projects/{id}/shares/{sid} # Remove share
```

### Synthetic Dataset Sharing

```
GET  /api/v1/sharing/datasets/{id}/shares        # List shares
POST /api/v1/sharing/datasets/{id}/shares        # Share with user
DELETE /api/v1/sharing/datasets/{id}/shares/{sid} # Remove share
```

### Accept Invitations

```
POST /api/v1/sharing/accept-invite  # Accept sharing invitation
GET  /api/v1/sharing/my-shares      # List all resources shared with me
```

### Share Roles

| Role | Permissions |
|------|-------------|
| `viewer` | Read-only access |
| `editor` | Read + write access |
| `admin` | Full access + can manage shares |

### Invite Status

| Status | Description |
|--------|-------------|
| `pending` | Invitation sent, awaiting acceptance |
| `accepted` | User accepted the invitation |
| `declined` | User declined the invitation |
| `expired` | Invitation link expired |

### Email Notifications

When sharing a resource, an email invitation is sent to the recipient:
- Project invitation emails
- Dataset invitation emails
- Configurable SMTP settings in environment variables

### Database Tables Added

| Table | Purpose |
|-------|---------|
| `users` | User accounts (email, password hash, Google ID) |
| `project_shares` | Project sharing records |
| `synthetic_dataset_shares` | Dataset sharing records |

### Configuration (Environment Variables)

```env
# Authentication
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Email (SMTP for invitations)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM_EMAIL=your-email@gmail.com
SMTP_FROM_NAME=AgentML Platform
FRONTEND_URL=http://localhost:5173
```

### Frontend Components

- **Login Page** (`/login`) - Email/password + Google sign-in
- **Signup Page** (`/signup`) - User registration
- **Accept Invite Page** (`/accept-invite?token=...`) - Handle sharing invitations
- **User Menu** - Avatar dropdown with logout
- **Share Dialog** - Modal for managing shares on projects/datasets

### Example Usage

#### Signup and Login
```bash
# Create account
curl -X POST http://localhost:8000/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepass123", "full_name": "John Doe"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepass123"}'

# Response includes access_token - use in Authorization header
# {"access_token": "eyJ...", "token_type": "bearer"}
```

#### Authenticated Requests
```bash
# All subsequent requests include the token
curl http://localhost:8000/projects \
  -H "Authorization: Bearer eyJ..."
```

#### Share a Project
```bash
curl -X POST http://localhost:8000/api/v1/sharing/projects/{project_id}/shares \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{"email": "collaborator@example.com", "role": "editor"}'
```

### Running Auth Tests
```bash
cd backend
pytest tests/test_auth.py -v  # Coming soon
pytest tests/test_sharing.py -v  # Coming soon
```

---

# Phase 4: Model Management & Deployment (Planned)

## What Will Be Done

Phase 4 completes the ML lifecycle with deployment:

### Model Serving
- [ ] `POST /models/{id}/predict` - Real-time inference
- [ ] Load model from artifact location
- [ ] Return predictions with confidence scores

### Deployment Workflow
- [ ] Shadow mode (run predictions, don't serve to users)
- [ ] A/B testing between candidate and production
- [ ] Automatic rollback on metric degradation

### Retraining
- [ ] Scheduled retraining via `RetrainingPolicy`
- [ ] Metric threshold triggers
- [ ] Data drift detection

## How It Connects

```
Phase 3 (Experiments) ──► Phase 4 (Deployment) ──► Production
         │                       │                    │
         │ Provides:             │ Provides:          │ Enables:
         │ - Trained models      │ - Serving endpoint │ - Real predictions
         │ - Metrics             │ - A/B testing      │ - Continuous improvement
         │                       │ - Auto-retraining  │ - Model monitoring
```

## Planned Test Files
- `tests/test_model_serving.py` - Inference endpoints
- `tests/test_deployment.py` - Promotion and rollback
- `tests/test_retraining.py` - Scheduled retraining

---

## API Endpoints Reference

### Health & Info
```
GET  /              # API info
GET  /health        # Health check
```

### Projects
```
POST /projects                      # Create project
GET  /projects                      # List projects
GET  /projects/{id}                 # Get project
PUT  /projects/{id}                 # Update project
DELETE /projects/{id}               # Delete project
```

### Data Sources
```
POST /projects/{id}/data-sources/upload  # Upload file (CSV/Excel/JSON/Parquet/TXT/DOCX)
POST /projects/{id}/data-sources         # Create data source
GET  /projects/{id}/data-sources         # List data sources
GET  /data-sources/{id}                  # Get data source
PUT  /data-sources/{id}                  # Update data source
DELETE /data-sources/{id}                # Delete data source
```

### Dataset Specs
```
POST /projects/{id}/dataset-specs   # Create dataset spec
GET  /projects/{id}/dataset-specs   # List dataset specs
GET  /dataset-specs/{id}            # Get dataset spec
PUT  /dataset-specs/{id}            # Update dataset spec
DELETE /dataset-specs/{id}          # Delete dataset spec
```

### Experiments
```
POST /projects/{id}/experiments     # Create experiment
GET  /projects/{id}/experiments     # List experiments
GET  /experiments/{id}              # Get experiment (with trial_count, best_model)
PUT  /experiments/{id}              # Update experiment
DELETE /experiments/{id}            # Delete experiment
POST /experiments/{id}/run          # Queue experiment for execution (Phase 3)
POST /experiments/{id}/cancel       # Cancel pending/running experiment (Phase 3)
```

### Trials
```
POST /experiments/{id}/trials       # Create trial
GET  /experiments/{id}/trials       # List trials
GET  /trials/{id}                   # Get trial
```

### Models
```
POST /projects/{id}/models          # Create model version
GET  /projects/{id}/models          # List models (optional: ?status_filter=)
GET  /models/{id}                   # Get model
POST /models/{id}/promote           # Promote model status
DELETE /models/{id}                 # Delete model
```

---

## Manual Testing Examples

### Using curl

```bash
# Create a project
curl -X POST http://localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Car Price Prediction", "task_type": "regression"}'

# List projects
curl http://localhost:8000/projects

# Create a data source (replace {project_id})
curl -X POST http://localhost:8000/projects/{project_id}/data-sources \
  -H "Content-Type: application/json" \
  -d '{"name": "Sales CSV", "type": "file_upload", "config_json": {"path": "/data/sales.csv"}}'

# Create an experiment
curl -X POST http://localhost:8000/projects/{project_id}/experiments \
  -H "Content-Type: application/json" \
  -d '{"name": "Baseline Experiment", "primary_metric": "rmse", "metric_direction": "minimize"}'

# Promote a model to production
curl -X POST http://localhost:8000/models/{model_id}/promote \
  -H "Content-Type: application/json" \
  -d '{"status": "production"}'
```

### Using Swagger UI

1. Open http://localhost:8000/docs
2. Click on any endpoint to expand it
3. Click "Try it out"
4. Fill in parameters and request body
5. Click "Execute"

---

## Database

### Migrations

```bash
cd backend
.\venv\Scripts\activate

# Apply all migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Rollback all migrations
alembic downgrade base

# Check current revision
alembic current

# View migration history
alembic history
```

### Direct Database Access

```bash
# Connect via Docker
docker exec -it agentic-ml-postgres psql -U postgres -d agentic_ml

# Useful SQL commands
\dt                           # List tables
\d projects                   # Describe projects table
SELECT * FROM projects;       # View all projects
SELECT * FROM alembic_version; # Check migration version
```

### Tables (Phase 1)
- `projects` - ML projects
- `data_sources` - Data connections (file, DB, S3, API)
- `dataset_specs` - Dataset definitions
- `experiments` - ML experiments
- `trials` - Experiment runs/variants
- `model_versions` - Trained models
- `retraining_policies` - Auto-retrain rules
- `alembic_version` - Migration tracking

---

## Troubleshooting

### PostgreSQL Enum Case Sensitivity

**Problem:** When creating projects or other entities with enum fields, you may see an error like:
```
psycopg2.errors.InvalidTextRepresentation: invalid input value for enum tasktype: "CLASSIFICATION"
```

**Cause:** PostgreSQL enum types are case-sensitive. The database expects lowercase values (`'classification'`) but SQLAlchemy was sending Python enum names which are uppercase (`'CLASSIFICATION'`).

**Solution:** All SQLAlchemy models now use `values_callable` to ensure enum values (lowercase) are sent instead of enum names (uppercase):

```python
# Before (broken):
task_type = Column(SQLEnum(TaskType), nullable=True)

# After (fixed):
task_type = Column(
    SQLEnum(TaskType, values_callable=lambda x: [e.value for e in x]),
    nullable=True
)
```

**Files Fixed:**
- `app/models/project.py` - TaskType, ProjectStatus
- `app/models/experiment.py` - ExperimentStatus, MetricDirection, TrialStatus
- `app/models/data_source.py` - DataSourceType
- `app/models/model_version.py` - ModelStatus
- `app/models/retraining_policy.py` - PolicyType

This fix ensures PostgreSQL receives the correct lowercase enum values that match the database enum types created by Alembic migrations.

---

### Port Conflicts

**PostgreSQL port 5432 in use:**
```bash
# Check what's using the port
netstat -aon | findstr :5432

# Stop local PostgreSQL (Windows Services or):
net stop postgresql-x64-17
```

**Backend port 8000 in use:**
```bash
# Use a different port
uvicorn app.main:app --reload --port 8001
```

### Database Connection Issues

```bash
# Verify Docker container is running
docker ps

# Check container logs
docker logs agentic-ml-postgres

# Restart containers
docker-compose down
docker-compose up -d postgres redis
```

### Test Failures

```bash
# Run with more verbose output
pytest -vvs

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Show local variables in tracebacks
pytest -l
```

### Import Errors

```bash
# Make sure venv is activated
.\venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

## Code Quality

### Backend

```bash
cd backend
.\venv\Scripts\activate

# Format code
black .

# Lint
ruff check .

# Type check
mypy app
```

### Frontend

```bash
cd frontend

# Lint
npm run lint

# Type check (via build)
npm run build
```

---

## Project Structure

```
AgentML/
├── backend/
│   ├── app/
│   │   ├── api/           # API route handlers
│   │   │   ├── health.py
│   │   │   ├── projects.py
│   │   │   ├── data_sources.py
│   │   │   ├── dataset_specs.py
│   │   │   ├── experiments.py
│   │   │   └── models.py
│   │   ├── core/          # Config, database setup
│   │   │   ├── config.py
│   │   │   └── database.py
│   │   ├── models/        # SQLAlchemy ORM models
│   │   │   ├── base.py           # GUID, JSONType helpers
│   │   │   ├── project.py
│   │   │   ├── data_source.py
│   │   │   ├── dataset_spec.py
│   │   │   ├── experiment.py
│   │   │   ├── model_version.py
│   │   │   └── retraining_policy.py
│   │   ├── schemas/       # Pydantic request/response schemas
│   │   │   ├── project.py
│   │   │   ├── data_source.py
│   │   │   ├── dataset_spec.py
│   │   │   ├── experiment.py
│   │   │   └── model_version.py
│   │   ├── services/      # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── schema_analyzer.py   # File analysis (Phase 2)
│   │   │   ├── dataset_builder.py   # Build DataFrames from specs (Phase 2)
│   │   │   └── automl_runner.py     # AutoGluon wrapper (Phase 3)
│   │   └── tasks/         # Celery async tasks
│   │       ├── __init__.py
│   │       ├── automl.py              # run_automl_experiment_task (Phase 3)
│   │       └── synthetic_data.py      # generate_synthetic_dataset_task
│   ├── alembic/           # Database migrations
│   │   ├── versions/
│   │   │   └── 001_initial_schema.py
│   │   └── env.py
│   ├── tests/             # pytest test files
│   │   ├── conftest.py          # Test fixtures
│   │   ├── test_health.py
│   │   ├── test_projects.py
│   │   ├── test_data_sources.py
│   │   ├── test_dataset_specs.py
│   │   ├── test_experiments.py
│   │   ├── test_models.py
│   │   ├── test_file_upload.py      # Phase 2
│   │   ├── test_schema_analyzer.py  # Phase 2
│   │   ├── test_dataset_builder.py  # Phase 2
│   │   └── test_automl.py           # Phase 3
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/         # React page components
│   │   │   ├── Home.tsx
│   │   │   ├── Projects.tsx
│   │   │   ├── Experiments.tsx
│   │   │   └── Models.tsx
│   │   ├── layouts/       # Layout components
│   │   │   └── MainLayout.tsx
│   │   └── App.tsx        # Router setup
│   └── package.json
├── docker-compose.yml
├── TESTING.md             # This file
└── README.md
```

---

## Environment Variables

Create `backend/.env` from `backend/.env.example`:

```env
# Application
DEBUG=true

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/agentic_ml

# Redis
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# LLM API (Phase 4+)
LLM_API_KEY=your-api-key-here
LLM_API_BASE_URL=https://api.openai.com/v1

# CORS
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# File uploads
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE_MB=100

# Model artifacts (Phase 3)
ARTIFACTS_DIR=./artifacts

# AutoML settings (Phase 3)
AUTOML_TIME_LIMIT=300
AUTOML_PRESETS=medium_quality
```
