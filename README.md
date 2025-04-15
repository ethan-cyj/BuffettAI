# BuffettAI
BuffettAI is a specialized language model that emulates Warren Buffett’s investing philosophy, tone, and personality.


## Setup

### macOS
1. Open a terminal.
2. From the project's root directory, run:
   ```bash
   source setup_venv.sh
   ```
   to set up virtual environment with all dependencies installed

### Windows
1. Create virtual environment (if not yet done):
  ```bash
  python -m venv venv
  ```
2. Activate virtual environment
3. Install required dependencies from `requirements.txt`
  ```bash
  pip install -r  requirements.txt
  ```

## Running backend
To run backend execute `scripts/main.py`

## Running Frontend
1. Change directory to frontend folder (buffettai)
  ```bash
  cd buffettai
  ```
2. Install frontend dependencies:
  ```bash
  npm install
  ```
3. Run the development server:
  ```bash
  npm run dev
  ```

Hosted at: http://localhost:5173/
   
