# Food Waste API & Frontend

This repository provides a FastAPI backend for food freshness estimation and surplus calculation, along with a small static frontend that consumes the API.

## Prerequisites

- Python 3.10 or newer (Windows/macOS/Linux)
- Node.js & npm (for TypeScript build, optional if you use precompiled JS)

## Setup

1. **Clone or download** the repository.

2. **Install Python packages:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies & compile TypeScript** (only needed when editing `src/app.ts`):
   ```bash
   npm install        # creates node_modules and installs typescript
   npx tsc            # compiles src/app.ts → static/app.js
   ```
   You can also run `npm run build` if a script exists in `package.json`.

4. **Start the API server:**
   ```bash
   uvicorn app_Back:app --reload
   ```
   By default it listens on `http://127.0.0.1:8000`.

5. **Open the UI:**
   Navigate to `http://localhost:8000/ui` in your browser. Use the forms to test freshness and surplus calculators.
   The OpenAPI docs are available at `http://localhost:8000/docs`.

## Development notes

- The frontend HTML is `Sanjith.html`. Static assets (compiled JS) live in `static/`.
- TypeScript source is under `src/app.ts`; editing requires rebuilding via `npx tsc`.
- The backend is `app_Back.py` with two POST endpoints (`/freshness`, `/estimate`) and a GET for `/ui`.
- CORS middleware is enabled to allow requests from any origin.

## Cleanup

If you clone fresh or need to regenerate dependencies:

```bash
rm -rf node_modules __pycache__
npm install  # rebuild JS
```

## Deployment hints

- You can deploy this FastAPI app to hosting providers like Render, Heroku, or Azure App Service.
- Ensure both Python and Node/npm build steps are run during deployment, or simply commit `static/app.js` so the frontend works out of the box.

---

Feel free to extend the API with more dishes, better models, or a richer frontend!