# Dashboard Test Guide

This guide explains how to run the project locally, open both dashboards, and walk through the main functionality in a realistic demo flow.

Important:
- The local dashboards use the repository's saved dataset snapshot and bundled forecast assets.
- In other words, the app behaves like a real deployed system, but it is not pulling live NASA POWER, WHO, or EM-DAT data during this local demo.

---

## 1. Current Dashboard Paths

- Django project root: `C:\Users\USER\Desktop\ai-disaster-disease-prediction\Dashboard\futureguard_dashboard`
- Local environment folder: `C:\Users\USER\Desktop\ai-disaster-disease-prediction\Dashboard\.venv`
- Community dashboard URL: `http://127.0.0.1:8000/`
- Admin dashboard URL: `http://127.0.0.1:8000/admin-dashboard/`

Legacy redirects still work:
- `http://127.0.0.1:8000/community-dashboard/` redirects to `/`
- `http://127.0.0.1:8000/dashboard/` redirects to `/admin-dashboard/`

---

## 2. Start the Application

Open PowerShell and run:

```powershell
cd C:\Users\USER\Desktop\ai-disaster-disease-prediction\Dashboard\futureguard_dashboard
$env:PYTHONPATH = "C:\Users\USER\Desktop\ai-disaster-disease-prediction\Dashboard\.venv\Lib\site-packages"
python manage.py check
python manage.py runserver
```

If `python` is not available in your terminal, replace it with `py`.

Expected result:
- `python manage.py check` should report no critical issues.
- `python manage.py runserver` should start the local server on port `8000`.

---

## 3. Quick Smoke Test

After the server starts, open these two URLs in your browser:

- Community dashboard: `http://127.0.0.1:8000/`
- Admin dashboard: `http://127.0.0.1:8000/admin-dashboard/`

You should see:
- The community-facing FutureGuard page at the root URL.
- The admin dashboard with KPI cards, charts, and sidebar sections at `/admin-dashboard/`.

---

## 4. Community Dashboard Walkthrough

URL:

```text
http://127.0.0.1:8000/
```

### What this dashboard is for

This is the public/community view. It helps a user:
- choose a county
- run a risk prediction
- see disease and disaster risk levels
- get an action plan
- switch between English and Kiswahili
- download or share the result

### Step-by-step demo

1. Open the community dashboard.
2. In `Select Your County`, choose a county such as:
   - `Nairobi`
   - `Mombasa`
   - `Garissa`
3. Click `Run Prediction`.
4. Wait for the loading state to finish.
5. Confirm that the results section appears and shows risk cards for:
   - Malaria
   - Cholera
   - Dengue
   - Floods
   - Drought
6. Confirm that each card shows a risk label and confidence percentage.
7. Scroll to `Recommended Action Plan` and confirm that actions are generated.
8. Click `Kiswahili` and confirm that the action plan and labels update.
9. Click `English` to switch back.
10. Click `Download Report` and confirm that a `.txt` report is downloaded.
11. Click `Share with Community` and confirm that either:
   - the browser share dialog opens, or
   - the summary text is copied to the clipboard.

### Best counties to test

Use different counties to make the behavior visibly change:

- `Garissa`: good for drought-heavy / arid risk patterns
- `Mombasa`: good for coastal and cholera-style variation
- `Nairobi`: good for central urban comparison

### What is actually live here

These community actions are backed by the app logic:
- `POST /api/predict/`
- `POST /api/action-plan/`
- dynamic language switching
- report generation
- share summary generation

### Extra verification

If you want to see the requests in a more real-world way:

1. Press `F12` in the browser.
2. Open the `Network` tab.
3. Run a prediction again.
4. Confirm that `/api/predict/` and `/api/action-plan/` return successful responses.

---

## 5. Admin Dashboard Walkthrough

URL:

```text
http://127.0.0.1:8000/admin-dashboard/
```

### What this dashboard is for

This is the operations/admin view. It gives a broader system picture:
- summary metrics
- climate and health trend charts
- risk map
- prediction form
- evaluation and XAI sections

Note:
- This admin dashboard is a single dashboard UI with multiple sections.
- Use the left sidebar to move between sections.
- Most section changes happen inside the same page rather than loading a new URL.

### Step-by-step demo

1. Open the admin dashboard.
2. Start on the default overview page and confirm you can see:
   - KPI cards
   - alerts
   - model performance widgets
   - forecast image cards
3. Click `Live Predictions` in the sidebar.
4. In `Run New Prediction`, choose:
   - Disease: `Drought`
   - Region: `Arid and Semi-Arid North-East (Garissa)`
   - Horizon: `30 days`
5. Click `Run Prediction`.
6. Confirm the `Prediction Result` card updates with:
   - risk confidence
   - risk label
   - county
   - climate zone
   - prediction drivers
7. Change the form and run another example, such as:
   - Disease: `Cholera`
   - Region: `Coastal Lowlands (Mombasa)`
   - Horizon: `14 days`
8. Click `Risk Map` and confirm the map section loads with region risk markers and the table below it.
9. Click through these additional sections from the sidebar:
   - `Data Ingestion`
   - `Climate Data`
   - `Health Records`
   - `Disaster Events`
   - `LSTM Model`
   - `XGBoost Model`
   - `Hybrid Ensemble`
   - `XAI / Explainability`
   - `Evaluation`
   - `Export Report`

### What is actually live here

These admin features are backed by real local dashboard data:
- chart payloads rendered from backend context
- climate-region prediction scoring
- the admin prediction form
- the risk map data
- region and disease selector behavior

### What is currently demo/presentation content

Some parts are currently informative UI sections rather than fully wired operational tools. For example:
- some export buttons are visual/demo controls
- some ingestion or sync buttons are presentation-only
- evaluation and XAI pages display prepared data for review rather than launching new backend jobs

That still makes them useful for a real product walkthrough, but they should be presented as dashboard views, not full production admin tooling.

---

## 6. Recommended Real-World Demo Flow

If you want to show the full system to a lecturer, client, or teammate, use this order:

1. Start with the community dashboard at `/`
2. Run a prediction for `Garissa`
3. Show the action plan and switch to `Kiswahili`
4. Download the report
5. Run a second community prediction for `Mombasa` to show different behavior
6. Move to the admin dashboard at `/admin-dashboard/`
7. Show overview KPIs and charts
8. Open `Live Predictions` and run a new admin prediction
9. Open `Risk Map` to show climate-region visualization
10. Finish with `XAI / Explainability` and `Evaluation` to show model transparency and performance

---

## 7. Optional Django Admin

If you also want the built-in Django admin site:

```powershell
cd C:\Users\USER\Desktop\ai-disaster-disease-prediction\Dashboard\futureguard_dashboard
$env:PYTHONPATH = "C:\Users\USER\Desktop\ai-disaster-disease-prediction\Dashboard\.venv\Lib\site-packages"
python manage.py createsuperuser
```

Then visit:

```text
http://127.0.0.1:8000/admin/
```

This is separate from the custom admin dashboard at `/admin-dashboard/`.

---

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'django'`

Run this before starting the server:

```powershell
$env:PYTHONPATH = "C:\Users\USER\Desktop\ai-disaster-disease-prediction\Dashboard\.venv\Lib\site-packages"
```

### Port 8000 is already in use

Run:

```powershell
python manage.py runserver 127.0.0.1:8001
```

Then open:

```text
http://127.0.0.1:8001/
http://127.0.0.1:8001/admin-dashboard/
```

### The browser opens the old paths

Use the canonical paths:
- Community: `/`
- Admin: `/admin-dashboard/`

### You want to confirm the code still passes tests

From the repository root:

```powershell
cd C:\Users\USER\Desktop\ai-disaster-disease-prediction
python -m pytest tests -q
```

---

## 9. Success Checklist

- Server starts without import errors
- Community dashboard opens at `/`
- Admin dashboard opens at `/admin-dashboard/`
- Community prediction runs successfully
- Action plan appears and language switch works
- Community report downloads
- Admin prediction form updates results
- Admin charts and risk map are visible
- Sidebar sections are navigable
- Tests pass
