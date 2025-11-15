# Patient Risk Management UI

A user-friendly web interface for managing patient health risk data.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install Streamlit directly:
```bash
pip install streamlit
```

### 2. Train the Model (First Time Only)

Before using the UI, you need to train the risk prediction model:

```bash
python riskcalculator.py
```

This will:
- Train the risk prediction model
- Save it as `trained_pipeline.pkl`
- Generate initial risk scores

### 3. Launch the Web UI

```bash
streamlit run patient_ui.py
```

The UI will open automatically in your web browser at `http://localhost:8501`

## Features

### 📊 Dashboard
- View total number of patients
- See average risk score
- View high-risk patient count
- Check model status
- See risk score distribution chart
- View recent patients

### ➕ Add Patient
- Simple form to add new patients
- Auto-generates Patient IDs
- Automatically calculates risk scores
- Input validation with helpful warnings

### ❌ Remove Patient
- Select patient from dropdown
- View patient details before removal
- Confirmation before deletion

### 👤 View Patient
- Select any patient to view full details
- Organized display of vital signs
- Complete patient information

### 📋 All Patients
- View all patients in a searchable table
- Filter by risk level
- Search by Patient ID
- Download filtered data as CSV

## Usage Tips

1. **Adding Patients**: Fill in the form with patient vital signs. The system will automatically calculate a risk score (0-100) if the model is loaded.

2. **Risk Scores**: Higher scores (closer to 100) indicate higher risk. The score is calculated using the trained machine learning model.

3. **Patient IDs**: Leave the Patient ID field as-is to auto-generate a unique ID (format: P0001, P0002, etc.)

4. **Data Persistence**: All changes are automatically saved to `Health_Risk_Dataset.csv`

## Troubleshooting

- **Model not found**: Make sure you've run `riskcalculator.py` first to generate `trained_pipeline.pkl`
- **No risk scores**: The UI will still work without the model, but risk scores won't be calculated
- **Port already in use**: If port 8501 is busy, Streamlit will automatically use the next available port

## Technical Details

- Built with Streamlit for easy web interface
- Uses the trained model from `riskcalculator.py` for risk score prediction
- All data stored in CSV format for easy access
- Real-time updates when adding/removing patients

