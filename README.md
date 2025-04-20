# Forest Fire Detection System

This project implements a real-time forest fire detection system using machine learning and Streamlit for visualization. The system uses CO2 levels, temperature, and humidity as input features to predict potential fire risks.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- streamlit
- python-dateutil

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run main.py
```

2. Access the web interface at `http://localhost:8501`

3. Use the interface to:

   - View model accuracy
   - Start/stop real-time simulation
   - Monitor CO2, temperature, and humidity readings
   - View detection results

## Screenshot 📸

![Antarmuka Streamlit 1](https://drive.google.com/uc?export=view&id=107a8Gm4506VN3mAvNArL3xVNpv9xIGry)
![Antarmuka Streamlit 2](https://drive.google.com/uc?export=view&id=1FEaJZxM3d-qduRhk3AuvLNGbwiOuwkhd)
