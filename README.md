# 🛡️ Fraud Detection System

A comprehensive machine learning-powered fraud detection system with an interactive Streamlit dashboard for real-time monitoring and analysis.

## 📋 Project Overview

This project implements a sophisticated fraud detection system using machine learning algorithms to identify suspicious transactions. It provides both a machine learning pipeline for model training and a user-friendly web interface for fraud detection, monitoring, and reporting.

### Key Features

- **Real-time Fraud Detection**: Instantly classify transactions as legitimate or fraudulent
- **Interactive Dashboard**: Beautiful, responsive Streamlit web interface
- **Data Visualization**: Interactive plots and charts for data analysis
- **Alert System**: Automated alerts for suspicious activities
- **Model Management**: Pre-trained ML models for immediate use
- **Data Analysis**: Comprehensive exploratory data analysis (EDA) and insights
- **CSV Upload**: Process and analyze your own transaction data
- **Report Generation**: Export results and analysis reports

## 🏗️ Project Structure

```
gestyt/
├── fraud_detection.py              # Main Streamlit application
├── fraud_detection_with_alerts.py  # Enhanced version with alert system
├── fraud_detection_pipeline.pkl    # Pre-trained ML model
├── analysis_model.ipynb            # Jupyter notebook for data analysis
├── AIML Dataset.csv                # Training dataset
├── data.txt                        # Additional data files
├── README.md                       # This file
└── env/                           # Python virtual environment
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- pip (Python package manager)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/gestyt.git
   cd gestyt
   ```

2. **Create and activate virtual environment**

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Start the main fraud detection dashboard:**

```bash
streamlit run fraud_detection.py
```

**Start the enhanced version with alerts:**

```bash
streamlit run fraud_detection_with_alerts.py
```

The application will open in your default browser at `http://localhost:8501`

## 📊 How to Use

### Dashboard Features

1. **Transaction Analysis**
   - Upload CSV files with transaction data
   - View real-time fraud predictions
   - Analyze transaction patterns

2. **Visualizations**
   - Interactive charts and graphs
   - Transaction distribution analysis
   - Fraud trends visualization
   - Statistical summaries

3. **Alerts & Notifications**
   - Real-time suspicious activity alerts
   - Severity levels (Low, Medium, High)
   - Alert history and trends

4. **Reports**
   - Export analysis results
   - Generate PDF reports
   - Download fraud statistics

## 🤖 Machine Learning Model

The fraud detection system uses a pre-trained machine learning model (`fraud_detection_pipeline.pkl`) that:

- Detects patterns associated with fraudulent transactions
- Provides probability scores for fraud likelihood
- Handles multiple features including transaction amount, merchant type, location, and more
- Achieves high accuracy on unseen data

### Model Performance

The model is trained on the AIML Dataset and optimized for:

- High sensitivity (catching fraud cases)
- Reasonable specificity (minimizing false alarms)
- Fast inference for real-time predictions

## 📁 Data Format

Expected CSV format for transaction data:

```csv
transaction_id,amount,merchant,location,transaction_type,timestamp,...
```

The system automatically validates and preprocesses the data before making predictions.

## 🛠️ Technologies Used

- **Streamlit**: Web framework for interactive dashboards
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive visualizations
- **Joblib**: Model serialization
- **ReportLab**: PDF report generation

## 📊 Exploratory Data Analysis

Check out `analysis_model.ipynb` for detailed EDA, including:

- Dataset statistics and distribution
- Fraud patterns analysis
- Feature correlation analysis
- Model performance metrics
- Data preprocessing steps

## 🔐 Security & Privacy

- Models process data locally
- No data is sent to external servers
- Supports encrypted transaction data
- Compliant with data protection standards

## 📈 Future Enhancements

- [ ] Real-time database integration
- [ ] Advanced anomaly detection
- [ ] Custom model training interface
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Integration with payment APIs
- [ ] Advanced statistical analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)

## 📧 Contact

For questions or support, please create an issue on GitHub or contact me at your.email@example.com

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

Made with ❤️ for fraud detection enthusiasts
