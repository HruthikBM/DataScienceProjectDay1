# Complete Data Handling & ML Experiments

This repository contains machine learning experiments and data handling projects, with a focus on cancer diagnosis classification and various case studies.

## 📁 Project Structure

```
├── kaggle.json                     # Kaggle API credentials
├── requirements.txt                # Python dependencies
├── case_study_4/                   # Case study experiments
│   ├── main_exp.ipynb             # Main experimental notebook
│   ├── data/                      # Case study data
│   └── src/                       # Source code
└── personal-cancer-diagnosis/      # Cancer diagnosis classification project
    ├── data-preprocess.ipynb      # Data preprocessing and model training
    ├── encode.py                  # Feature encoding utilities
    ├── train_model.py            # Model training utilities
    ├── visual.py                 # Visualization utilities
    └── data/                     # Training datasets
        ├── training_text/
        └── training_variants/
```

## 🔬 Personal Cancer Diagnosis Project

A machine learning project for classifying genetic variants in cancer diagnosis using text analysis and multiple encoding techniques.

### Features

- **Multi-feature Encoding**: Implements various encoding strategies for genetic data
  - One-hot encoding for genes
  - Response coding for variations
  - TF-IDF vectorization for text data
- **Advanced Preprocessing**: Custom NLP preprocessing pipeline for medical text
- **Model Comparison**: Tests multiple ML algorithms (SGD, Naive Bayes, etc.)
- **Feature Fusion**: Combines genetic, variation, and text features using sparse matrices
- **Visualization**: Confusion matrices and performance plots

### Dataset

The project uses cancer genetic mutation data with:
- **Training Variants**: Gene, Variation, Class information
- **Training Text**: Medical literature text associated with mutations
- **9 Classes**: Different types of genetic variations

### Key Components

#### 1. Data Preprocessing (`data-preprocess.ipynb`)
- Loads and merges variant and text data
- Implements custom NLP preprocessing
- Handles missing values and data cleaning
- Stratified train/validation/test splits

#### 2. Feature Encoding (`encode.py`)
```python
class Encoder:
    def get_onehotCoding()      # One-hot encoding for categorical features
    def get_response_coding()   # Response-based encoding using class probabilities
```

#### 3. Model Training (`train_model.py`)
```python
def best_parameter()    # Grid search for hyperparameter optimization
def predict_model()     # Model prediction with probability/class output
def visualize_model()   # Training curves and validation visualization
```

#### 4. Visualization (`visual.py`)
- Confusion matrix plots
- Performance metrics visualization
- Class distribution analysis

### Performance

The project achieves competitive performance using:
- **SGDClassifier** with optimized hyperparameters
- **Feature combination** of TF-IDF text vectors, response-coded variations, and one-hot genes
- **Cross-validation** for robust model evaluation

### Models Tested
- Stochastic Gradient Descent (SGD)
- Multinomial Naive Bayes
- Support Vector Machines
- Random Forest
- Logistic Regression

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab
- Required packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd compleate-data-handle
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
```

### Usage

#### Cancer Diagnosis Project

1. Navigate to the project directory:
```bash
cd personal-cancer-diagnosis
```

2. Open the main notebook:
```bash
jupyter notebook data-preprocess.ipynb
```

3. Run cells sequentially to:
   - Load and preprocess data
   - Generate different feature encodings
   - Train and evaluate models
   - Visualize results

#### Key Notebook Sections

1. **Data Loading & EDA**: Load training variants and text data
2. **Text Preprocessing**: Custom NLP pipeline for medical text
3. **Feature Engineering**: Multiple encoding strategies
4. **Model Training**: Grid search with cross-validation
5. **Model Evaluation**: Performance metrics and visualization

## 📊 Results

The project demonstrates:
- Effective handling of multi-modal genetic data
- Performance comparison across different ML algorithms
- Feature importance analysis for genetic variant classification
- Robust evaluation using stratified cross-validation

## 🛠️ Technical Details

### Key Technologies
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and utilities
- **NLTK**: Natural language processing
- **matplotlib/seaborn**: Data visualization
- **scipy**: Sparse matrix operations

### Feature Engineering Techniques
- **TF-IDF Vectorization**: For medical text analysis
- **One-Hot Encoding**: For categorical gene features
- **Response Coding**: Probability-based encoding for variations
- **Sparse Matrix Operations**: Efficient feature combination

## 📈 Performance Metrics

Models are evaluated using:
- **Log Loss**: Primary metric for multi-class probability prediction
- **Accuracy**: Classification accuracy
- **Confusion Matrix**: Detailed class-wise performance
- **Cross-Validation**: 5-fold stratified CV for robust evaluation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kaggle for providing the cancer genetics dataset
- The scientific community for genetic variant classification research
- Open source contributors to the machine learning libraries used

## 📧 Contact

For questions or collaboration opportunities, please open an issue or reach out through GitHub.

---

**Note**: This project is for educational and research purposes. Medical predictions should always be validated by qualified healthcare professionals.
