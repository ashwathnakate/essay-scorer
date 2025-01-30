# Automated Essay Scorer

## 📜 Project Description

The **Automated Essay Scorer** is a Streamlit-based web application designed to predict a score for user-submitted essays. Powered by a trained AI model, this app utilizes natural language processing (NLP) techniques to evaluate essays and generate feedback.

Features include:

- **Essay Scoring:** Predicts a score between 1 and 6 based on the essay quality.
- **Word Cloud:** Displays a visual representation of the most frequent words in the essay.
- **Feedback Messages:** Provides constructive feedback based on the predicted score.
- **Interactive UI:** An easy-to-use interface built with Streamlit.



## 🗂️ Project Structure

```
├── model
│   ├── model.h5
│   ├── new_model.joblib
│   ├── saved_model.keras
│   └── training_data
│       └── ielts_writing_dataset.csv
├── __pycache__
├── app.py
├── requirements.txt
└── README.md
```



## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

1. **Python 3.8+**
2. **Virtual Environment (Optional)**: Recommended for package management.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/username/automated-essay-scorer.git
   cd automated-essay-scorer
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # For macOS/Linux
   venv\Scripts\activate      # For Windows
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Add the Model Files

Ensure the `model` directory contains the required model files (`new_model.joblib`) and the training dataset (`ielts_writing_dataset.csv`).

---

## 🏃‍♂️ Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default browser. If not, navigate to the URL displayed in your terminal (e.g., `http://localhost:8501`).

---

## 🛠 Features

1. **Text Input:**
   - Enter an essay in the provided text box.

2. **Score Prediction:**
   - Click the **Predict Score** button to get a score between 1 and 6.

3. **Word Cloud Visualization:**
   - Displays the most frequently used words in the essay.

4. **Feedback:**
   - Get actionable feedback based on the predicted score.

---

## 🧾 Dependencies

The required Python packages are listed in the `requirements.txt` file. Key dependencies include:

- `streamlit`
- `tensorflow`
- `joblib`
- `nltk`
- `matplotlib`
- `wordcloud`
- `scikit-learn`

---

## 📄 Dataset

The training data used to develop the model is included in `training_data/ielts_writing_dataset.csv`.

---

## ✍️ Customization

To train a new model or improve performance, replace the current model files (`new_model.joblib`, etc.) with updated versions and adjust the `vectorizer` and `predict_essay_score` functions as needed.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## 🤝 Acknowledgements

- [Streamlit](https://streamlit.io) for the interactive interface.
- [WordCloud](https://github.com/amueller/word_cloud) for visualizations.
- [NLTK](https://www.nltk.org) for text preprocessing.

---
