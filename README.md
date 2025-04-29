# 🎬 Mood-Based Movie Recommender

This is a simple **Movie Recommendation System** that suggests movies based on your **mood** and your **favorite movie**. It uses **Natural Language Processing** and **Machine Learning (TF-IDF + Cosine Similarity)** to recommend similar movies from a dataset.

Built using **Python**, **Streamlit**, and **scikit-learn**.

---

## 🚀 Live Demo
[Click here to try the app on Streamlit Cloud](https://mood-based-movie-recommender.streamlit.app/)  

---

## 📂 Project Structure

mood-movie-recommender/ <br>
      ├── app.py # Main Streamlit app <br>
      ├── movies_dataset.csv # Dataset used for recommendations <br>
      ├── requirements.txt # Python dependencies <br>
      ├── README.md # Project description<br>


---

## 💡 Features

- Recommend movies based on:
  - Your favorite movie
  - Your current mood (happy, sad, romantic, etc.)
  - Optional year filter
- Mood-to-genre mapping
- Emoji-enhanced output
- Fall-back recommendations if movie is not found

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/laksitha-s/Mood-Based-Movie-Recommender
   cd Mood-Based-Movie-Recommender
2. Optional but recommended) Create a virtual environment:
   ```bash
     python -m venv venv
     source venv/bin/activate  # or .\venv\Scripts\activate on Windows
3. Install dependencies:
   ```bash
     pip install -r requirements.txt
4. Run the app:
   ```bash
     streamlit run app.py

## 🌐 Deployment
1. This app is deployed using Streamlit Cloud:

2. Push your code to GitHub

3. Go to https://streamlit.io/cloud

4. Connect your GitHub repo

5. Set app.py as the main file and deploy!

## 📧 Contact<br>
Built with ❤️ by Laksitha<br>
Feel free to contribute or reach out!
