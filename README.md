# MedOptix AI – Personalized Treatment Optimization in European Healthcare

**Amdari Internship Accelerator – June 2025**
**Facilitator:** Muhammad Yekini

---

## 🔍 Overview

MedOptix is a UK-based digital health startup helping NHS and private orthopedic clinics improve patient outcomes. This full-stack data science project predicts therapy dropout, identifies adherence patterns, and powers personalized nudges for recovery.

---

## 🚀 Objectives

- Build an end-to-end pipeline: from CSV → PostgreSQL → Analysis → API
- Segment patients into personas based on recovery behavior
- Predict likelihood of dropout by Week 3
- Visualize trends in therapy adherence across clinics

---

## 📦 Project Structure

```
etl/          → Scripts to clean/load CSVs into PostgreSQL
eda/          → Jupyter notebooks for exploratory analysis
models/       → ML models: dropout prediction, adherence forecast
api/          → FastAPI endpoints for predictions
dashboards/   → Visual interfaces using Streamlit or Power BI
```

---

## 🧶 Tech Stack

- **Languages**: Python, SQL
- **Libraries**: Pandas, Seaborn, Plotly, scikit-learn, XGBoost, SHAP
- **Database**: PostgreSQL (Aiven Cloud)
- **Deployment**: Docker, Heroku/AWS EC2
- **API**: FastAPI, Swagger
- **CI/CD**: GitHub Actions
- **Visualization**: Streamlit, Power BI

---

## 📊 Current Progress

| Phase            | Status         | Notes                           |
| ---------------- | -------------- | ------------------------------- |
| Data Engineering | ✅ Completed   | Tables created, ETL ready       |
| EDA              | ⏳ In Progress | Pain trends & dropouts underway |
| Modeling         | ⏳ Pending     | Starts June 22                  |
| API + CI/CD      | ⏳ Pending     | After model training            |
| Dashboard        | ⏳ Pending     | Streamlit or Power BI           |

---

## 📚 Resources

- [Project Brief PDF](docs/medoptix-brief.pdf)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [Healthcare ML Fairness](https://fairlearn.org)
- [Streamlit Dashboards](https://streamlit.io)

---

## 🤝 Contributions

This project is part of the **Amdari Internship Accelerator**
Mentor: Muhammad Yekini
Intern Cohort: Jan–Jul 2025

---

## 👤 Author (Submission Info)

**Name:** Gideon Okomayin  
**GitHub:** [github.com/GideonOkomayin](https://github.com/gideonkomayin)  
**LinkedIn:** [linkedin.com/in/gideonokomayin](https://linkedin.com/in/gideonokomayin)  
**Demo Video:** Coming Soon  

**Personal Contributions:**  
- Conducted exploratory data analysis (EDA) on therapy adherence and dropout patterns  
- Contributed to model development for early dropout prediction  
- Assisted with API testing and CI/CD integration  
- Documented workflows and insights for team collaboration

