### Simple AI/ML Use Cases for Energy Professionals 

These use cases can be built following the structured project steps outlined in the boot camp: defining the problem and success metrics, accessing data, outlining algorithm steps (data loading, cleaning, transforming, model training, prediction, visualization), building code, assessing performance, and creating a final report and GitHub repository. 

### Always use publicly available data or synthetic data

#### Upstream Examples (5 Use Cases)

1.  **Oil Well Shut-in Prediction (Classification)**
    *   **Why it's simple:** This is a **classification problem** aiming to predict a binary outcome (e.g., "shut-in" or "producing"). The task involves analyzing historical operational data to identify patterns leading to a well shut-in.
    *   **Data needs:** Historical operational data for wells (e.g., rates, pressures), clearly labeled with their status (shut-in/producing). The **Volve Field Data** is an excellent public dataset resource for this.
    *   **Relevant ML techniques:** Simple **Classification Algorithms** such as **Logistic Regression**, **Decision Trees**, or basic **Random Forests**. These are straightforward to implement using the **Scikit-learn** library.

2.  **Virtual Metering (Regression)**
    *   **Why it's simple:** This involves **predicting a continuous numerical value** (e.g., oil, gas, or water flow rates) based on other readily available sensor data like pressure and temperature. The course provides a **case study on data-driven virtual metering** with explicit Python code examples for creating input features, splitting data, and building a Random Forest estimator.
    *   **Data needs:** Historical data from wellhead sensors (pressure, temperature) correlated with measured flow rates from well tests. A sample `Well_Rates.csv` is used in the provided examples.
    *   **Relevant ML techniques:** **Regression Algorithms** like **Linear Regression**, **Random Forests**, or **Support Vector Regression (SVR)**. **Pandas** is essential for data handling, and **Matplotlib/Seaborn** for visualization.

3.  **Log Permeability Prediction (Regression)**
    *   **Why it's simple:** This is a classic **regression problem** in the Upstream sector, where the objective is to predict a continuous numerical value (permeability) from other well log measurements.
    *   **Data needs:** Well log data (e.g., gamma ray, resistivity, density, neutron porosity) and corresponding core-measured permeability values. For a simple project, a small, pre-cleaned dataset could be used. A sample 'spwla_volve_data.csv` is used in the provided examples.
    *   **Relevant ML techniques:** **Regression models** such as **Random Forest** or **Artificial Neural Networks (ANN)**.

4.  **Production Profile Clustering (Unsupervised Learning)**
    *   **Why it's simple:** This is an **unsupervised learning** task, meaning it doesn't require pre-labeled data, which can simplify the initial data preparation significantly for beginners. The goal is to group wells that exhibit similar production decline behaviors.
    *   **Data needs:** Historical daily or monthly production rates (oil, gas, water) for multiple wells. The **Volve Field Data** is suitable.
    *   **Relevant ML techniques:** **Clustering Algorithms** like **K-means clustering** or **DBSCAN**.

5.  **Simple Anomaly Detection in Well Sensor Data (Statistical Anomaly Detection)**
    *   **Why it's simple:** This involves identifying unusual behavior in sensor readings. For a beginner, this can start with simple statistical methods (e.g., setting thresholds based on historical mean/standard deviation) or basic **unsupervised learning** techniques like **Isolation Forest** to identify outliers.
    *   **Data needs:** Time-series data from well sensors (e.g., pressure, temperature, or flow rates) where "normal" behavior can be established. This could be derived from **Volve Field Data** or simulated.
    *   **Relevant ML techniques:** Simple statistical methods or **Anomaly Detection Algorithms** like **Isolation Forest** or **One-Class SVM** (from Scikit-learn).

#### Downstream/Midstream Examples (5 Use Cases)

1.  **Anomaly Detection in SCADA Systems (Unsupervised Anomaly Detection)**
    *   **Why it's simple:** This is a direct application of anomaly detection to sensor data. The goal is to identify deviations from normal behavior in SCADA system readings to ensure asset integrity.
    *   **Data needs:** Time-series data from SCADA sensors (e.g., pressure, temperature, flow, vibration). Simulated data or publicly available industrial sensor datasets can be used.
    *   **Relevant ML techniques:** **Unsupervised Anomaly Detection Algorithms** such as **Isolation Forest** or **One-Class SVM**.

2.  **Flow Rate Forecasting in Transportation Networks (Time-Series Forecasting)**
    *   **Why it's simple:** This is a **time-series forecasting** problem, predicting future continuous values based on historical, time-ordered data. It is explicitly mentioned as a Midstream workflow.
    *   **Data needs:** Historical flow rates (e.g., pipeline throughput, vehicle traffic counts). Public transportation datasets are available on platforms like Kaggle.
    *   **Relevant ML techniques:** **Time-Series Forecasting Algorithms** like **ARIMA**, **Exponential Smoothing**, or simple **Recurrent Neural Networks (RNNs)**.

3.  **Demand Forecasting at Refineries (Time-Series Forecasting)**
    *   **Why it's simple:** Similar to flow rate forecasting, this is a **time-series forecasting** task within the Downstream sector. It involves predicting future product demand using historical data.
    *   **Data needs:** Historical product sales or consumption data from a refinery or distribution network. Generic demand data can be found on public data platforms.
    *   **Relevant ML techniques:** **Time-Series Models** like **ARIMA**, **Prophet**, or simple **RNNs**.

4.  **Energy Consumption Modeling (Regression)**
    *   **Why it's simple:** This Downstream use case involves **predicting a continuous numerical value** (energy consumption) based on various plant operational parameters or environmental factors. It's essentially a regression problem.
    *   **Data needs:** Historical energy consumption data (e.g., electricity, fuel) paired with relevant input features such as production volume, temperature, or equipment status. Public datasets like the UCI Combined Cycle Power Plant (CCPP) dataset are available.
    *   **Relevant ML techniques:** **Regression Models** such as **Linear Regression**, **Random Forests**, or simple **Artificial Neural Networks (ANN)**.

5.  **Quality Prediction in Process Streams (Regression/Classification)**
    *   **Why it's simple:** This Downstream application focuses on predicting product quality attributes (continuous values like sulfur content or binary "on-spec/off-spec" categories) from real-time process parameters. It can be approached as either a regression or a classification problem.
    *   **Data needs:** Historical process data (e.g., temperatures, pressures, flow rates) from a refinery unit, paired with lab-measured product quality data. Simulated data or open-source chemical process datasets can be used.
    *   **Relevant ML techniques:** **Regression Algorithms** (for continuous quality prediction) or **Classification Algorithms** (for categorical quality prediction like on-spec/off-spec) such as **Random Forests**, **SVMs**, or simple **ANNs**.

#### Enterprise & Cross-Functional Examples (10 Use Cases)

1.  **Simple Text Summarization for Daily Reports (NLG)**
    *   **Why it's simple:** This leverages **Natural Language Generation (NLG)**, a subset of Generative AI, to generate concise summaries. For a quick build, one could use a pre-trained Large Language Model (LLM) through a framework like **LangChain** and a local model (e.g., Mistral via Ollama as demonstrated in the RAG pipeline examples).
    *   **Data needs:** Short, unstructured text documents such as daily operational logs, shift reports, or incident summaries (can be simulated or small public text datasets).
    *   **Relevant ML techniques:** Application of pre-trained **LLMs** for summarization, via libraries like **LangChain**.

2.  **Document Classification for Regulatory Compliance (NLP/Classification)**
    *   **Why it's simple:** This is a **classification problem** applying **Natural Language Processing (NLP)** to categorize text documents (e.g., as "safety report," "environmental permit," "contract"). This helps in organizing and retrieving compliance-related documents.
    *   **Data needs:** A small dataset of text documents with predefined categories. This can be simulated or a subset of publicly available regulatory documents (e.g., SEC filings for energy companies, if a simpler subset is available).
    *   **Relevant ML techniques:** **Text Classification Algorithms** like **Naïve Bayes**, **SVM**, or **Logistic Regression** (from Scikit-learn).

3.  **Simple Market Trend Analysis for Crude/Product Pricing (Regression/Time-Series Forecasting)**
    *   **Why it's simple:** This involves predicting market prices, which is a **regression** or **time-series forecasting** problem. For a beginner, predicting the next day's closing price for crude oil based on historical data would be a simple starting point.
    *   **Data needs:** Historical daily or weekly crude oil/product prices. The **Energy Information Agency (EIA)** provides abundant public energy data.
    *   **Relevant ML techniques:** **Linear Regression**, simple **Time-Series Forecasting Algorithms** like **ARIMA** (if time-series features are prepared), or **Random Forests**.

4.  **Employee Skill Keyword Extraction (NLP)**
    *   **Why it's simple:** This uses **NLP** to extract key skills from text, which is a foundational NLP task. It doesn't require a predictive model initially, just text processing.
    *   **Data needs:** Textual descriptions of employee skills, resumes, or job descriptions (can be simulated or generic public data).
    *   **Relevant ML techniques:** Basic **Text Preprocessing** techniques (tokenization, lowercasing, stop-word removal) and simple **Keyword Extraction** (e.g., frequency analysis or using pre-defined lists).

5.  **Simplified Fraud Detection (Anomaly Detection)**
    *   **Why it's simple:** This applies **anomaly detection** to a simulated or simplified dataset to identify unusual patterns that might indicate fraud.
    *   **Data needs:** A small, simulated dataset of "transaction" data with numerical features (e.g., amount, frequency) where some data points are designed as outliers.
    *   **Relevant ML techniques:** **Anomaly Detection Algorithms** like **Isolation Forest** or **One-Class SVM**.

6.  **Customer Feedback Sentiment Analysis (NLP/Classification)**
    *   **Why it's simple:** This involves classifying the sentiment of text (e.g., positive, negative, neutral), which is a common **NLP classification** task.
    *   **Data needs:** Small dataset of simulated or generic public customer reviews/feedback, labeled with sentiment.
    *   **Relevant ML techniques:** **Text Classification Algorithms** like **Naïve Bayes** or **Logistic Regression**.

7.  **Simple Resource Allocation Optimization (Optimization)**
    *   **Why it's simple:** The concept of **optimization** is covered, with simple examples using `scipy.optimize.minimize` or `linprog`. A beginner can adapt these examples to a small, specific resource allocation scenario.
    *   **Data needs:** Defined objective function (e.g., minimize cost, maximize profit) and a few decision variables and constraints (e.g., budget, resource availability), which can be conceptualized and defined numerically.
    *   **Relevant ML techniques:** Application of **Optimization Algorithms** like **Linear Programming** or **Scipy's `minimize` function**.

8.  **Basic Data Quality Check - Outlier Detection (Anomaly Detection)**
    *   **Why it's simple:** This is a direct application of anomaly detection focused on identifying data quality issues within a single feature. It is a foundational step in Data Cleaning.
    *   **Data needs:** Any publicly available numerical dataset, such as a single column from the **Volve Field Data** (e.g., daily oil production).
    *   **Relevant ML techniques:** Simple **Statistical Process Control (SPC) methods** (e.g., z-score, IQR) or **Anomaly Detection Algorithms** like **Isolation Forest**.

9.  **Predicting Project Success/Failure (Classification)**
    *   **Why it's simple:** This is a binary **classification problem** (success/failure) that demonstrates how AI/ML can contribute to "better informed decision-making".
    *   **Data needs:** A small, simulated dataset of past projects with relevant features (e.g., project duration, budget, team size, number of change orders) and a clear "success" or "failure" label.
    *   **Relevant ML techniques:** **Classification Algorithms** such as **Logistic Regression**, **Decision Trees**, or **Random Forests**.

10. **Simple Equipment Maintenance Classification (Classification)**
    *   **Why it's simple:** While "predictive maintenance" is complex, a simplified version focuses on a single equipment and classifies whether it needs maintenance or not (binary classification) based on a few key parameters.
    *   **Data needs:** A small, simulated dataset of equipment readings (e.g., temperature, vibration levels, runtime hours) and a corresponding label indicating if maintenance was performed or required.
    *   **Relevant ML techniques:** **Classification Algorithms** like **Logistic Regression**, **Decision Trees**, or **K-Nearest Neighbor (KNN)**.


11. **Retrieval-Augmented Generation (RAG) Pipelines** 
    ◦ LangChain is used to build lightweight RAG pipelines that can retrieve information from documents to provide more informed and contextual responses.
    ◦ This typically involves:
        ▪ Loading a local LLM (e.g., using Ollama's Mistral model).
        ▪ Loading PDF documents using tools like PyMuPDFLoader.
        ▪ Splitting documents into chunks (e.g., using RecursiveCharacterTextSplitter with defined chunk_size and chunk_overlap).
        ▪ Embedding and storing these chunks in a vector store (e.g., FAISS) using embedding models (e.g., OllamaEmbeddings).
        ▪ Setting up a retriever and a RetrievalQA chain to enable asking questions directly about the content of the PDF documents. This allows the LLM to provide answers grounded in the specific source material.

12. **Agentic Workflows and Automation**
    ◦ LangChain forms a "brain" or "Agent Layer" in agent-driven operational frameworks.
    ◦ It's used for intent classification, understanding user prompts, and mapping them to predefined actions.
    ◦ LangChain agents are capable of tool use, which involves executing tasks such as:
        ▪ Querying GitHub.
        ▪ Triggering Continuous Integration/Continuous Deployment (CI/CD) processes.
        ▪ Fetching ESG (Environmental, Social, and Governance) metrics.
        ▪ Analyzing codebases, extracting documentation, or answering technical questions via a LangChain-GitHub Agent.
        ▪ Interacting with external tools and APIs, including monitoring systems (like Grafana, Prometheus, Azure Monitor), Jupyter Notebooks (to trigger runs or fetch results), cloud infrastructure (to provision resources or scale workloads), and ESG dashboards (to query sustainability metrics and generate reports).
    ◦ Agents can maintain memory and context across interactions, which is crucial for tasks like tracking a release cycle or anomaly investigation.

13. **Automated Report Generation**
    ◦ LangChain supports automated report generation by enabling LLM-powered summarization and RAG to combine structured data (like Key Performance Indicators) with unstructured narratives (such as logs, emails, and free-text reports) for contextual reporting.

14. **Regulatory Compliance**
    ◦ It assists in document classification and natural language programming for regulatory compliance by using RAG for contextual report synthesis from structured KPIs and unstructured logs.

15. **Market Analysis**
    ◦ LangChain, as part of Generative AI with RAG, can synthesize structured and unstructured data to provide contextual pricing insights in AI-driven market analysis for crude/product pricing.