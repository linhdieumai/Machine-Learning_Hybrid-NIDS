

<h1 align="center"> Hybrid Network Intrusion Detection System </h1>
<h5 align="center"> Project Machine Learning - HUST (2025.2) <h5>

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview"> ➤ Overview</a></li>
    <li><a href="#project-files-description"> ➤ Project Files Description</a></li>
    <li><a href="#requirements"> ➤ Requirements </a></li>
    <li><a href="#execution-instructions"> ➤ Execution Instructions  </a></li>
    <li><a href="#experimental-results"> ➤ Experimental Results  </a></li>
    <li><a href="#contributors"> ➤ Contributors </a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- OVERVIEW -->
<h2 id="overview">Overview</h2>

<p align="justify"> 
  This project implements a Hybrid NIDS framework designed to handle modern network threats. By leveraging the NSL-KDD dataset, the system employs a two-tier approach:
</p>

<ul>
  <li><b>Supervised Learning</b> : High-accuracy classification of known attack signatures (e.g., Neptune, Smurf).</li>
 <li><b>Unsupervised Learning</b> : Detection of novel "Zero-day" anomalies based on statistical deviations and distance thresholds.</li>
</ul>
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PROJECT FILES DESCRIPTION -->
<h2 id="project-files-description">Project Files Description</h2>

<ul>
  <li><b>xử lý dữ liệu.py</b> - Data preprocessing: converts raw CSV datasets into optimized '.npy' arrays.</li>
  <li><b>BaoKmean.py</b> - Trains the K-Means unsupervised model and calculates the anomaly threshold.</li>
  <li><b>RandomForest.py</b> -   
Implementation and training of the Random Forest supervised classifier.</li>
  <li><b>XG_Boost.py</b> - XGBoost classifier script including hyperparameter tuning logic.</li>
  <li><b>SVM.py</b> - Support Vector Machine (SVM) implementation for supervised detection.</li>
  <li><b>ISFR.py</b> - Isolation Forest algorithm for unsupervised anomaly detection.</li>
</ul>

<h3>Integration files</h3>
<ul>
  <li><b>main_integration_*.py</b> - Entry points for the hybrid system (e.g., 'RF-Kmeans', 'XGB-ISFR').</li>
</ul>

<h3>Artifacts files</h3>
<ul>
  <li><b>*.pkl</b> - Saved pre-trained models (e.g., 'supervised_rf_model.pkl') for instant deployment.</li>
  <li><b>*.npy</b> - SPre-processed numerical data arrays for high-speed loading.</li>
  <li><b>*.csv</b> - Original NSL-KDD dataset files used for training and zero-day testing.</li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- REQUIREMENTS -->
<h2 id="requirements">Requirements</h2>

<p>Ensure you have Python installed, then run the following command to install the necessary libraries:</p>
<pre><code>pip install numpy pandas scikit-learn joblib xgboost scipy</code></pre>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- EXECUTION INSTRUCTIONS -->
<h2 id="execution-instructions">Execution Instructions</h2>

<div align="left">
  <p>Follow the steps below to prepare the data, train models, and deploy the hybrid system:</p>

<h3>Step 1: Data Preprocessing</h3>
  <p>The first step is to process the raw NSL-KDD datasets. This script converts <code>.csv</code> files into optimized <code>.npy</code> arrays for faster loading and training:</p>
  <pre><code>python "xử lý dữ liệu.py"</code></pre>

  <h3>Step 2: Individual Component Training (Optional)</h3>
  <p>If you wish to retrain individual models or update the <code>.pkl</code> artifacts, execute the specific algorithm scripts. For example, to train the K-Means anomaly detector and recalculate the threshold:
  </p>
  <pre><code>python BaoKmean.py</code></pre>
  <p><i>Note: This will generate or update files like <code>unsupervised_kmeans_model.pkl</code> in your working directory.</i></p>

  <h3>Step 3: Running the Hybrid System</h3>
  <p>The core of the project lies in its hybrid integration. You can choose from several predefined integration scenarios depending on the models you wish to combine:</p>
<ul>

 <li><b>XGBoost + Isolation Forest (Recommended):</b></li>
    <pre><code>python main_integration_XGB-ISFR.py</code></pre>
    
   <li><b>Random Forest + K-Means:</b></li>
    <pre><code>python main_integration_RF-Kmeans.py</code></pre>
    
   <li><b>SVM + Isolation Forest:</b></li>
    <pre><code>python main_integration_SVM-ISFR.py</code></pre>
  </ul>
  
  <p align="center">
    <i>Check the console output for real-time detection results and performance metrics (Accuracy, F1-Score, etc.).</i>
  </p>
</div>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- EXPERIMENTAL RESULTS -->
<h2 id="experimental-results">Experimental Results</h2>

<p>The system's performance was evaluated using the NSL-KDD test set. Below are the visual results for the XGBoost component, which demonstrated high accuracy in identifying various attack types:
</p>

<table border="0">
    <tr>
      <td>
        <p align="center"><b>Confusion Matrix</b></p>
        <img src="xgboost_confusion_matrix.png" width="400" alt="XGBoost Confusion Matrix">
      </td>
      <td>
        <p align="center"><b>Hyperparameter Tuning</b></p>
        <img src="xgboost_hyperparameter_tuning.png" width="400" alt="Tuning Process">
      </td>
    </tr>
  </table>
  
<p><i>The hybrid integration significantly reduces False Positives by cross-verifying supervised classifications with unsupervised anomaly scores.</i></p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CONTRIBUTORS -->
<h2 id="contributors">Contributors</h2>

<p>This project was developed by a team of students:
</p>

<table border="1" style="width: 60%; text-align: center; border-collapse: collapse;">
    <thead>
      <tr style="background-color: #f2f2f2;">
        <th>Full Name</th>
        <th>Student ID</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Thân Hoàng Bách</td>
        <td>202416778</td>
      </tr>
      <tr>
        <td>Nguyễn Bùi Gia Bảo</td>
        <td>202416779</td>
      </tr>
      <tr>
        <td>Nguyễn Minh Hằng</td>
        <td>202416792</td>
      </tr>
      <tr>
        <td>Mai Thị Diệu Linh</td>
        <td>202416807</td>
      </tr>
      <tr>
        <td>Nguyễn Anh Quân</td>
        <td>202416821</td>
      </tr>
    </tbody>
  </table>



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

