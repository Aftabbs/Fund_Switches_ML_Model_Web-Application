# ML Web Application

## Project Overview 

This project, is developed as a prototype proposal for Invergence Analytics, aims to predict managers who are likely to switch to other funds. The dataset, created by our data experts and SMEs, contains 120 features and 460,000 records. Due to the real-world nature of the data, it is highly imbalanced, with fewer instances of managers switching funds.   
      
The project uses an ensemble model combining three base classifiers: RandomForestClassifier, XGBClassifier, and LightGBMClassifier, integrated using a VotingClassifier. The primary metric for this classification problem is recall, crucial for our use case.
     
## Problem Statement

Predicting fund switches is a challenging task due to:   
- The highly imbalanced dataset
- The complex nature of the financial data
- there are very few switches observed in industry

## Solution

### Project Highlights
- Finalized an ensemble model combining RandomForestClassifier, XGBClassifier, and LightGBMClassifier.
- Achieved high accuracy with a focus on recall as the primary metric.

### Model Performance Metrics
- **Accuracy**: 97.62%
- **Precision**: 62.66%
- **Recall**: 65.88%
- **F1-Score**: 64.23%
- **ROC-AUC**: 94.9%

### Web Application
A demo ML web application was developed using Flask, HTML, and CSS to showcase the model. The application features:
- Input of the original dataset
- Internal training, preprocessing, and validation
- Display of model metrics results
- Option to download prediction results as an Excel file
![d2](https://github.com/Aftabbs/Fund_Switches_Model_ML_Web-Application/assets/112916888/05eec131-3cd2-479c-9163-8532a825d9bc)

![d3](https://github.com/Aftabbs/Fund_Switches_Model_ML_Web-Application/assets/112916888/57ae4dbc-bcf9-4b67-a27a-7513e608f65f)

**NOTE** - The Data and predictions used are just prototypes and  dummy

### Further Improvements
- Integrating additional models into the UI
- Enhancing the UI for better user experience
- Utilizing Django for large-scale deployment

### Libraries Used
- `sklearn`
- `pandas`
- `numpy`
- `openpyxl`
- `scipy`
- `xgboost`
- `lightgbm`
- `flask`

## Implementation

### Steps
1. **Data Preparation**: Handled imbalanced data and performed necessary preprocessing.
2. **Model Building**: Used ensemble learning with RandomForestClassifier, XGBClassifier, and LightGBMClassifier.
3. **Model Evaluation**: Focused on recall as the key metric.
4. **Web Application Development**: Built a Flask web application to demonstrate the model's capabilities.

## Results and Learnings

The Fund Switches Model achieved impressive accuracy and recall, highlighting the effectiveness of ensemble learning in handling imbalanced datasets. The web application provided a practical demonstration of the model's performance and potential for deployment.

## Future Work

Further enhancements can focus on:
- Improving model performance by fine-tuning parameters and exploring other classifiers.
- Expanding the web application with more features and a better user interface.
- Deploying the application on a larger scale using Django or other frameworks.

## Conclusion

The Fund Switches Model project successfully predicts managers likely to switch funds, leveraging an ensemble of powerful classifiers. The accompanying web application demonstrates the model's practical applications and potential for further development and deployment.

## Contact

For more information, please contact:
- Name: Mohammed Aftab
- Email: maftab@convergenceinc.com
- Organization: Invergence Analytics

  
![image](https://github.com/Aftabbs/Fund_Switches_Model_ML_Web-Application/assets/112916888/3dbcbb52-61e4-4db6-b92a-c9e94df3a6a3)

---

*This project is part of Invergence Analytics' ongoing efforts to utilize advanced machine learning techniques for predictive modeling in the finance industry.*
