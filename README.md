# Data driven approach for forecasting employee absenteeism

Creating a model for forecasting employee absenteeism which helps HR's to take an advantage of making intelligent decisions.

Employee absenteeism can have significant financial and operational impacts on a business, including reduced productivity, increased costs, and lowered morale. A data-driven approach for forecasting employee absenteeism can help businesses anticipate and mitigate these impacts by identifying patterns and trends in employee absence data. By using machine learning algorithms to analyze historical absence data, businesses can gain insights into factors that contribute to employee absenteeism and develop proactive strategies to reduce absenteeism in the future.

## Features
Accepts user input for different features such as reason for absence, month value, day of the week, transportation expense, distance to work, age, daily work load average, body mass index, education level, number of children, and presence of pets.

Performs necessary preprocessing and scaling on the input data.

Uses a trained machine learning model (Logistic Regression) to make predictions.

Displays the predicted absenteeism value to the user.

## Technologies Used
Python

Flask (Web framework)

HTML

CSS

Scikit-learn (Machine learning library)

Pickle (Model serialization)

## Dataset
The machine learning model used in this project was trained on a dataset containing employee information, including absenteeism records.

## Model Training
The machine learning model was trained using scikit-learn on the provided dataset. The training process involved feature engineering, preprocessing, model selection, and evaluation. The trained model and associated scaler were serialized using pickle for deployment in the web application.

## Contributions
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.


