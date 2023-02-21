# Data-Science-Project

This project is about crime analysis and prediction in New York city. Using Streamlit framework, python and Machine Learning on our dataset we are able to predict the probability that a crime ocurs in spatial and temporal conditions.

###### The notebook "Data-exploration.ipynb" contains the exploration of the dataset using graphs. For example: a graph of number of crimes per bourough.
###### The notebook "Data-cleaning.ipynb" contains steps to clean our dataset.
###### The notebook "Crime-prediction-using-RF.ipynb" is about crime prediction using RandomForest model.
## To run the web app 
* $ streamlit run app.py

## The application page

This interface is the section that presents all charts that describe our cleaned data.

![269976679_382115040351359_7481065632652412970_n](https://user-images.githubusercontent.com/93519108/220284917-470cd19f-8a6e-4568-bdad-d905af65f6fd.png)

This interface is the section that presents the prediction part, where the user enter some informations such as date, place to go (the place is selected using a map. When the user click on the place in the map, its coordinates will be shows and used for the prediction process) etc... .
