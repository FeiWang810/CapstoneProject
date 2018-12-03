# Capstone Project

# Data
https://www.yelp.com/dataset/challenge

# Report
https://github.com/FeiWang810/CapstoneProject/blob/master/Capstone%20Report_Wei%20Hao%20Fei%20Wang.pdf

# Slides
https://github.com/FeiWang810/CapstoneProject/blob/master/Capstone%20Project%20Wei%20Hao_Fei%20Wang.pdf

# Content
## Fei_coding
1. Business Data clean
2. Classification Models
3. Review Analysis
4. Review Generator

## Wei_coding
1. Business Data Processing
2. Regression Models
3. Cross Validation 

# Run
## Fei_coding
### business_data_prediction
1. Download yelp data and unzip it, put the data in the 'dataPreprocessing' file.
2. Run the data Processing.
3. Run all the models in the 'ClassificationModel' file. All the result will save in the result files, text and graph. Such as:

```sh
python knn.py
```
### reviewExploration
1. Put the 'yelp_academic_dataset_business.json' in the datapreProcessing file.
2. Now you can run all the notebook file.
3. Train the LSTM model should on GPU, it will cost a long time.
```sh
python Training_LSTM.py
```
3. Run review generator. first download the pretrained model(https://drive.google.com/file/d/1-PNTwH2kw_mb6-Ni5z3DE3G0npI1lhyu/view?usp=sharing) in the same file.

```sh
python reviewGenerator.py
```
## Wei_coding
### Data Processing
1. Download yelp data and unzip it,and put the 'yelp_academic_dataset_business.json' in the datapreProcessing file.
2. Run the data processing in R.mnd, and EDA jupiter notebook.

### Buiness Data Predicitive Model
1. Use the 'business(1).csv' data in the Wei_coding file.
2. Run all the jupyter notebook file. 
3. SVC model should take a longer time to finish. 




