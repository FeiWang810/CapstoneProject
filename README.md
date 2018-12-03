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
3. Run review generator. First download the pretrained model in the same file.

```sh
python reviewGenerator.py
```
