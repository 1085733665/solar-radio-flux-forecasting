# solar-radio-flux-forecasting
In this project, we adopt the deep-learning Informer model based on the transformer architecture to predict medium-term F10.7 index, which uses 48 historical daily F10.7 index as inputs to directly forecast the following 1-27 days’ F10.7 index. 


# Requirements
* Python 3.6
* matplotlib == 3.1.1
* numpy == 1.19.4
* pandas == 0.25.1
* scikit_learn == 0.21.3
* torch == 1.8.0

# Run
python run.py
* A sample of the input file can be seen in  `/data` (It is worth noting that the amount of data in the input file is kept at 100, which can avoid the influence of normalization on the accuracy of the results.)
* A sample of the output file can be seen in  `/results`
* The trained model can be found in `/checkpoints`
# Result
The daily F10.7 indices used in this study are obtained from NASA OmniWeb, ranging from January 1, 1965, to December 31, 2021 which covers solar cycles 20-24. The entire dataset was split into 3 subsets with the ratios of 50%: 25%: 25%, i.e., the training set covering the period from January 1965 to January 1994, the validation set covering the period from January 1994 to December 2007, and the test set covering the period from January 2008 to December 2021.

The test results are in forecasting results during 20080101-20211231.csv

# Acknowledgments
We greatly acknowledge British Geological Survey (BGS), Collecte Localisation Satellites (CLS), and the Space Weather Prediction Center (SWPC), as well as the National Geophysical Data Center (NOAA) for providing the data necessary to carry out this work. We also thank Haoyi Zhou, the creator of the Informer model, for answering questions on the details and providing valuable insights.
* The observed values of F10.7 are downloaded from OMNIWeb service (https://omniweb.gsfc.nasa.gov/form/dx1.html).
* Informer original project (https://github.com/zhouhaoyi/Informer2020?tab=readme-ov-file).
