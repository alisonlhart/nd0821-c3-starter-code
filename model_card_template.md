# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was created by Alison Hart to predict a salary classification range for individuals based on certain attributes. The model was created from a Random Forest Classifier with the default random state of 42. 

## Intended Use
The intention of this model is to predict whether the salary of an input subject is greater than or less than 50k yearly. 


## Training Data
The training dataset is a split of publically available data from the United States Census Bureau. 

## Evaluation Data
The evaluation data comes from the other side of the split of publically available data from the Census Bureau. 

## Metrics
Metrics on the model overall:
- precision: 1.0
- recall: .9996807
- fbeta: .9998403

Slice model metrics are all located in `src/slice_data/slice_output.txt`. 

Examples of slices located in this output file are:
`race = White  | precision: 0.9510641378311857, recall: 0.9230012645777715, fbeta: 0.9368225898459783`
`race = Black  | precision: 0.9622641509433962, recall: 0.9224806201550387, fbeta: 0.9419525065963061`

## Ethical Considerations
Census data must be carefully considered to ensure there is enough representation across education, racial, geographical, and gender categories. Machine Learning has historically had poorly trained models when it comes to marginalized groups, so this must be considered. 

## Caveats and Recommendations
Feel free to adjust the testing data to check against other test cases. 