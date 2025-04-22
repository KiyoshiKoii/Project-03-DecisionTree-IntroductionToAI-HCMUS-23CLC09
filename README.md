# Decision Tree Classifier Experiments on Real-World Datasets
This project explores the use of `DecisionTreeClassifier` from `scikit-learn` across three real-world datasets for supervised classification tasks. The experiments include data preparation, model training, evaluation, and analysis of tree depth impact on accuracy.

## Datasets
1. **Breast Cancer Dataset**
   - Binary classification: Malignant vs. Benign
   - Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

2. **Wine Quality Dataset**
   - Multi-class classification (mapped to 3 groups: Low, Standard, High quality)
   - Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

3. **Pima Indians Diabetes Dataset**
   - Binary classification: Diabetic vs. Non-Diabetic
   - Source: Found via GitHub mirror of UCI dataset (https://github.com/jbrownlee/Datasets/blob/master/pima-indians-diabetes.data.csv)

## Key Tasks
- Preprocessing and stratified splitting into training/test sets (40/60, 60/40, 80/20, 90/10)
- Visualizing class distributions using pie charts
- Training decision trees with `entropy` criterion (information gain)
- Visualizing the decision trees using Graphviz
- Evaluating models with `classification_report` and `confusion_matrix`
- Investigating the effect of `max_depth` on accuracy

## Requirements
Install Graphviz and add it to your PATH. 
Install dependencies with:
```bash
pip install -r requirements.txt
