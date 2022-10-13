# **Chances of surviving the Titanic!**
---------------------------
* Built a machine learning model to classify who survived or not (77.03% prediction accuracy).
* Dealt with 3 types of Data cleaning problems through median/mode imputation and dropping data then dealt with outliers.
* Engineered 2 new features by extracting name titles from name strings and combining 2 numerical features into a categorical one.
* Optimized 7 different classifiers using gridsearch then deployed an Ensemble model using a voting classifier on their results.
* Achieved 83.51% test accuracy on the voting classifier.
![Ensamble Model](https://i.imgur.com/x2GZbhc.png "final_model")


## **Code and resources used**
---------------------------
**Python Version:** 3.9
**Packages:** pandas, numpy, matplotlib, seaborn, sklearn, collections, xgboost
**Outliers detection:** https://www.kaggle.com/code/yassineghouzam/titanic-top-4-with-ensemble-modeling#2.2-Outlier-detection
**Cross-Validate:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
**Plotting Learning Curve:** https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

## **Ultimate goal of the analysis**
---------------------------
- Build a machine learning model to classify who survived or not.

## **Bonus Quests on the way**
---------------------------
+ Did women and children have priority? (**Yes**)
+ What were the chances of families surviving? (**Yes, 3 members to be specific**)
+ Did the rich have higher chance of surviving? (**Yes**)

## **Data**
--------------

### **Files:**
---------------------------
* 'train.csv': labeled data
* 'test.csv': prediction target

### **Features:**
--------------
| Feature | Description | DType | Variable Type | Comments |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| `Survived` | Passenger survived or not | int64 | Categorical | 0 = No, 1 = Yes |
| `Pclass` | Ticket class | int64 | Categorical | 1 = 1st, 2 = 2nd, 3 = 3rd |
| `Embarked` | Port of Embarkation | object | Categorical | C = Cherbourg, Q = Queenstown, S = Southampton |
| `Sex` | Male, Female | object | Categorical | - |
| `Age` | Age in years | float64 | Numerical | - |
| `SibSp` | # of siblings / spouses aboard the Titanic | int64 | Numerical | - |
| `Parch` | # of parents / spouses aboard the Titanic | int64 | Numerical | - |
| `Fare` | Passenger fare | float64 | Numerical | - |
| `Cabin` | Cabin code | object | Combined | - |
| `Ticket` | Ticket code | object | Combined | - |

### **Engineered Features:**
--------------
| Feature | Description | DType | Variable Type | Comments |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| `Prefix` | Title extracted from `Name` col mapped into 4 categories | object | Categorical | Mr, Mrs, Master, Noble/Dr/Religion/Military |
| `FamilySize` | `Parch` + `SibSp` and mapped into 4 categories | object | Categorical | Alone, Small, Medium, Big |

--------------
--------------

## **Descriptive Analysis**
--------------
- **Data shape**
  1. `Name` feature could be useful later if we extract the Prefixes (titles).
  2. `PassengerId` feature wont be of any use.
  3. `Ticket` and `Cabin` features contain numbers prefixed by characters.


- **Summary statistics**
  1. We might have to deal with some missing values later by looking at `Age` feature.
  2. We dont have negative values in any col.
  3. `Fare` feature has a huge discrepancy between 75% quartile and max values indicating skewness and potential outliers.


- **Null values**
  1. We have alot of missing data.
  2. We will have to impute most of the missing data since the dataset isn't that big.

![Missing values 1](https://i.imgur.com/isIjiM6.png)
![Missing values 2](https://i.imgur.com/FLHYWPD.png)

- **Correlation Matrix**
  1. Decent positive correlation between `SibSp` and `Parch`, potentially indicates they could be feature engineered into 1 feature.
  2. Decent negative correlation between `Fare` and `Pclass`, makes sense the richer the better the class and potentially the older the age too which is why `Age` is also related.
  3. Decent negative correlation between `Suvived` and `Pclass` and decent positive correlation between `Survived` and `Fare` potentially indicates the rich had more survival chance.

![Heat Map](https://i.imgur.com/V8MziHn.png)


## **Exploratory Data Analysis**
--------------
- **Data Cleaning**
  1. Checked some distributions of the features with missing values.
  2. Decided to impute `Fare`,`Age` and `Embarked`.
  3. Decided to drop `Cabin`,`Ticket` and `PassengerId`.
  4. Extracted the `Prefix` out of the `Name` feature.
     + `Fare` > imputed using median of `Fare` feature of the same `Pclass`.
     + `Age` > imputed using the median of `Age` of the same `Prefix`, `Pclass`, `Parch` and `SibSp` if existed otherwise the median of the `Age` feature.
     + `Embarked` > imputed using the most frequent value (Mode imputing).


- **Distribution plots**
  1. Created 2 distribution plots for each feature:
     + Feature distribution plot.
     + Feature distribution by survival rates.

  2. `Fare`, `Parch` and `SibSp` are skewed and will need some preprocessing

![image](https://user-images.githubusercontent.com/88734429/195544610-cf7ee8ec-a7b5-4167-b18d-8fa270b79818.png)

![image](https://user-images.githubusercontent.com/88734429/195544916-10ac4d97-9bbb-4952-8b2a-9a38ca96f88b.png)

![image](https://user-images.githubusercontent.com/88734429/195545029-fae4e714-bda8-4a7c-8944-65d9a8e82ff8.png)

![image](https://user-images.githubusercontent.com/88734429/195545118-8e3446aa-e012-4e1b-a5dd-8c0062657b5d.png)


- **Survival Rates**
  + Created heatmaps for each feature by survival counts showing how each feature affected the chances of surviving.

![image](https://user-images.githubusercontent.com/88734429/195545418-39897a5d-c8b2-411d-a67d-efc898edff36.png)

![image](https://user-images.githubusercontent.com/88734429/195545670-5bfb459e-dd77-4c78-ab8f-87ea2ff4dd73.png)


- **Checking for Outliers**
  1. Created boxplots and noticed how some features had extreme outliers (`Age`, `Fare`).
  2. Implemented IQR method to filter out outliers.
  3. Manually checked outliers to check if any of them shouldnt be dropped.
  4. dropped the outliers.

![image](https://user-images.githubusercontent.com/88734429/195545754-3ec45846-486b-47ac-9e42-c2d5e9fd591f.png)


## **Feature Engineering**
--------------
1. **`Prefix` Feature**
   - Extracted `Prefix` from the title from the `Name` col.
   - Mapped `Prefix` into 4 categories (`Mr`,`Mrs`,`Master`,`Noble/Dr/Religion/Military`).
   - Answered some side quests about survival rates compared to women, children and rich people.

![image](https://user-images.githubusercontent.com/88734429/195545988-cbe60892-913b-40f1-befd-37159eb41ce2.png)
![image](https://user-images.githubusercontent.com/88734429/195546235-67af7497-0d4d-47f5-bdb6-43cb833321b1.png)


2. **`FamilySize` Feature**
   - Crafted `FamilySize` feature by adding `Parch` and `SibSp` and it turned out to have 10 categories.
   - Concluded the best surviving chance by familysize is a family of 3 members.
   - Mapped `FamilySize` into 4 categories:
     + `Alone` > 0 family members.
     + `Small` > 1 to 3 family members.
     + `Medium` > 4 to 5 family members.
     + `Big` > 6+ family members.
   - Dropped `Name`, `Parch` and `SibSp` cols.
   - 
![image](https://user-images.githubusercontent.com/88734429/195546431-509454ed-345e-4ddd-858b-c5e74992a1f9.png)
![image](https://user-images.githubusercontent.com/88734429/195546557-20ca8236-a08e-43a1-92fa-5dfd09600239.png)


3. **`Pclass`**
   - Converted dtype from int64 to str


## **Preprocessing**
--------------
- **Normalizing**
  1. Checked the numerical features `Age` and `Fare` distributions from earlier
  2. `Age` feature was normally distributed
  3. `Fare` feature was skewed and didnt have a normal distribution so I decided to normalize it using log normalization which turned out decent

![image](https://user-images.githubusercontent.com/88734429/195546732-172eda96-688f-4e1c-8391-abe75cb5592c.png)


- **Splitting**
  + Splitted the combined dataframe back into 2 dataframes (Labeled, unlabeled)
    1. Training data -> driven from the labeled data and split into **X_train**, **y_train**
    2. Testing data -> also driven from the labeled data and split into **X_test**, **y_test** (**30% of the training data**)
    3. Target data -> driven from the non labeled data -> **target_X** (the target to predict)


- **Transforming**
  + Defined a function to apply the transformations to keep the code DRY
  + Created 2 transformer objects (**OneHotEncoder**, **StandardScaler**)
  + Transformed the features of the 3 dataframes (**X_train**, **X_test**, **target_X**)


## **Modeling**
--------------
1. **Model baseline Evaluation**
   - Started things by creating a list of model names and objects then setup stratified KFolds for the cross validation.
   - Created a dataframe to store the results.
   - Evaluated each model's cross validation performance (Accuracy (%) +/- Error (%)).
   - Average accuracy across all classifiers means = **81.35% +/- 3.33%**.
   - Adaboosting and RF had noticeably lower error margin despite having no tuning at all.
   
   
2. **Model Tuning**
   - Defined 3 functions to help with tuning the models and plotting the learning curve / feature importance.
   - Average accuracy across all classifiers means = **82.829% +/- 2.94%**.
   - Managed to tune all models to increase all the model's accuracy while preventing overfitting.
   - GradientBoosting, XGBoost and RF are the top performers.


   
| Model | Pre Tuning Mean Accuracy (%) | Pre Tuning Mean Error (%) | Post Tuning Mean Accuracy (%) | Post Tuning Mean Error (%) |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| LogisticRegression | 82.32% | 3.45% | 82.661% | 3.214% |
| KNeighborsClassifier | 81.64% | 3.9% | 82.325% | 4.239% |
| SVC | 81.98% | 3.21% | 82.486% | 2.364% |
| RandomForestClassifier | 79.77% | 2.58% | 83.0% | 2.758% |
| AdaBoostClassifier | 80.95% | 2.42% | 81.473% | 3.449% |
| GradientBoostingClassifier | 81.98% | 4.68% | 84.187% | 2.11% |
| XGBClassifier | 80.79% | 3.06% | 83.675% | 2.48% |



3. **Ensemble Modeling**
   - Combined the tuned models using VotingClassifier.
   - Tuned the voting classfier using gridsearch to optimize all of its possible parameters.
   - Obtained overall final accuracy of **83.51% +/- 3.041%**

![Ensamble Model](https://i.imgur.com/x2GZbhc.png "final_model")
