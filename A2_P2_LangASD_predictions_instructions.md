Welcome to the second exciting part of the Language Development in ASD exercise
-------------------------------------------------------------------------------

In this exercise we will delve more in depth with different practices of model comparison and model selection, by first evaluating your models from last time against some new data. Does the model generalize well? Then we will learn to do better by cross-validating models and systematically compare them.

The questions to be answered (in a separate document) are: 1- Discuss the differences in performance of your model in training and testing data 2- Which individual differences should be included in a model that maximizes your ability to explain/predict new data? 3- Predict a new kid's performance (Bernie) and discuss it against expected performance of the two groups

Learning objectives
-------------------

-   Critically appraise the predictive framework (contrasted to the explanatory framework)
-   Learn the basics of machine learning workflows: training/testing, cross-validation, feature selections

Let's go
--------

N.B. There are several datasets for this exercise, so pay attention to which one you are using!

1.  The (training) dataset from last time (the awesome one you produced :-) ).
2.  The (test) datasets on which you can test the models from last time:

-   Demographic and clinical data: <https://www.dropbox.com/s/ra99bdvm6fzay3g/demo_test.csv?dl=1>
-   Utterance Length data: <https://www.dropbox.com/s/uxtqqzl18nwxowq/LU_test.csv?dl=1>
-   Word data: <https://www.dropbox.com/s/1ces4hv8kh0stov/token_test.csv?dl=1>

### Exercise 1) Testing model performance

How did your models from last time perform? In this exercise you have to compare the results on the training data () and on the test data. Report both of them. Compare them. Discuss why they are different.

-   recreate the models you chose last time (just write the model code again and apply it to your training data (from the first assignment))
-   calculate performance of the model on the training data: root mean square error is a good measure. (Tip: google the function rmse())
-   create the test dataset (apply the code from assignment 1 to clean up the 3 test datasets)
-   test the performance of the models on the test data (Tips: google the functions "predict()")
-   optional: predictions are never certain, can you identify the uncertainty of the predictions? (e.g. google predictinterval())

Part 1
======

``` r
# Load training Data
LU_train <- read.csv("LU_train.csv")
LU_test <- read.csv("LU_test.csv")
demo_train <- read.csv("demo_train.csv")
demo_test <- read.csv("demo_test.csv")
token_train <- read.csv("token_train.csv")
token_test <- read.csv("token_test.csv")


# Cleanup data
train_data <- CleanUpData(demo_train, token_train, LU_train)
test_data <- CleanUpData(demo_test, token_test, LU_test)

# Add visit^2
train_data$Visit2 <- train_data$Visit^2 
test_data$Visit2 <- test_data$Visit^2 

# Remove NAs
train_data <- subset(train_data, !is.na(CHI_MLU))
test_data <- subset(test_data, !is.na(CHI_MLU))


# Null model
null <- lmer(CHI_MLU ~ Visit + Diagnosis + (1 | Child.ID) + (1 + Visit | Child.ID), data=train_data)
```

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : unable to evaluate scaled gradient

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : Model failed to converge: degenerate Hessian with 1
    ## negative eigenvalues

    ## Warning: Model failed to converge with 1 negative eigenvalue: -4.1e-03

``` r
summary(null)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: 
    ## CHI_MLU ~ Visit + Diagnosis + (1 | Child.ID) + (1 + Visit | Child.ID)
    ##    Data: train_data
    ## 
    ## REML criterion at convergence: 602.2
    ## 
    ## Scaled residuals: 
    ##      Min       1Q   Median       3Q      Max 
    ## -2.46271 -0.56912 -0.08732  0.42215  2.70430 
    ## 
    ## Random effects:
    ##  Groups     Name        Variance Std.Dev. Corr 
    ##  Child.ID   (Intercept) 0.2332   0.4829        
    ##  Child.ID.1 (Intercept) 0.1356   0.3683        
    ##             Visit       0.0275   0.1658   -0.67
    ##  Residual               0.1613   0.4016        
    ## Number of obs: 352, groups:  Child.ID, 61
    ## 
    ## Fixed effects:
    ##              Estimate Std. Error       df t value Pr(>|t|)    
    ## (Intercept)   1.03831    0.12236 59.90865   8.486 7.39e-12 ***
    ## Visit         0.23353    0.02475 59.78969   9.437 1.89e-13 ***
    ## DiagnosisASD  0.29042    0.15435 58.14226   1.882   0.0649 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr) Visit 
    ## Visit       -0.405       
    ## DiagnossASD -0.662  0.000
    ## convergence code: 0
    ## unable to evaluate scaled gradient
    ## Model failed to converge: degenerate  Hessian with 1 negative eigenvalues

``` r
#- recreate the models you chose last time (just write the code again and apply it to Train Data)
model4 <- lmer(CHI_MLU ~ Diagnosis * verbalIQ1 * (Visit + Visit2) + (1 + Visit + Visit2 | Child.ID), data = train_data)
```

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : Model failed to converge with max|grad| = 0.0313095
    ## (tol = 0.002, component 1)

``` r
summary(model4)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: 
    ## CHI_MLU ~ Diagnosis * verbalIQ1 * (Visit + Visit2) + (1 + Visit +  
    ##     Visit2 | Child.ID)
    ##    Data: train_data
    ## 
    ## REML criterion at convergence: 494.2
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -2.6365 -0.5395 -0.0388  0.4497  3.2174 
    ## 
    ## Random effects:
    ##  Groups   Name        Variance  Std.Dev. Corr       
    ##  Child.ID (Intercept) 0.0472468 0.21736             
    ##           Visit       0.0533123 0.23089  -0.24      
    ##           Visit2      0.0009169 0.03028  -0.20 -0.90
    ##  Residual             0.1192993 0.34540             
    ## Number of obs: 352, groups:  Child.ID, 61
    ## 
    ## Fixed effects:
    ##                                Estimate Std. Error        df t value
    ## (Intercept)                    0.683006   0.316585 64.303144   2.157
    ## DiagnosisASD                  -0.202018   0.576368 63.436476  -0.351
    ## verbalIQ1                      0.021753   0.016775 63.448352   1.297
    ## Visit                         -0.431913   0.226464 62.253103  -1.907
    ## Visit2                         0.052160   0.031243 61.508717   1.669
    ## DiagnosisASD:verbalIQ1        -0.019213   0.028593 63.482182  -0.672
    ## DiagnosisASD:Visit             0.213127   0.412974 61.839030   0.516
    ## DiagnosisASD:Visit2            0.065975   0.057211 61.564744   1.153
    ## verbalIQ1:Visit                0.041443   0.011942 60.534834   3.470
    ## verbalIQ1:Visit2              -0.004549   0.001646 59.728148  -2.763
    ## DiagnosisASD:verbalIQ1:Visit   0.007786   0.020487 61.893308   0.380
    ## DiagnosisASD:verbalIQ1:Visit2 -0.004316   0.002840 61.752243  -1.520
    ##                               Pr(>|t|)    
    ## (Intercept)                   0.034716 *  
    ## DiagnosisASD                  0.727123    
    ## verbalIQ1                     0.199425    
    ## Visit                         0.061111 .  
    ## Visit2                        0.100102    
    ## DiagnosisASD:verbalIQ1        0.504052    
    ## DiagnosisASD:Visit            0.607640    
    ## DiagnosisASD:Visit2           0.253291    
    ## verbalIQ1:Visit               0.000965 ***
    ## verbalIQ1:Visit2              0.007592 ** 
    ## DiagnosisASD:verbalIQ1:Visit  0.705199    
    ## DiagnosisASD:verbalIQ1:Visit2 0.133733    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr) DgnASD vrbIQ1 Visit  Visit2 DgASD:IQ1 DgASD:V DASD:V2
    ## DiagnossASD -0.549                                                      
    ## verbalIQ1   -0.922  0.506                                               
    ## Visit       -0.805  0.442  0.739                                        
    ## Visit2       0.684 -0.376 -0.626 -0.961                                 
    ## DgnsASD:IQ1  0.541 -0.953 -0.587 -0.434  0.367                          
    ## DgnssASD:Vs  0.442 -0.805 -0.405 -0.548  0.527  0.767                   
    ## DgnssASD:V2 -0.374  0.684  0.342  0.525 -0.546 -0.652    -0.961         
    ## vrblIQ1:Vst  0.743 -0.408 -0.802 -0.922  0.885  0.470     0.505  -0.484 
    ## vrblIQ1:Vs2 -0.630  0.346  0.678  0.886 -0.922 -0.398    -0.486   0.503 
    ## DgASD:IQ1:V -0.433  0.767  0.467  0.537 -0.516 -0.804    -0.954   0.917 
    ## DASD:IQ1:V2  0.365 -0.652 -0.393 -0.514  0.534  0.683     0.917  -0.954 
    ##             vrIQ1:V vIQ1:V2 DgASD:IQ1:V
    ## DiagnossASD                            
    ## verbalIQ1                              
    ## Visit                                  
    ## Visit2                                 
    ## DgnsASD:IQ1                            
    ## DgnssASD:Vs                            
    ## DgnssASD:V2                            
    ## vrblIQ1:Vst                            
    ## vrblIQ1:Vs2 -0.961                     
    ## DgASD:IQ1:V -0.583   0.560             
    ## DASD:IQ1:V2  0.557  -0.580  -0.961     
    ## convergence code: 0
    ## Model failed to converge with max|grad| = 0.0313095 (tol = 0.002, component 1)

``` r
model5 <- lmer(CHI_MLU ~ Diagnosis + verbalIQ1 + Visit + Visit2 + Diagnosis:verbalIQ1 + Diagnosis:Visit +  verbalIQ1:Visit + verbalIQ1:Visit2 + Diagnosis:verbalIQ1:Visit + (1 + Visit + Visit2 | Child.ID), data = train_data)
```

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : unable to evaluate scaled gradient

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : Model failed to converge: degenerate Hessian with 1
    ## negative eigenvalues

    ## Warning: Model failed to converge with 1 negative eigenvalue: -2.3e+00

``` r
summary(model5)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: 
    ## CHI_MLU ~ Diagnosis + verbalIQ1 + Visit + Visit2 + Diagnosis:verbalIQ1 +  
    ##     Diagnosis:Visit + verbalIQ1:Visit + verbalIQ1:Visit2 + Diagnosis:verbalIQ1:Visit +  
    ##     (1 + Visit + Visit2 | Child.ID)
    ##    Data: train_data
    ## 
    ## REML criterion at convergence: 481.2
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -2.6280 -0.5553 -0.0658  0.4649  3.2451 
    ## 
    ## Random effects:
    ##  Groups   Name        Variance  Std.Dev. Corr       
    ##  Child.ID (Intercept) 0.0431245 0.20766             
    ##           Visit       0.0561458 0.23695  -0.24      
    ##           Visit2      0.0009969 0.03157  -0.18 -0.91
    ##  Residual             0.1195765 0.34580             
    ## Number of obs: 352, groups:  Child.ID, 61
    ## 
    ## Fixed effects:
    ##                                Estimate Std. Error         df t value
    ## (Intercept)                    0.829368   0.292505 166.785714   2.835
    ## DiagnosisASD                  -0.647995   0.419115  59.647832  -1.546
    ## verbalIQ1                      0.010113   0.015262 158.903025   0.663
    ## Visit                         -0.581778   0.193848 128.144695  -3.001
    ## Visit2                         0.073705   0.026404  87.377257   2.791
    ## DiagnosisASD:verbalIQ1         0.009962   0.020824  60.032304   0.478
    ## DiagnosisASD:Visit             0.669161   0.114017  56.588095   5.869
    ## verbalIQ1:Visit                0.053353   0.009805 131.159560   5.441
    ## verbalIQ1:Visit2              -0.006262   0.001327  87.255539  -4.718
    ## DiagnosisASD:verbalIQ1:Visit  -0.022033   0.005661  56.769556  -3.892
    ##                              Pr(>|t|)    
    ## (Intercept)                  0.005143 ** 
    ## DiagnosisASD                 0.127369    
    ## verbalIQ1                    0.508498    
    ## Visit                        0.003233 ** 
    ## Visit2                       0.006443 ** 
    ## DiagnosisASD:verbalIQ1       0.634116    
    ## DiagnosisASD:Visit           2.41e-07 ***
    ## verbalIQ1:Visit              2.51e-07 ***
    ## verbalIQ1:Visit2             8.97e-06 ***
    ## DiagnosisASD:verbalIQ1:Visit 0.000264 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr) DgnASD vrbIQ1 Visit  Visit2 DgASD:IQ1 DASD:V vrIQ1:V
    ## DiagnossASD -0.435                                                     
    ## verbalIQ1   -0.930  0.410                                              
    ## Visit       -0.771  0.134  0.708                                       
    ## Visit2       0.617 -0.004 -0.562 -0.947                                
    ## DgnsASD:IQ1  0.430 -0.953 -0.478 -0.132  0.003                         
    ## DgnssASD:Vs  0.323 -0.733 -0.306 -0.186  0.010  0.699                  
    ## vrblIQ1:Vst  0.730 -0.130 -0.759 -0.944  0.891  0.151     0.181        
    ## vrblIQ1:Vs2 -0.583  0.004  0.592  0.897 -0.947 -0.004    -0.011 -0.941 
    ## DgASD:IQ1:V -0.321  0.700  0.357  0.184 -0.011 -0.734    -0.953 -0.212 
    ##             vIQ1:V2
    ## DiagnossASD        
    ## verbalIQ1          
    ## Visit              
    ## Visit2             
    ## DgnsASD:IQ1        
    ## DgnssASD:Vs        
    ## vrblIQ1:Vst        
    ## vrblIQ1:Vs2        
    ## DgASD:IQ1:V  0.014 
    ## convergence code: 0
    ## unable to evaluate scaled gradient
    ## Model failed to converge: degenerate  Hessian with 1 negative eigenvalues

``` r
#- calculate performance of the model on the training data: root mean square error is a good measure. (Tip: google the function rmse())
rmse(predict(model4), train_data$CHI_MLU) # 0.303
```

    ## [1] 0.2968337

``` r
rmse(predict(model5), train_data$CHI_MLU) #0.303
```

    ## [1] 0.2979779

``` r
#- test the performance of the models on the test data (Tips: google the functions "predict()")
predictions <- predict(model4, test_data)
rmse(predictions,test_data$CHI_MLU)
```

    ## [1] 0.6778492

``` r
predictions2 <- predict(model5, test_data)
rmse(predictions2,test_data$CHI_MLU)
```

    ## [1] 0.6872557

``` r
predictions_null <- predict(null, test_data)
rmse(predictions_null,test_data$CHI_MLU)
```

    ## [1] 1.45377

``` r
#- optional: predictions are never certain, can you identify the uncertainty of the predictions? (e.g. google predictinterval())

predictInterval(model4, test_data, n.sims=500, level=0.9, stat="median")
```

    ##          fit      upr       lwr
    ## 1  1.2591580 1.889300 0.5849592
    ## 2  1.5624398 2.247514 0.9074064
    ## 3  1.6770699 2.291012 1.0156107
    ## 4  1.7832408 2.436981 1.0549212
    ## 5  1.7612170 2.416471 1.1739714
    ## 6  2.3110362 3.024557 1.6038751
    ## 7  3.0980135 3.765643 2.4018999
    ## 8  3.5996664 4.337392 2.9384051
    ## 9  3.8218672 4.482686 3.1231385
    ## 10 3.8364244 4.565012 3.1291289
    ## 11 3.5149938 4.306307 2.7716483
    ## 12 1.6037478 2.126126 0.9884905
    ## 13 2.0017342 2.639145 1.3497130
    ## 14 2.2417972 2.946676 1.5498679
    ## 15 2.3559287 2.910417 1.7089304
    ## 16 2.3379963 2.996541 1.6818123
    ## 17 2.1244483 2.775019 1.4895089
    ## 18 1.3052636 1.877299 0.6524577
    ## 19 2.0094775 2.641100 1.2995915
    ## 20 2.5992422 3.270638 1.9860098
    ## 21 3.0656954 3.744706 2.3840392
    ## 22 3.4198487 4.015646 2.7399580
    ## 23 3.6232715 4.279458 2.9314661
    ## 24 0.9102554 1.559386 0.2956769
    ## 25 1.3162376 1.988875 0.6367028
    ## 26 1.7484278 2.406392 1.0774123
    ## 27 2.1573863 2.780417 1.5529184
    ## 28 2.5893009 3.291777 1.8559704
    ## 29 3.0247090 3.757591 2.3032950
    ## 30 0.8988173 1.514895 0.3034176
    ## 31 1.3677452 2.089610 0.6886929
    ## 32 1.8251890 2.476272 1.2019515
    ## 33 2.2447349 2.883836 1.5916891
    ## 34 2.5622829 3.255994 1.9707970
    ## 35 2.9210335 3.623305 2.2295271

\[HERE GOES YOUR ANSWER\]

### Exercise 2) Model Selection via Cross-validation (N.B: ChildMLU!)

One way to reduce bad surprises when testing a model on new data is to train the model via cross-validation.

In this exercise you have to use cross-validation to calculate the predictive error of your models and use this predictive error to select the best possible model.

-   Use cross-validation to compare your model from last week with the basic model (Child MLU as a function of Time and Diagnosis, and don't forget the random effects!)
-   (Tips): google the function "createFolds"; loop through each fold, train both models on the other folds and test them on the fold)

-   Now try to find the best possible predictive model of ChildMLU, that is, the one that produces the best cross-validated results.

-   Bonus Question 1: What is the effect of changing the number of folds? Can you plot RMSE as a function of number of folds?
-   Bonus Question 2: compare the cross-validated predictive error against the actual predictive error on the test wwaq

``` r
#- Create the basic model of ChildMLU as a function of Time and Diagnosis (don't forget the random effects!).
# Already defined earlier

#- Make a cross-validated version of the model. (Tips: google the function "createFolds";  loop through each fold, train a model on the other folds and test it on the fold)
folds <- fold(train_data, k = 5, id_col = "Child.ID")
rmse_list <- c() # creating an empty list for rmse-values

# creating a for-loop and crossvalidation our 'best' model
for(i in 1:5){
  total_train <- filter(folds, .folds != i)
  total_test <- filter(folds, .folds == i)
  
  model <- lmer(CHI_MLU ~ 1 + Diagnosis * verbalIQ1 * (Visit +  Visit2) + (1 + Visit + Visit2 | Child.ID), total_train, REML=FALSE)
  
  rmse_list <- c(rmse_list,rmse(total_test$CHI_MLU, predict(model, total_test, allow.new.levels = T)))
}
```

    ## boundary (singular) fit: see ?isSingular
    ## boundary (singular) fit: see ?isSingular

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : Model failed to converge with max|grad| = 0.549517
    ## (tol = 0.002, component 1)

    ## boundary (singular) fit: see ?isSingular

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : Model failed to converge with max|grad| = 0.0539374
    ## (tol = 0.002, component 1)

``` r
mean(rmse_list)
```

    ## [1] 0.5422916

``` r
# Trying CVMS package 
# cv <- fold(train_data, k = 5, cat_col = 'Diagnosis', id_col = 'Child.ID') %>% cross_validate(null, folds_col # = '.folds',family='gaussian', REML = FALSE)

#- Report the results and comment on them. 
# Cross validation highly improves prediction error

#- Now try to find the best possible predictive model of ChildMLU, that is, the one that produces the best cross-validated results.
# Trying the more complicated model5
rmse_list_model5 <- c() # creating an empty list for rmse-values

# creating a for-loop
for(i in 1:5){
  total_train_model5 <- filter(folds, .folds != i)
  total_test_model5 <- filter(folds, .folds == i)
  
  model <- lmer(CHI_MLU ~ Diagnosis + verbalIQ1 + Visit + Visit2 + Diagnosis:verbalIQ1 + Diagnosis:Visit +  verbalIQ1:Visit + verbalIQ1:Visit2 + Diagnosis:verbalIQ1:Visit + (1 + Visit + Visit2 | Child.ID), data = total_train_model5, REML=FALSE)
  
  rmse_list_model5 <- c(rmse_list_model5, rmse(total_test_model5$CHI_MLU, predict(model, total_test_model5, allow.new.levels = T)))
}
```

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : Model failed to converge with max|grad| = 0.103822
    ## (tol = 0.002, component 1)

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : Model failed to converge with max|grad| = 0.0207934
    ## (tol = 0.002, component 1)

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : Model failed to converge with max|grad| = 2.50274 (tol
    ## = 0.002, component 1)

    ## boundary (singular) fit: see ?isSingular

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : unable to evaluate scaled gradient

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : Model failed to converge: degenerate Hessian with 1
    ## negative eigenvalues

    ## Warning: Model failed to converge with 1 negative eigenvalue: -1.5e+03

``` r
mean(rmse_list_model5)
```

    ## [1] 0.5446822

``` r
# Bonus Question 1: What is the effect of changing the number of folds? Can you plot RMSE as a function of number of folds?
# Bonus Question 2: compare the cross-validated predictive error against the actual predictive error on the test data
```

\[HERE GOES YOUR ANSWER\]

### Exercise 3) Assessing the single child

Let's get to business. This new kiddo - Bernie - has entered your clinic. This child has to be assessed according to his group's average and his expected development.

Bernie is one of the six kids in the test dataset, so make sure to extract that child alone for the following analysis.

You want to evaluate:

-   how does the child fare in ChildMLU compared to the average TD child at each visit? Define the distance in terms of absolute difference between this Child and the average TD.

-   how does the child fare compared to the model predictions at Visit 6? Is the child below or above expectations? (tip: use the predict() function on Bernie's data only and compare the prediction with the actual performance of the child)

\[HERE GOES YOUR ANSWER\]

### OPTIONAL: Exercise 4) Model Selection via Information Criteria

Another way to reduce the bad surprises when testing a model on new data is to pay close attention to the relative information criteria between the models you are comparing. Let's learn how to do that!

Re-create a selection of possible models explaining ChildMLU (the ones you tested for exercise 2, but now trained on the full dataset and not cross-validated).

Then try to find the best possible predictive model of ChildMLU, that is, the one that produces the lowest information criterion.

-   Bonus question for the optional exercise: are information criteria correlated with cross-validated RMSE? That is, if you take AIC for Model 1, Model 2 and Model 3, do they co-vary with their cross-validated RMSE?

### OPTIONAL: Exercise 5): Using Lasso for model selection

Welcome to the last secret exercise. If you have already solved the previous exercises, and still there's not enough for you, you can expand your expertise by learning about penalizations. Check out this tutorial: <http://machinelearningmastery.com/penalized-regression-in-r/> and make sure to google what penalization is, with a focus on L1 and L2-norms. Then try them on your data!
