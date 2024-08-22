# FunctionalIterativeImputer

The scikit-learn version of IterativeImputer can't be used with a dataset that contains both categorical and quantitative features. An estimator should be used that predicts a quantitative variables with a regressor and predicts categorical features with a classifier.\
\
In this script you can find a modified version of IterativeImputer that passes feat_idx (the feature that is going to be predicted by neighbor_features). This way you can check if the feature is categorical or quantitative and use an appropriate estimator.\

\
# AdaptiveImputerEstimator
\
With this estimator you can pass a regressor and a classifier and the index of categorical features so in combination with the FunctionalIterativeImputer, the function uses the suitable estimator.
