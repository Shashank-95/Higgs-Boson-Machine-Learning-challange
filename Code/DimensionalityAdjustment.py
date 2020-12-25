import utils
#Feature Dimensionality Adjustment- Analysis using PCA
def DimensionalityReduction(train_preprocessed, valtest_preprocessed):
    pca_model = utils.PCA()
    pca_out = pca_model.fit(train_preprocessed)
    #print("Variance Ratio:", pca_model.explained_variance_ratio_)

    num_pc = pca_out.n_features_
    pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
    #Eigen values for all the principal components
    #print(pca_model.explained_variance_)
    #cluster.screeplot(obj=[pc_list, pca_model.explained_variance_ratio_])

    #PCA scatter plot - biplot
    #pca_components = pca_model.components_
    #cluster.pcaplot(x=pca_components[0], y=pca_components[1],labels=FeatureMatrix.columns.values, var1=round(pca_model.explained_variance_ratio_[0]*100, 2),var2=round(pca_model.explained_variance_ratio_[1]*100, 2))
    #cluster.pcaplot(x=pca_components[0], y=pca_components[1], z=pca_components[2],labels=FeatureMatrix.columns.values, var1=round(pca_model.explained_variance_ratio_[0]*100, 2),var2=round(pca_model.explained_variance_ratio_[1]*100, 2), var3=round(pca_model.explained_variance_ratio_[2]*100, 2))

    #Apply Dimensionality Reduction: PCA
    pca = utils.PCA(n_components=15)
    PCATrain = pca.fit_transform(train_preprocessed)
    PCAval = pca.transform(valtest_preprocessed)
    return PCATrain, PCAval
