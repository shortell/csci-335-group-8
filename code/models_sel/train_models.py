from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import model_functions as mf


# code copied originally from jackson's
# mainly because we were going for similar things
# seperated a lot of the repeated code though

# these 2 things are repeated in modelfunctions and here

PCA_COMPONENTS = 400
RANDOMSTATE = 35

if __name__ == '__main__':
    # preprocessing
    X, row_ids = mf.load_embeddings_with_features()

    labels = mf.build_labels(row_ids)
    X, y   = mf.align_and_clean(X, labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    X_train, X_test, pca = mf.apply_pca(X_train, X_test, PCA_COMPONENTS)

    # training

    nn =  MLPClassifier(
                        hidden_layer_sizes=(1000,100),
                        activation='relu',
                        solver='adam',
                        max_iter=1000,
                        random_state= RANDOMSTATE
                        )
    


    model, scaler         = mf.train(X_train, y_train,nn)
    y_pred                = mf.evaluate(model, scaler, X_test, y_test)

    mf.plot_results(pca, y, y_test, y_pred)
