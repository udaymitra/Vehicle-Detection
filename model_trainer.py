from sklearn.svm import LinearSVC
from training_data_supplier import get_train_test_data
import time

for spatial in [8, 16, 32, 64]:
    for histbins in [8, 16, 32, 64]:
        print("Spatial: %d, Histbins: %d"%(spatial, histbins))
        X_train, X_test, y_train, y_test = get_train_test_data(spatial=spatial, histbins=histbins)

        # print("Training classifier...")
        start_time = time.time()

        svc = LinearSVC()
        svc.fit(X_train, y_train)

        # print("Done training")
        elapsed_time = time.time() - start_time
        print("Total training time: %.2f seconds" % elapsed_time)

        test_accuracy = svc.score(X_test, y_test)
        print('Test Accuracy of SVC = ', test_accuracy)

        print('###%d,%d,%.4f'%(spatial, histbins, test_accuracy))
        # print('My SVC predicts: ', svc.predict(X_test[0:10]).reshape(1, -1))
        # print('For labels: ', y_test[0:10])