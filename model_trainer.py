from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import svm
from training_data_supplier import get_train_test_data
import time

# Best parameters for feature extraction
# This is based on exploring parameter space for these variables
spatial = 16
histbins = 32
orientations = 12
pixels_per_cell = 16
cells_per_block = 2
transform_sqrt = False
#
# for orientations in [8, 12, 16]:
#     for pixels_per_cell in [4, 8, 16]:
#         for cells_per_block in [2]: #[2, 3, 4]:
#             for transform_sqrt in [False, True]:
# print("orientations: %d, pixels_per_cell: %d, cells_per_block: %d, transform_sqrt: %d"
#       %(orientations, pixels_per_cell, cells_per_block, transform_sqrt))


# Best parameters for SVM training
kernel = 'rbf'
C = 10
gamma = 0.001000

X_train, X_test, y_train, y_test = get_train_test_data(spatial=spatial, histbins=histbins,
       orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                                                       transform_sqrt=transform_sqrt)

# for kernel in ['linear', 'rbf']:
#     for c in [1, 10, 100, 1000]:
#         for gamma in [0.001, 0.0005, 0.0001]:
#             print("kernel=%s,c=%d,gamma=%f"%(kernel, c, gamma))

print("Training classifier...")
start_time = time.time()

svc = SVC(C=c, kernel=kernel, gamma=gamma)
svc.fit(X_train, y_train)

print("Done training")
elapsed_time = time.time() - start_time
print("Total training time: %.2f seconds" % elapsed_time)

test_accuracy = svc.score(X_test, y_test)
print('Test Accuracy of SVC = ', test_accuracy)
print('###%d,%d,%d,%d,%.4f'%(orientations, pixels_per_cell, cells_per_block, transform_sqrt, test_accuracy))
print('My SVC predicts: ', svc.predict(X_test[0:10]).reshape(1, -1))
print('For labels: ', y_test[0:10])