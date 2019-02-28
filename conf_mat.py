from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def conf_mat(train_pred_list,train_gt_list, val_pred_list, val_gt_list, test_p_list, test_g_list):
    print("Confusion Matrix:")

    # convert Train Pred into 1d array form
    Train_pred_1 = np.asarray(train_pred_list)
    print("Train_pred_1 shape:",np.shape(Train_pred_1[0]))

    TRaining = np.array([])
    for train in range(0,len(train_pred_list)):
        tr = np.argmax(Train_pred_1[train], axis=-1)
        TRaining = np.append(TRaining,tr)

    # convert Train gt into 1d array form
    TRaining_GT = np.array([])
    for train_gt in range(0, len(train_gt_list)):
        tr_gt = np.reshape(train_gt_list[train_gt], [-1])
        TRaining_GT = np.append(TRaining_GT, tr_gt)

    CM_TRAINING = confusion_matrix(y_true=TRaining_GT,y_pred=TRaining)
    print("Training: \n",CM_TRAINING)

    # convert Val Pred into 1d array form
    VALIDATION = np.array([])
    for val_pred in range(0,len(val_pred_list)):
        val = np.argmax(val_pred_list[val_pred], axis=-1)
        VALIDATION = np.append(VALIDATION,val)

    # convert Val gt into 1d array form
    VALIDATION_GT = np.array([])
    for val_gt in range(0, len(val_gt_list)):
        vl_gt = np.reshape(val_gt_list[val_gt], [-1])
        VALIDATION_GT = np.append(VALIDATION_GT, vl_gt)

    CM_VALIDATION = confusion_matrix(y_true=VALIDATION_GT, y_pred=VALIDATION)
    print("Validation: \n",CM_VALIDATION)

    # convert Test Pred into 1d array form
    TESTING = np.array([])
    for test_pred in range(0, len(test_p_list)):
        TESTING = np.append(TESTING, test_p_list[test_pred])

    # convert Test gt into 1d array form
    TESTING_GT = np.array([])
    for test_gt in range(0, len(test_g_list)):
        TESTING_GT = np.append(TESTING_GT, test_g_list[test_gt])

    CM_TESTING = confusion_matrix(y_true=TESTING_GT, y_pred=TESTING)
    print("Testing: \n",CM_TESTING)

    # plot confusion matrix as an image
    plt.matshow(CM_TRAINING)
    plt.matshow(CM_VALIDATION)
    plt.matshow(CM_TESTING)





