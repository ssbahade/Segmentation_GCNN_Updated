import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
from imageio import imread
from skimage.transform import resize
import cv2
from skimage import color
import Preprocessing
from Create_patches import remove_unsized_patched
import pickle
from sklearn.model_selection import KFold
from del_ import color_image
from Test_sliding_window import test_image_into_patches
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support




import sys
sys.path.append('D:/PhD_Image_Data/PhD project/PycharmProjects/2D/data/vgg')
from Create_patches import sliding_window

#data_folder = 'D:\PhD_Image_Data\PhD project\PycharmProjects\2D\data\vgg\data'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(x_train, x_test, y_train, y_test,image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size, epochs, sess):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        '''
        image_paths = glob(os.path.join(data_folder, 'Images', '*.tif'))
        label_paths = {
            re.sub(r'.Label.png', '', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'Label', '*.png'))}
        #background_color = np.array([255, 0, 0])
        random.shuffle(image_paths)
        '''
        '''image_data, label_data = remove_unsized_patched(training_dir)          # this contain 414 image patches and 414 label patches
        #print("data_from_preprocessin image shape:{} and label shape:{}".format(np.shape(image_data),np.shape(label_data)))

        ########################################### kfold CROSS VALIDATION
        kf = KFold(n_splits=5)
        val = 1
        def gen():
             for train, test in kf.split(image_data):
                 x_train, x_test = image_data[train],image_data[test]
                 y_train, y_test = label_data[train], label_data[test]
                 yield x_train,y_train,x_test,y_test
        ########################################### kfold CROSS VALIDATION END
        for x_train,y_train,x_test,y_test in gen():
            print("Validation K-fold : {} start".format(val))'''
        for epoch in range(epochs):
            #print("Epoch {} Start :".format(epoch+1))
            x_train_ep, y_train_ep, x_test_ep, y_test_ep = x_train,y_train,x_test,y_test
            for batch_i in range(0, x_train_ep.shape[0], batch_size):
                images_ = []
                gt_images_ = []
                images_test = []
                gt_images_test = []
                for image_file in x_train[batch_i:batch_i+batch_size]:
                    images_.append(image_file)

                for label_file in y_train[batch_i:batch_i+batch_size]:
                    gt_images_.append(label_file)

                for image_file_test in x_test[batch_i:batch_i+batch_size]:
                    images_test.append(image_file_test)

                for label_file_test in y_test[batch_i:batch_i+batch_size]:
                    gt_images_test.append(label_file_test)

                #print("NUM:images_{}, NUM:gt_images_{},images_test{},gt_images_test{}".format(np.shape(images_),np.shape(gt_images_),np.shape(images_test),np.shape(gt_images_test)))
                yield np.array(images_), np.array(gt_images_), np.array(images_test), np.array(gt_images_test),int(x_train.shape[0]/batch_size)+1 , int(x_test.shape[0]/batch_size)+1, epoch
                    #print("Epoch {} Finished :".format(epoch+1))
            #print("Validation K-fold : {} end".format(val))
            #print("initialise global variables")
            #tf.global_variables_initializer()
            #val += 1
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape, cross_entropy_loss, acc, correct_label, Test_write, merged_summaries_):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    import matplotlib.pyplot as plt
    test_images,tl_dummy_gt = test_image_into_patches(data_folder)
    test_images = np.asarray(test_images)
    tl_dummy_gt = np.asarray(tl_dummy_gt)
    print("Test image batch shapes", test_images.shape[0])


    ###################### first solution##########

        ################solution 2
    batch_sz = 1
    cf_test_pred_list = []
    cf_gt_list = []
    name = 0
    acc_list_test = 0
    Total_metric = 0
    test_loss = 0
    counter = 0
    test_summary_list = []
    for image_file in range(0,test_images.shape[0],56):

        counter += 1
        #x_batch_arr = test_images[image_file].reshape(62, 32, 3)
        t_image = test_images[image_file:image_file + 56].reshape(-1, 32, 32, 3)
        gt_img_t = tl_dummy_gt[image_file:image_file + 56].reshape(-1, 32, 32)
        im_softmax = sess.run([tf.argmax(tf.nn.softmax(logits),axis=1)], feed_dict={keep_prob: 1.0, image_pl: t_image})

        loss_test_batch, acc_test_batch, test_batch_summary = sess.run([cross_entropy_loss, acc, merged_summaries_],feed_dict={image_pl: t_image,correct_label: gt_img_t})
        print("Counter: {} and Test Batch Accuracy: {}".format(counter,acc_test_batch))
        acc_list_test += acc_test_batch
        test_loss += loss_test_batch
        Test_write.add_summary(test_batch_summary, image_file)
        # conf_matrix
        im_softmax_conf = np.reshape(im_softmax,[-1])
        gt_img_conf = np.reshape(gt_img_t,[-1])
        #Cm_batch = confusion_matrix(gt_img_conf, im_softmax)
        #print("Confusion_matrix:",Cm_batch)
        #recall = np.diag(Cm_batch) / np.sum(Cm_batch, axis=1)                                   # sklearn precision
        #precision = np.diag(Cm_batch) / np.sum(Cm_batch, axis=0)                                # sklearn recall

        metric = precision_recall_fscore_support(gt_img_conf, im_softmax_conf, average='micro')         # precision recall
        print("Batch precision_recall_fscore_support : {}".format(metric))

        # softmax_list.append(im_softmax)
        #im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        #segmentation = (im_softmax > 0.05).reshape(image_shape[0], image_shape[1], 1)
        segmentation = np.reshape(im_softmax,(-1,image_shape[0], image_shape[1]))
        '''for i in range(0,image_shape[0]):
            for j in range(0,image_shape[1]):
                if segmentation[i][j]==0:
                    segmentation[i][j] = 100
                if segmentation[i][j]==1:
                    segmentation[i][j] = 127
                if segmentation[i][j] == 2:
                    segmentation[i][j] =  200'''
        #mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask_color = color_image(segmentation)
        #mask_color_rgb = scipy.misc.toimage(mask_color,mode='RGB')
        mask_color_rgb = mask_color
        ##############################################################  mask_ = scipy.misc.toimage(segmentation)
        ##############################################################  street_im = mask_
        #plt.imshow(mask_)
        #plt.show()
        ###########street_im = scipy.misc.toimage(x_batch_arr)
        #street_im = np.zeros((224,224))
        ##########street_im = scipy.misc.toimage(street_im)
        ##########street_im.paste(mask_, box=None, mask=mask_)

        ############################################################### softmax_list.append(street_im)
        #print("softmax list", np.shape(softmax_list))
        name = name + 1
        ############################################################### name_ = 'test'+str(name)+'.png'
        name_rgb = 'test_rgb' + str(name) + '.png'
        yield 'name_', 'np.array(street_im)', 'segmentation', name_rgb ,mask_color_rgb, im_softmax_conf,gt_img_conf
        #print("Check 1:")
    vg_test_accuracy = np.mean(test_summary_list)
    test_acc_1234 = acc_list_test / counter
    test_ls_1234 = test_loss / counter
    #Test_Acc_summary = tf.summary.scalar('Test_Accuracy', test_acc_1234)
    #sess.run([Test_Acc_summary],)
    #Test_write.add_summary(Test_Acc_summary)
    #Test_write.add_summary(test_acc_1234)
    #Test_write.add_summary(test_ls_1234)

    print("Unseen : Test Acc = {}.....Average(Total Acc/no.Image) = {} and ........... Loss = {}".format(acc_test_batch, test_acc_1234, test_ls_1234 ))
    #print("Precision : {} and Prec_op: {} ".format(prec, prec_op))
###########################################



def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, cross_entropy_loss, acc,  correct_label, Test_write, merged_summaries_):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'testing'), image_shape, cross_entropy_loss, acc,  correct_label, Test_write, merged_summaries_)

    Pred_gray = []
    result_arr = []
    cf_test_list = []
    cf_gt_list = []
    result = Image.new("RGB", (1366, 768))
    for name, image, segm, name_rgb, mask_color_rgb, cf_test_pred, cf_test_gt in image_outputs:
        cf_test_list.append(cf_test_pred)
        cf_gt_list.append(cf_test_gt)
        #scipy.misc.imsave(os.path.join(output_dir, name), image)
        ######################################################scipy.misc.imsave(os.path.join(output_dir, name_rgb), mask_color_rgb)         # save each patch in directory
        for nu_Test_patches in range(0,len(mask_color_rgb)):
            img_name = name_rgb+'_patch'+str(nu_Test_patches)+ '.png'
            scipy.misc.imsave(os.path.join(output_dir, img_name), mask_color_rgb[nu_Test_patches])         # save each patch in directory
            # plt.imshow(mask_color_rgb[nu_Test_patches])
            # plt.show()
            result_arr.append(mask_color_rgb[nu_Test_patches])
        #print("check 3:")
        #Pred_gray.append(segm)

    count = 0
    for h in range(0, 768, 32):
        for w in range(0, 1343, 32):
            img=result_arr[count]
            img_patch_ = scipy.misc.toimage(img)
            result.paste(img_patch_, (w, h, w + 32, h + 32))
            #plt.imshow(result)
            #plt.show()
            count += 1
            #if count == 1008:
                #break

    scipy.misc.imsave(os.path.join(output_dir, "ReconstructedImage_from_Patcimg_patch.png"), result)
    '''with open("Pred_Test_Gray.txt","wb") as fp:
        pickle.dump(Pred_gray,fp)'''

    print("Helper done:")
    return cf_test_list, cf_gt_list

