from sklearn.model_selection import train_test_split
import os, cv2, numpy as np

class io_handler():
    def __init__(self, batchsize, data_path, class0_dir_name, class1_dir_name):
        assert batchsize % 2 == 0, 'batchsize has to be divisible by 2'
        self.batchsize = batchsize
        self.data_path = data_path
        self.class0_dir_name = class0_dir_name
        self.class1_dir_name = class1_dir_name

    def get_train_val_names(self):
        names = os.listdir(os.path.join(self.data_path, self.class0_dir_name))
        train_names, val_names = train_test_split(names, test_size=0.05, random_state=1)
        return train_names, val_names

    def load_image_label_batch(self, iter, names):
        '''
        load a batch of images during training or validation
        :param iter:    current training or validation iteration step
        :param names:   training images names or validation images names,
                        used to identify training or validation phase
        :return:        4d np array, images batch with shape [batchsize, height, width, channel]
        '''
        img_batch = []
        label_batch = []
        for i in range(int(self.batchsize/2)):
            img_a = cv2.imread(os.path.join(self.data_path,
                                          self.class0_dir_name,
                                          names[iter + i])) / 255.
            img_b = cv2.imread(os.path.join(self.data_path,
                                            self.class1_dir_name,
                                            names[iter + i])) / 255.
            img_batch.extend([img_a, img_b])
            label_batch.extend([[0],[1]])
        img_batch = np.array(img_batch)
        label_batch = np.array(label_batch)
        return img_batch, label_batch

if __name__=='__main__':
    test = io_handler(8,
                      '/media/lewisluk/新加卷1/dataset/色彩浓艳度分类',
                      'A',
                      'B')
    train_names, val_names = test.get_train_val_names()
    img_batch, label_batch = test.load_image_label_batch(0, train_names)