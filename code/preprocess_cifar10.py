import os
import sys
from random import shuffle
import cPickle as pickle


def process(dir):
    image_test = []
    image_unlabel = []
    image_label = []

    image_db = []
    image_map = {}
    for fname in os.listdir(dir):
        with open(os.path.join(dir, fname), 'rb') as f:
            print("process %s" % fname)
            images = pickle.load(f)
            print("images_num: %d" % (len(images['data'])))
            count = 0
            for data, label, file_name in zip(images['data'], images['labels'], images['filenames']):
                if label in image_map:
                    image_map[label].append((data, label, file_name))
                else:
                    ls = [(data, label, file_name)]
                    image_map[label] = ls
                count += 1
            print("count: %d" % count)

    for key, value in image_map.items():
        print("%s : %d" % (key, len(value)))
    # sample image_test image
    for key in image_map.keys():
        print("process category %s" % key)
        images = image_map[key]
        shuffle(images)
        # image_test.append(images[:100])
        for im in images[:100]:
            label_vector = []
            label = im[1]
            for i in range(10):
                if i == label:
                    label_vector.append(1)
                else:
                    label_vector.append(0)

            im = im + (label_vector,)
            image_test.append(im)

        images = images[100:]
        shuffle(images)

        for im in images[:500]:
            label_vector = []
            label = im[1]
            for i in range(10):
                if i == label:
                    label_vector.append(1)
                else:
                    label_vector.append(0)

            im = im + (label_vector,)
            image_unlabel.append(im)
            image_label.append(im)
            image_db.append(im)


        for im in images[500:]:
            im = im + ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0],)
            image_unlabel.append(im)

    shuffle(image_label)
    shuffle(image_unlabel)
    shuffle(image_test)

    # shuffle(image_db)
    print("image_unlabel:%d" % len(image_unlabel))
    print("image_test:%d" % len(image_test))
    print("image_label:%d" % len(image_label))

    print("start to write to disk")
    with open("cifar10_unlabel.pkl", 'wb') as f:
        dict = {"cifar10_unlabel": image_unlabel}
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

    with open("cifar10_test.pkl", 'wb') as f:
        dict = {"cifar10_test": image_test}
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

    with open("cifar10_label.pkl", 'wb') as f:
        dict = {"cifar10_label": image_unlabel}
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

    print("done!")


if __name__ == "__main__":
    process("/home/ttf/dataset/cifar-10-batches-py")
