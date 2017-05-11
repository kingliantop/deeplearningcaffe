#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
python createImageSets.py args [file path containing image, eg:JPEGImages]
'''

import os,sys

def createImageSets(argv):
    root = os.path.dirname(os.path.abspath(argv[1]))
    if not os.path.exists(os.path.join(root, 'JPEGImages')):
        print "Please create JPEGImages."
        return
    if not os.path.exists(os.path.join(root, 'Annotations')):
        print "Please create Annotations."
        return
    if not os.path.exists(os.path.join(root, 'ImageSets/Main')):
        os.makedirs('ImageSets/Main')

    filenames = sorted([x[:-4] for x in os.listdir('Annotations') if not x.startswith('.') and x.endswith('.xml')])
    amount = len(filenames)
    trainval = int(amount*0.9)
    train = int(amount*0.9*0.9)

    print 'create trainval.txt'
    with open('ImageSets/Main/trainval.txt', 'w') as fid:
        for name in filenames[ : trainval]:
            fid.write('{}\n'.format(name))

    print 'create train.txt'
    with open('ImageSets/Main/train.txt', 'w') as fid:
        for name in filenames[ : train]:
            fid.write('{}\n'.format(name))

    print 'create val.txt'
    with open('ImageSets/Main/val.txt', 'w') as fid:
        for name in filenames[train : trainval]:
            fid.write('{}\n'.format(name))

    print 'create test.txt'
    with open('ImageSets/Main/test.txt', 'w') as fid:
        for name in filenames[trainval : ]:
            fid.write('{}\n'.format(name))

def main():
    import sys
    if len(sys.argv) != 2:
        print(__doc__)
        return
    createImageSets(sys.argv)

if __name__ == "__main__":
    main()
