

from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import cv2
import os

# import multiprocessing.dummy as multiprocessing

#################################
eps = 1e-7

debug = True


def border0(im, bor):
    bor += 1  # make sure bor > 0
    im[:, :bor, :] = 0
    im[:, -bor:, :] = 0
    im[:bor, :, :] = 0
    im[-bor:, :, :] = 0
    return im


#######################################################
class DataProvider(Dataset):
    """
    Class for the data provider
    """

    def __init__(self, shinyfolder, shininess, \
                 isrgb=True, \
                 ismergewithdiffuse=False, \
                 mergelow=0.0, \
                 mergehigh=1.0, \
                 dropratio=0.0, \
                 mode='train', \
                 datadebug=False):
        """
        split: 'train', 'train_val' or 'val'
        """
        self.mode = mode
        self.datadebug = datadebug
        
        self.mergelow = mergelow
        self.mergehigh = mergehigh
        self.mergerange = mergehigh - mergelow
        
        self.dropratio = dropratio
        
        self.ismerge = ismergewithdiffuse
        if self.ismerge:
            self.diffusefolder = '%s-diffuse' % shinyfolder
        
        self.isrgb = isrgb
        if isrgb:
            self.loader = cv2.IMREAD_COLOR
        else:
            self.loader = cv2.IMREAD_GRAYSCALE
        
        ##########################
        self.shinyfolder = shinyfolder
        self.shininess = shininess
        
        # different models in md5 name
        # assume md5 is kind of random
        subfolders = glob.glob('%s/%d/*' % (shinyfolder, shininess));
        subfolders = sorted(subfolders)
        
        modelnum = len(subfolders)
        modelnum1 = modelnum * 70 // 100
        modelnum2 = modelnum * 85 // 100
        if mode == 'train':
            subfolders = subfolders[:modelnum1]
        elif mode == 'train_val':
            subfolders = subfolders[modelnum1:modelnum2]
        else:
            subfolders = subfolders[modelnum2:]
        
        ################################
        # self.masks = loadmask()
        
        ################################
        self.imfolders = []
        
        # image items
        for j, subfolder in enumerate(subfolders):
            pa = subfolder
            imfolders = glob.glob('%s/*' % pa)
            self.imfolders.extend(imfolders)
        
        self.imfolders = sorted(self.imfolders)
        self.imnum = len(self.imfolders)
        if mode == 'train_val' and self.imnum > 200:
            step = self.imnum / 200 + 1
            self.imfolders = self.imfolders[::step]
            self.imnum = len(self.imfolders)
        
        print(self.imfolders[0])
        print(self.imfolders[-1])
        print('imnum%d' % self.imnum)
        
        # self.poolpool = multiprocessing.Pool(8)
    
    def __len__(self):
        return self.imnum

    def __getitem__(self, idx):
        return self.prepare_instance(idx)
    
    def mask0(self, im, lightx, lighty, rad):
        h, w, _ = im.shape
        lightx = int((lightx + 1) * w / 2)
        lighty = int((-lighty + 1) * h / 2)
        rad = int(rad * w / 2)
        cv2.circle(im, (lightx, lighty), rad, (0, 0, 0), -1)
        return im
    
    def prepare_instance(self, idx):
        """
        Prepare a single instance
        """
        imshinyfolder = self.imfolders[idx]
        
        re = {}
        re['valid'] = True
        re['shinyfolder'] = imshinyfolder
        
        try:
            imgtname = glob.glob('%s/ori*.txt' % imshinyfolder);
            imgtfile = cv2.FileStorage(imgtname[0], flags=cv2.FileStorage_READ)
            imgt = imgtfile.getFirstTopLevelNode().mat()
            imgt = imgt / (np.max(imgt) + eps)
            imgt = np.power(imgt, 0.3);
            # it should be 3 channel?
            if not self.isrgb:
                imgt = cv2.cvtColor(imgt, cv2.COLOR_BGR2GRAY)
                imgt = np.expand_dims(imgt, axis=2)
        except:
            re['valid'] = False
            return re
        
        if self.ismerge:
            # merge with diffuse
            imdiffusefolder = imshinyfolder.replace(self.shinyfolder, self.diffusefolder)
            pos1 = imdiffusefolder.find('shine_')
            pos2 = imdiffusefolder.find('-rot_')
            strr = 'shine_0.0000'
            strr0 = imdiffusefolder[pos1:pos2]
            imdiffusefolder = imdiffusefolder.replace(strr0, strr)
            re['diffusefolder'] = imdiffusefolder
        
        # we will go along ourter border, but drop some images
        idxs = []
        
        lightxnum = 7
        lightynum = 7
        for x in range(lightxnum):
            for y in range(lightynum):
                borx = (x == 0) or (x == lightxnum - 1)
                bory = (y == 0) or (y == lightxnum - 1)
                bor = borx or bory
                if bor:
                    idx = x * lightynum + y
                    idxs.append(idx)
        
        # next, w will random select them
        for i in range(len(idxs)):
            if np.random.rand() < self.dropratio:
                idxs[i] = -1
        
        # read images
        limages_ims = []
        limages_lposes = []
        limages_maxvals = []
        
        idx = -1
        gtheight, gtwidth, _ = imgt.shape
        for x in range(lightxnum):
            for y in range(lightynum):
                borx = (x == 0) or (x == lightxnum - 1)
                bory = (y == 0) or (y == lightynum - 1)
                bor = borx or bory
                if not bor:
                    continue
                
                lx = 0.4 * (x - 3)
                ly = 0.4 * (y - 3)
                lxim = lx * np.ones(shape=(gtheight, gtwidth, 1), dtype=np.float32)
                lyim = ly * np.ones(shape=(gtheight, gtwidth, 1), dtype=np.float32)
                lightpos = np.concatenate((lxim, lyim), axis=2)
                limages_lposes.append(lightpos)
                
                idx += 1
                
                # assume it is a bad capture!
                if idxs[idx] == -1:
                    im = np.zeros((gtheight, gtwidth, 1), dtype=np.float32)
                    if self.isrgb:
                        im = np.tile(im, (1, 1, 3))

                    limages_ims.append(im)
                    limages_maxvals.append(0)
                    continue
                
                imshy = cv2.imread('%s/combine-light_%d_%d.png' % (imshinyfolder, x, y), self.loader)
                if imshy is None:
                    re['valid'] = False
                    return re
                
                # blur it
                imshy = cv2.blur(imshy, (5, 5)).astype(np.float32) / 255.0
                if not self.isrgb:
                    imshy = np.expand_dims(imshy, axis=2)
                
                if self.ismerge:
                    imdiff = cv2.imread('%s/combine-light_%d_%d.png' % (imdiffusefolder, x, y), self.loader)
                    if imdiff is None:
                        re['valid'] = False
                        return re
                
                    imdiff = cv2.blur(imdiff, (5, 5)).astype(np.float32) / 255.0
                    if not self.isrgb:
                        imdiff = np.expand_dims(imdiff, axis=2)
                    
                    # ratio = 0.7 + 0.2 * np.random.rand()
                    # ratio = 0.1 * np.random.rand()
                    ratio = self.mergelow + self.mergerange * np.random.rand()
                    # print ratio
                    im = (1 - ratio) * imshy + ratio * imdiff
                else:
                    im = imshy
                
                # noise!
                noise1 = 0.05 * np.random.randn(256, 256, 1)
                noise2 = 0.03 * np.random.randn(256, 256, 1)
                im = im + noise1 + noise2 * im
                im = np.clip(im, 0.0, 1.0)
                
                bor = np.random.randint(9)
                im = border0(im, bor)
                
                '''
                maskim = self.masks[idx]
                im = im * maskim
                '''
                
                limages_ims.append(im)
                limages_maxvals.append(np.max(im))
        
        maxval = np.max(limages_maxvals) + eps
        limages_ims = [ im / maxval for im in limages_ims]
        
        limages_lightims = [np.concatenate((im * 2 - 1, lxy), axis=2) for im, lxy in zip(limages_ims, limages_lposes)]
        
        re['imlight'] = np.concatenate(limages_lightims, axis=2)
        re['gt'] = imgt * 2 - 1
        
        if self.datadebug:
            cv2.imshow("imgt", imgt)
            cv2.waitKey(0)
            
            imall = np.concatenate(limages_ims, axis=1)
            cv2.imshow("merge", imall)
            cv2.waitKey(0)
        
        return re


def collate_fn(batch_list):
    for data in batch_list:
        if not data['valid']:
            '''
            print 'invalid folder: '
            print data['shinyfolder']
            '''
    
    collated = {}
    batch_list = [data for data in batch_list if data['valid']]
    
    # keys = batch_list[0].keys()
    keys = ['gt', 'imlight']
    for key in keys:
        val = [item[key] for item in batch_list]
        val = np.stack(val, axis=0)
        collated[key] = val

    return collated


def get_data_loaders(shinyfolder, shininess, isrgb, ismergewithdiffuse, mergelow, mergehigh, dropratio, mode, bs, numworkers):
    
    # print 'Building dataloaders'
    
    dataset_train = DataProvider(shinyfolder, shininess, isrgb, \
                                 ismergewithdiffuse, \
                                 mergelow=mergelow, \
                                 mergehigh=mergehigh, \
                                 dropratio=dropratio, \
                                 mode=mode, datadebug=False)
    
    shuffle = True
    if mode == 'train_val' or mode == 'test':
        shuffle = False
    
    train_loader = DataLoader(dataset_train, batch_size=bs, \
                              shuffle=shuffle, num_workers=numworkers, collate_fn=collate_fn)
    '''
    print 'train num', len(dataset_train)
    print 'train iter', len(train_loader)
    '''
    
    return train_loader


##############################################
if __name__ == '__main__':
    
    # masks = loadmask()
    # cloadrealdata(masks)
    
    shinyfolder = '/u6/a/wenzheng/remote3/datasets/shapenet/mnist-render-steady'
    shininess = 0
    
    if shininess > 0:
        ismerge = True
    else:
        ismerge = False
    
    train_loader = get_data_loaders(shinyfolder, shininess, isrgb=False, \
                                                     ismergewithdiffuse=ismerge, \
                                                     mergelow=0.0, mergehigh=0.0, \
                                                     dropratio=0.0, mode='train_val', \
                                                     bs=1, numworkers=0)
    
    ##############################################
    for i, data in enumerate(train_loader):
        for key in data.keys():
            print('{} {} {}'.format(i, key, data[key].shape))

