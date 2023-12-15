import numpy as np
from scipy.stats import ortho_group
from PIL import Image
import matplotlib.pyplot as plt
import random,os,math
from Cipher import AES as aes


class RMT:

    def __init__(self,image_size=(32,32), block_size = 4,Shuffle=False):

        self.img_size = image_size

        self.block_size = block_size

        self.block_num = int((image_size[0]/block_size)*(image_size[1]/block_size))

        self.RMT_Matrixes = self.Create_RMT()

        self.shuffle = Shuffle

    def normalize(self, img):

        img1 = img.astype(np.float32)

        return img1

    def Create_RMT(self):

        mats=[]

        for i in range(self.block_num):

            mats.append(ortho_group.rvs(dim=self.block_size))


        return mats

    def Add_Noise(self,blocks,N):

        block = []

        for i in blocks:

            noise = np.divide(np.random.randint(1, N, self.block_size*self.block_size).reshape(i.shape),255)

            block.append(i+noise)

        return blocks


    def M2block(self,array,noise_level=100, noise=False):

        h = array.shape[0]
        r = array.shape[1]

        blocks=[]

        hight = [i*self.block_size for i in range(int(h/self.block_size))]

        width = [i*self.block_size for i in range(int(r/self.block_size))]

        for i in hight:

            for j in width:

                blocks.append(array[i:i+self.block_size,j:j+self.block_size])

        if noise:

            return self.Add_Noise(blocks,noise_level)

        else:

            return blocks

    def block2M(self,block_list,seed=1):

        Row = []

        Column = []

        blocks=block_list

        if self.shuffle:

            random.Random(seed).shuffle(block_list)

        for i in range(self.block_num):

            if (i+1)%(self.img_size[1]/self.block_size)!=0:

                Row.append(blocks[i])

            else:

                Row.append(blocks[i])

                Column.append(np.hstack(Row))

                Row=[]

        return np.vstack(Column)

    def Encode(self,image,noise=True,noise_level=100,shuffling_seed=1):

        img = self.normalize(image)

        if len(img.shape) != 3:

            blocks = self.M2block(img,noise_level,noise)

            block_enc = [ np.matmul(blocks[i], self.RMT_Matrixes[i]) for i in range(len(blocks))]

            return self.block2M(block_enc)

        else:

            img2 = img.copy()

            for c in range(img.shape[2]):

                blocks = self.M2block(img[:,:,c])

                block_enc = [ np.matmul(blocks[i], self.RMT_Matrixes[i]) for i in range(len(blocks))]

                img2[:,:,c] = self.block2M(block_enc)

            return img2

    def RMT_M(self):

        return self.RMT_Matrixes


    def Estimate_block_list(self, Original, Encrypted):

        Original_blocks = []

        Encrypted_blocks = []

        Fin_O = []

        Fin_E = []

        if Original.shape[0] == Encrypted.shape[0]:

            for i in range(Original.shape[0]):

                Original_blocks.append(self.M2block(Original[i]))

                Encrypted_blocks.append(self.M2block(Encrypted[i]))

            for i in range(self.block_num):

                temp_o = []

                temp_e =[]

                for j in range(Original.shape[0]):

                    temp_o.append(Original_blocks[j][i])

                    temp_e.append(Encrypted_blocks[j][i])

                Fin_O.append(temp_o.copy())

                Fin_E.append(temp_e.copy())

        else:

            print("Clean dataset has different size with encrypted dataset")

        return Fin_O, Fin_E

    def block_list_recover(self,Fin_O,Fin_E):
    # For test
        img_num = len(Fin_O[0])

        block_o = []

        block_e = []

        temp_O = np.array(Fin_O)

        temp_E = np.array(Fin_E)

        for i in range(temp_O.shape[1]):

            temp_o2 = []

            temp_e2 = []

            for j in range(temp_O.shape[0]):

                temp_o2.append(temp_O[j,i,:,:])

                temp_e2.append(temp_E[j,i,:,:])

            img_o = self.block2M(temp_o2)

            img_e = self.block2M(temp_e2)

            block_o.append(img_o)

            block_e.append(img_e)

        return np.array(block_o), np.array(block_e)


    def Estimate_one_position(self,Original,Encrypted):

        (nrows, ncols) = Original[0].shape

        R = np.array([[]])

        X = Original[0]

        for j in range(1,len(Encrypted)):

            X = np.append(X,Original[j],0)

        det = np.linalg.det(np.dot(X.T,X))

        if det == 0:

            return []

        Xbar = np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)

        for i in range(ncols):

            y = np.array(Encrypted[0][:,[i]])

            for j in range(1,len(Encrypted)):

                y = np.append(y, Encrypted[j][:,[i]],0)

            r = np.dot(Xbar,y)

            if i==0:

                R = r

            else:

                R=np.append(R,r,1)

        if np.linalg.det(R)==0:

            return []

        else:

            return R

    def Estimate_one_channel(self, Original, Encrypted):

        Original_blocks, Encrypted_blocks = self.Estimate_block_list(Original,Encrypted)

        RMT_Encoder = []

        for i in range(self.block_num):

            RMT_Encoder.append(self.Estimate_one_position(Original_blocks[i],Encrypted_blocks[i]))

        return RMT_Encoder

    def Estimate(self, Original, Encrypted):

        if len(Original.shape) < 4:

            return self.Estimate_one_channel(self.normalize(Original),Encrypted)

        elif len(Original.shape) == 4:

            print("RGB image, estimate for every channel")

            channels=[]

            for i in range(self.img_size[2]):

                channels.append(self.Estimate_one_channel(self.normalize(Original)[:,:,:,i],Encrypted[:,:,:,i]))

            return channels

    def Recover_one_channel(self,Encrypted,RMT_encoders):

        Blocks = self.M2block(Encrypted)

        Blocks_recovered = []

        for i in range(len(Blocks)):

            if len(RMT_encoders[i])==0:

                Blocks_recovered.append(np.zeros(Blocks[1].shape))

            else:

                Blocks_recovered.append(np.matmul(Blocks[i], np.linalg.inv(RMT_encoders[i])))

        return self.block2M(Blocks_recovered)

    def Recover(self,Encrypted,RMT_encoders):

        if len(Encrypted.shape) < 3:

            return self.Recover_one_channel(Encrypted,RMT_encoders)

        else:

            if len(RMT_encoders) == 3:

                img = np.zeros(self.img_size)

                for i in range(3):

                    img[:,:,i] = self.Recover_one_channel(Encrypted[:,:,i],RMT_encoders[i])

                return img

            else:

                img = np.zeros(shape = self.img_size)

                for i in range(3):

                    img[:,:,i] = self.Recover_one_channel(Encrypted[:,:,i],RMT_encoders)

                return img


class AES:

    def __init__(self, image_size=(32, 32), block_size=(4, 4), One_cipher=True, Shuffle=False):

        self.img_size = image_size

        self.block_size = block_size

        self.block_num = int((image_size[0] / block_size[0]) * (image_size[1] / block_size[1]))

        print(image_size)
        print(block_size)

        print('block number = ' + str(self.block_num))

        block_bytes = block_size[0] * block_size[1]

        self.scale = [1, 1]

        self.shuffle = Shuffle

        self.one_cipher = One_cipher

        if block_bytes < 16:
            # scale it
            if self.block_size[0] < self.block_size[1]:

                less = 0

            else:

                less = 1

            if block_bytes == 2:

                self.scale[less] = 4

                self.scale[(less + 1) % 2] = 2

            elif block_bytes == 4:

                if self.block_size == (2, 2):

                    self.scale = [2, 2]

                else:

                    self.scale[less] = 4

            elif block_bytes == 8:

                self.scale[less] = 2

            self.block_size = (4, 4)

        self.update_params()  # due to updated parameters

        if not One_cipher:

            self.ciphers = [aes.new(os.urandom(16), aes.MODE_ECB) for i in range(self.block_num)]

        else:

            self.ciphers = aes.new(os.urandom(16), aes.MODE_ECB)

    def update_params(self):

        image_size = self.img_size

        block_size = self.block_size

        if image_size[0] % block_size[0] != 0:

            p0 = block_size[0] - image_size[0] % block_size[0]

            self.p0_left = int(p0 / 2)

            self.p0_right = p0 - self.p0_left

        else:

            p0 = 0

            self.p0_left = 0

            self.p0_right = 0

        if image_size[1] % block_size[1] != 0:

            p1 = block_size[1] - image_size[1] % block_size[1]

            self.p1_left = int(p1 / 2)

            self.p1_right = p1 - self.p1_left

        else:

            p1 = 0

            self.p1_left = 0

            self.p1_right = 0

        if p0 == 0 and p1 == 0:

            self.pad = False

        else:
            self.pad = True

        self.block_num = math.ceil(image_size[0] / block_size[0]) * math.ceil(image_size[1] / block_size[1])

    def padded_size(self):

        return (self.img_size[0] + self.p0_left + self.p0_right, self.img_size[1] + self.p1_left + self.p1_right)

    def padding(self, img):

        if len(self.img_size) == 3:

            assert ((img.shape[0], img.shape[1], img.shape[2]) == self.img_size)

        else:

            assert ((img.shape[0], img.shape[1]) == self.img_size)

        if not self.pad:
            return img.copy()

        if len(self.img_size) == 3:

            img1 = np.zeros((img.shape[0] + self.p0_left + self.p0_right, img.shape[1] + self.p1_left + self.p1_right,
                             img.shape[2]))

            for c in range(img.shape[2]):
                img1[:, :, c] = np.pad(img[:, :, c], ((self.p0_left, self.p0_right), (self.p1_left, self.p1_right)))

        else:

            img1 = np.zeros((img.shape[0] + self.p0_left + self.p0_right, img.shape[1] + self.p1_left + self.p1_right))

            img1[:, :] = np.pad(img[:, :], ((self.p0_left, self.p0_right), (self.p1_left, self.p1_right)))

        return img1

    def scaleup(self, img):

        ''' img: w*h, each pixel duplicate to the corresponding 4x4 block'''

        assert (self.scale != [1, 1] and img.shape[0:2] == self.img_size)

        img1 = np.ones(img.shape)

        if len(img.shape) == 3:

            for c in range(img.shape[2]):

                for i in range(img.shape[0]):

                    for j in range(img.shape[1]):
                        img1[i * self.scale[0]:(i + 1) * self.scale[0], j * self.scale[1]:(j + 1) * self.scale[1], c] *= \
                        img[i, j, c]

        else:

            for i in range(img.shape[0]):

                for j in range(img.shape[1]):
                    img1[i * self.scale[0]:(i + 1) * self.scale[0], j * self.scale[1]:(j + 1) * self.scale[1]] *= img[
                        i, j]

        return img1.astype(np.byte)

    def M2vector(self, block):

        vec = block.reshape((1, block.shape[0] * block.shape[1]))

        return vec

    def vector2M(self, vector):

        M = vector.reshape((self.block_size[0], self.block_size[1]))

        return M

    def M2block(self, array):

        h, r = array.shape[0:2]

        blocks = []

        hight = [i * self.block_size[0] for i in range(int(h / self.block_size[0]))]

        width = [i * self.block_size[1] for i in range(int(r / self.block_size[1]))]

        for i in hight:

            for j in width:
                blocks.append(array[i:i + self.block_size[0], j:j + self.block_size[1]])

        return blocks

    def block2M(self, block_list, seed=1):

        Row = []

        Column = []

        blocks = block_list

        if self.shuffle:
            random.Random(seed).shuffle(block_list)

        for i in range(self.block_num):

            if (i + 1) % (self.img_size[1] / self.block_size[1]) != 0:

                Row.append(blocks[i])


            else:

                Row.append(blocks[i])

                Column.append(np.hstack(Row))

                Row = []

        return np.vstack(Column)

    def block_enc(self, block, cipher):

        block1 = self.M2vector(block)

        assert (block1.shape[1] % 16 == 0)

        for i in range(int(block1.shape[1] / 16)):
            bytes = cipher.encrypt(block1[:, i * 16:(i + 1) * 16].tobytes())

            block1[:, i * 16:(i + 1) * 16] = np.frombuffer(bytes, dtype=np.byte)

        return self.vector2M(block1)

    def Encode(self, img, noise = False, noise_level = 1):

        if (self.scale != [1, 1]):

            img1 = self.scaleup(img)

        else:

            img1 = img

        img2 = self.padding(img1).astype(np.byte)

        if len(img2.shape) == 3:

            for c in range(img2.shape[2]):  # channels

                blocks = self.M2block(img2[:, :, c])

                if not self.one_cipher:

                    blocks_e = [self.block_enc(b, e) for b, e in zip(blocks, self.ciphers)]

                else:

                    blocks_e = [self.block_enc(b, self.ciphers) for b in blocks]

                img2[:, :, c] = self.block2M(blocks_e)

        else:

            blocks = self.M2block(img2[:, :])

            if not self.one_cipher:

                blocks_e = [self.block_enc(b, e) for b, e in zip(blocks, self.ciphers)]

            else:

                blocks_e = [self.block_enc(b, self.ciphers) for b in blocks]



            img2[:, :] = self.block2M(blocks_e)

        return img2

    def AES_Ciphers(self):

        return self.ciphers


class test():

    def __init__(self,encoder):

        self.encoder=encoder

        self.test_Vectorize()

        self.test_blocking()

    def test_Vectorize(self):

        #test vector to matrix and matrix to vector

        scores=[]

        img = np.array(list(range(self.encoder.block_size[0]*self.encoder.block_size[1])), dtype=np.byte).reshape(self.encoder.block_size[0], self.encoder.block_size[1])

        vector = self.encoder.M2vector(img)

        img0 = self.encoder.vector2M(vector)

        if np.linalg.norm(img-img0) < 0.00000001:

            self.vectorize = True

        else:

            self.vectorize = False

            self.vectorize_error = [img0,img]

    def test_blocking(self):

        #test M2block and block2M
        img_pix = 1

        for i in self.encoder.img_size:

            img_pix*=i

        img = np.array(list(range(img_pix)), dtype=np.byte).reshape(self.encoder.img_size)

        blocks = self.encoder.M2block(img)

        img0 = self.encoder.block2M(blocks)

        if np.linalg.norm(img-img0) < 0.00000001:

            self.blocking = True

        else:

            self.blocking = False

            self.blocking_error = [img0,img]

class test_RMT(test):

    def __init__(self,encoder):

        self.encoder = encoder

        self.test_blocking()

        self.test_Recover()

        self.test_block_list()

        self.test_Estimate_times()

    def test_block_list(self):

        img_pix = 1

        for i in self.encoder.img_size[0:2]:

            img_pix*=i

        num_imgs = 5

        imgs = self.encoder.normalize(np.array(list(range(img_pix*num_imgs))).reshape((num_imgs,self.encoder.img_size[0],self.encoder.img_size[1])))

        a,b = self.encoder.Estimate_block_list(imgs,imgs)

        a1,b1 = self.encoder.block_list_recover(a,b)

        if np.linalg.norm(a1-imgs) < 0.000001:

            self.test_block_list_s = True

        else:

            self.test_block_list_s = False

    def test_Recover(self):

        img_pix = 1

        for i in self.encoder.img_size:

            img_pix*=i

        img = []

        a=34

        for i in range(img_pix):

            img.append(a)

            if a<255:

                a+=1

            else:

                a = 0

        img = np.array(img)

        img = img.reshape(self.encoder.img_size)

        a = img.shape[0:2]

        if len(self.encoder.img_size)==3:

            m = [np.full(img.shape[0:2],0.4914),np.full(img.shape[0:2],0.4822),np.full(img.shape[0:2],0.4465)]

            std = [np.full(img.shape[0:2],0.2023),np.full(img.shape[0:2],0.1994),np.full(img.shape[0:2],0.2010)]

            for i in range(img.shape[2]):

                img[:,:,i] = np.divide(np.multiply(img[:,:,i],m[i]),std[i])

        enc_img = self.encoder.Encode(img)

        rec_img = self.encoder.Recover(enc_img,self.encoder.RMT_M())

        if np.linalg.norm(self.encoder.normalize(img)-rec_img) < 0.001:

            self.test_Recover = True

        else:

            self.test_Recover = False

            print(np.linalg.norm(self.encoder.normalize(img)-rec_img))

    def test_Estimate_times(self):

        scores = []

        for i in range(1,5,100):

            scores.append(self.test_Estimate_one_time(i))

        flag = 0

        i = 1

        while i < len(scores):

            if(scores[i] < scores[i - 1]):

                flag = 1

                i += 1

        if (not flag):

            self.test_Estimate = True

        else:

            self.test_Estimate = False



    def test_Estimate_one_time(self,num_pic):

        img_pix = 1

        for i in self.encoder.img_size:

            img_pix*=i

        img = []

        a=1

        if len(self.encoder.img_size) == 3:

            for i in range(img_pix*num_pic):

                img.append(a)

                if a<200:

                    a+=1

                else:

                    a = 0

            imgs = np.array(img)

            img = imgs.reshape((num_pic,self.encoder.img_size[0],self.encoder.img_size[1],self.encoder.img_size[2]))

            img2 = img

            m = [np.full(img.shape[1:3],0.4914),np.full(img.shape[1:3],0.4822),np.full(img.shape[1:3],0.4465)]

            std = [np.full(img.shape[1:3],0.2023),np.full(img.shape[1:3],0.1994),np.full(img.shape[1:3],0.2010)]

            for i in range(img.shape[0]):

                for j in range(img.shape[3]):

                    img[i,:,:,j] = np.divide(np.multiply(img[i,:,:,j],m[j]),std[j])

            enc_imgs = []

            for i in range(num_pic):

                enc_imgs.append(self.encoder.Encode(img[i,:,:,:]))

            enc_imgs = np.array(enc_imgs)

            Estimated_RMT = self.encoder.Estimate(img,enc_imgs)

            rec_imgs = []

            for i in range(num_pic):

                rec_imgs.append(self.encoder.Recover(enc_imgs[0,:,:,:],Estimated_RMT))

            rec_imgs = np.array(rec_imgs)


        else:

            for i in range(img_pix*num_pic):

                img.append(a)

                if a<255:

                    a+=1

                else:

                    a = 0

            imgs = np.array(img)

            imgs = imgs.reshape((1,imgs.shape[0],imgs.shape[1]))

            enc_imgs = self.encoder.Encode(imgs[0,:,:]).reshape((1,imgs.shape[1],imgs.shape[2]))

            Estimated_RMT = self.encoder.Estimate(imgs,enc_imgs)

            rec_imgs = self.encoder.Recover(enc_imgs[0,:,:],Estimated_RMT)

        return np.linalg.norm(self.encoder.normalize(img[0,:,:,:])-rec_imgs[0,:,:,:])




if __name__ == "__main__":

    Tester = test_AES(AES(image_size=(32,32,3), block_size=(4,4), One_cipher=True, Shuffle=False))

    print('Test Vectorize: '+('pass' if Tester.vectorize else 'failed')+'\n' +
          'Test Blocking: '+('pass' if Tester.blocking else 'failed')+'\n' +
          'Test Block encryption: '+('pass' if Tester.block_encryption else 'failed')+'\n' +
          'Test AES encryption: '+('pass' if Tester.Encoding else 'failed'))

    if not Tester.vectorize:

        print('Vectorize error: ')

        print(Tester.vectorize_error)

    if not Tester.blocking:

        print('Blocking error: ')

        print(Tester.blocking_error[0])

        print(Tester.blocking_error[1])

    if not Tester.block_encryption:

        print('Block encryption error: ')

        print(Tester.block_encryption_error[0])

    if not Tester.Encoding:

        print('AES encryption error: ')

        print(Tester.Encoding_error)


    Tester_RMT = test_RMT(RMT(image_size=(32,32,3),block_size=4,Shuffle=False))

    print('Test Recover: '+('pass' if Tester_RMT.test_Recover else 'failed')+'\n' +
          'Test Estimate: '+('pass' if Tester_RMT.test_Estimate else 'failed')+'\n'+
          'Test block_list_s: '+('pass' if Tester_RMT.test_block_list_s else 'failed')+'\n'+
          'Test blocking: '+('pass' if Tester_RMT.blocking else 'failed')+'\n')
