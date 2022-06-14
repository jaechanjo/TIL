import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms as trn
import importlib
from scipy.optimize import minimize, LinearConstraint
from tqdm import tqdm

importlib.reload(nn)
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import minimize, LinearConstraint
import scipy as sp


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    Save tensor to cv2 format
         :param input_tensor: tensor to save
         :param filename: saved file name
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # Make a copy
    input_tensor = input_tensor.clone().detach()
    # To cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # Denormalization
    # input_tensor = unnormalize(input_tensor)
    # Remove batch dimension
    input_tensor = input_tensor.squeeze()
    # Convert from [0,1] to [0,255], then from CHW to HWC, and finally to cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB to BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # resnet network 앞단
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.weight_p = self.conv1.weight.data.numpy()
        self.relu = nn.ReLU(inplace=True)
        self.maxPooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxPooling(out)

        return out

    def maxpooling_inverse(self, features, kernel_size=3, stride=2, padding=1):
        # feature : (1, 64, 56, 56)
        unpooling = np.zeros((1, 64, 112, 112))

        for i, feat in enumerate(features[0]):
            # print(f'i : {i}, feat.shape : {feat.shape}')
            fm = np.zeros((114, 114))  # 나중에 padding 제거할 것. 지금은 padding 적용 상태

            for r in range(feat.shape[0]):
                for c in range(feat.shape[1]):
                    fm[(2*r)+1][(2*c)+1] = feat[r][c]

            fm = np.delete(fm, (fm.shape[0] - 1, 0), axis=0)
            fm = np.delete(fm, (fm.shape[1] - 1, 0), axis=1)

            unpooling[0][i] = fm

        # 이걸로 unpooling이 제대로 되었는지 확인 가능 --> 0이 아닌 요소의 갯수가 같다.
        # 못 믿겠으면.. 다시 유사도 검사해봐.. 나는 1.0 나왔어용..
        print("len(unpooling[np.nonzero(unpooling)]) :", len(unpooling[np.nonzero(unpooling)]))
        print("len(features[np.nonzero(features)]) :", len(features[np.nonzero(features)]))

        # 유사도 검사 flow는 아래의 흐름과 같이 진행된다.
        # 해당 함수의 unpooling을 self.conv1(torch.Tensor(unpooling))을 해서 결과값을 얻고 다시 cpu()로 변환
        # 그 다음에 파라미터로 넘어온 features와 유사도를 검사함.

        return unpooling

    def conv1_inverse(self, un_maxpooling_features):
        print("conv1_inverse")
        print("features.shape :", un_maxpooling_features.shape)  # (1, 64, 112, 112)
        print("self.weight_p.shape :", self.weight_p.shape)  # #(64, 3, 7, 7)


        # 나중에 padding 제거해야함
        # [0, 1] channel은 모두 다 하이퍼 파라미터로 고정하고 2번 째의 channel만 변수를 설정한다.
        c_1 = np.zeros((224, 224)) + 0.2
        c_2 = np.zeros((224, 224)) + 0.2
        c_3 = np.zeros((224, 224)) + 0.2

        c_1 = np.expand_dims(np.pad(c_1, pad_width=3, mode='constant', constant_values=0), axis=0)
        c_2 = np.expand_dims(np.pad(c_2, pad_width=3, mode='constant', constant_values=0), axis=0)
        c_3 = np.expand_dims(np.pad(c_3, pad_width=3, mode='constant', constant_values=0), axis=0)

        for r_i in range(c_3.shape[1]):
            for c_i in range(c_3.shape[2]):
                if r_i % 2 == 0 and r_i > 5 and r_i < 227 and c_i % 2 == 0 and c_i > 5 and c_i < 227:
                    c_3[0][r_i][c_i] = np.NaN

        # conv_input_with_padding : (3, 230, 230)
        conv_input_with_padding = np.concatenate((c_1, c_2, c_3), axis=0)
        weights = self.weight_p

        print("="*99)
        print("BEFORE")
        print("conv_input_with_padding.shape :", conv_input_with_padding.shape)
        print("conv_input_with_padding NAN COUNT :", len(conv_input_with_padding[np.isnan(conv_input_with_padding)]))
        print("=" * 99)

        for feature_map, weight in tqdm(zip(un_maxpooling_features[0], weights)):
            # feature_map : (112, 112)
            # weight는 (3, 7, 7)
            # print("featuremap.shape :", feature_map.shape)
            # print("weight.shape :", weight.shape)
            # print("conv_input_with_padding :", conv_input_with_padding.shape)

            for rn in range(feature_map.shape[0]):
                for cn in range(feature_map.shape[1]):
                    partial = conv_input_with_padding[0:3, 2*rn:(2*rn)+7, 2*cn:(2*cn)+7].copy()

                    partial_not_NAN = partial[~np.isnan(partial)]
                    weight_not_NAN = weight[~np.isnan(partial)]

                    dot = np.sum(partial_not_NAN * weight_not_NAN)

                    partial[2][6][6] = (feature_map[rn][cn] - dot) / weight[2][6][6]
                    conv_input_with_padding[0:3, 2 * rn:(2 * rn) + 7, 2 * cn:(2 * cn) + 7] = partial

        print("="*99)
        print("MEDIUM")
        print("conv_input_with_padding.shape :", conv_input_with_padding.shape)
        print("conv_input_with_padding NAN COUNT :", len(conv_input_with_padding[np.isnan(conv_input_with_padding)]))
        print("=" * 99)

        # conv_input_with_padding = np.delete(conv_input_with_padding,
        #                       (conv_input_with_padding.shape[1] - 1, conv_input_with_padding.shape[1] - 2, conv_input_with_padding.shape[1] - 3, 0, 1, 2), axis=1)
        # conv_input_with_padding = np.delete(conv_input_with_padding,
        #                       (conv_input_with_padding.shape[2] - 1, conv_input_with_padding.shape[2] - 2, conv_input_with_padding.shape[2] - 3, 0, 1, 2), axis=2)
        print("="*99)
        print("AFTER")
        print("conv_input_with_padding.shape :", conv_input_with_padding.shape)
        print("conv_input_with_padding NAN COUNT :", len(conv_input_with_padding[np.isnan(conv_input_with_padding)]))
        print("=" * 99)

        np.save('/workspace/conv_input_with_padding', conv_input_with_padding)
        np.save('/workspace/un_maxpooling_features', un_maxpooling_features)

        try:
            np.save('/workspace/weights', weights)
        except:
            pass

        w = weights[0]

        result = []

        for rn in range(112):
            for cn in range(112):
                partial = conv_input_with_padding[0:3, (2*rn):(2*rn)+7, (2*cn):(2*cn)+7]
                mul = partial * w
                result.append(np.sum(mul))

        ffff = np.array(result).reshape(112, 112)

        # ffff = self.conv1(torch.Tensor(np.expand_dims(conv_input_with_padding, axis=0)).cuda())
        # ffff = ffff.cpu().detach().numpy()[0][0]
        ffff = np.array(ffff).reshape(1, 12544)

        gggg = un_maxpooling_features[0][0]
        gggg = np.array(gggg).reshape(1, 12544)

        print("fff :", ffff)
        print("fff NAN COUNT :", len(ffff[np.isnan(ffff)]))
        print("ggg :", gggg)

        ffff = np.squeeze(np.asarray(ffff))
        gggg = np.squeeze(np.asarray(gggg))



        from numpy.linalg import norm
        from numpy import dot

        def cos_sim(A, B):
            return dot(A, B) / (norm(A) * norm(B))

        print("COS SIM :", round(cos_sim(ffff, gggg), 7))
        print("distance :", 1 - round(cos_sim(ffff, gggg), 7))

    def feature_inversion(self):
        pass


def main():
    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    nn = Network()
    model = nn.cuda()

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    path = "/workspace/0000##--4ziAJuyZ0.mp4##000023.jpg"
    image = Image.open(path)
    new_image = torch.rand(image.size)
    # print("image size :", image.size)

    model.eval()
    # image = torch.from_numpy(np.expand_dims(transform(image), axis=0))
    #
    # feat = model(image)
    # feat = feat.cpu().detach().numpy()

    input = np.load('/workspace/dontPLZPLZPZLZLZLZ/conv_input_with_padding.npy')
    weights = np.load('/workspace/dontPLZPLZPZLZLZLZ/weights.npy')
    output = np.load('/workspace/dontPLZPLZPZLZLZLZ/un_maxpooling_features.npy')


    print("input :", input)
    # input = model.conv1(torch.Tensor(np.expand_dimsinput, axis=0).cuda())
    # w = weights[0]
    #

    # un_maxpooling_features = model.module.maxpooling_inverse(feat)
    # de_conv_features = model.module.conv1_inverse(un_maxpooling_features)

    # Dcon = ConvDeconv(image, un_maxpooling_features, nn.weight_p[0][0], 1, 1, 3, 3, 58, False)
    # arbit_0 = Dcon.deconv2d()  #R


if __name__ == '__main__':
    main()
