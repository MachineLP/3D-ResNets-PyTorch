# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
-------------------------------------------------
   Description :
   Author :       libing
   Date :         2020/11/14 15:40
-------------------------------------------------

"""
import torch
import torch.nn.functional as F
import torch
import onnx
import pickle
import tensorrt as trt
from onnx_tensorrt.tensorrt_engine import Engine
from src.conf.params_common import *
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.model_zoo import get_model
from src.third_party.action_3d_resnets.model import (generate_model)
from src.third_party.action_3d_resnets.opts import Ops
from torch import nn
from src.third_party.action_3d_resnets.mean import get_mean_std
from ..action_recg_handler.action_utils import get_spatial_transform
from ..action_recg_handler.action_utils import batch_preprocessing,preprocessing
import numpy as np



class ResNetModel:
    def __init__(self):
        opt = Ops()
        opt.n_input_channels = 3
        opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
        model = generate_model(opt)
        model.fc = nn.Linear(model.fc.in_features, 2)
        pretrain_path =resnet_model_url
        pretrain = torch.load(pretrain_path, map_location=lambda storage, loc: storage.cuda(0))
        model.load_state_dict(pretrain['state_dict'])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        model.eval()
        self.spatial_transform = get_spatial_transform(opt)
        self.model = model
        self.opt =opt

    def forward(self,clips,  batch_flag=True):
        # do some preprocessing steps
        try:
            if batch_flag:
                clip = batch_preprocessing(clips, self.spatial_transform)
            else:
                clip = preprocessing(clips, self.spatial_transform)
            # don't calculate grads
            with torch.no_grad():
                # apply model to input
                outputs = self.model(clip.cuda())
                # apply softmax and move from gpu to cpu
                outputs = F.softmax(outputs, dim=1).cpu()
                # get best class
                score, class_prediction = torch.max(outputs, 1)
                classes = ['basketball_shot','negtive']
                return score, [classes[i] for i in class_prediction]
        except Exception as e :
            return [0] ,['']

class Res3DNetModel():
    def __init__(self):
        opt = Ops()
        opt.n_input_channels = 3
        opt.sample_size=(112,56)
        opt.inference_crop = 'resize'
        opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
        model = generate_model(opt)
        model.fc = nn.Linear(model.fc.in_features, 13)
        pretrain_path ='/DATA/disk1/libing/online/3D-ResNets-PyTorch/data/results_112_56_pos/save_30.pth'
        pretrain = torch.load(pretrain_path, map_location=lambda storage, loc: storage.cuda(0))
        model.load_state_dict(pretrain['state_dict'])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        # model = nn.DataParallel(model, device_ids=[3]).cuda()
        model.eval().cuda()
        self.spatial_transform = get_spatial_transform(opt)
        self.model = model
        self.opt =opt

    def forward(self,clips,  batch_flag=True):
        # do some preprocessing steps
        try:
            if batch_flag:
                clip = batch_preprocessing(clips, self.spatial_transform)
            else:
                clip = preprocessing(clips, self.spatial_transform)
            # don't calculate grads
            with torch.no_grad():
                # apply model to input
                outputs = self.model(clip.cuda())
                # apply softmax and move from gpu to cpu
                outputs = F.softmax(outputs, dim=1).cpu()
                # get best class
                score, class_prediction = torch.max(outputs, 1)
                classes = ["ballet_foot_stretch", "negtive", "ballet_openfly", "ballet_waist", "basketball_bounce", "basketball_dribble", "basketball_shoot", "soccer_carry", "soccer_highfive", "soccer_shoot", "backhand_hit", "forehand_hit", "serve"]
                return score.item(), classes[class_prediction]
        except Exception as e :
            return 0 ,''


class ModelTensorRT:
    def __init__(self,trt_logger,resnet_model_url,opt):
        self.trt_logger = trt_logger
        self.resnet_model_url = resnet_model_url
        resnet_dir = os.path.dirname(resnet_model_url)
        file_name = resnet_model_url.split('/')[-1].split('.')[0]
        self.resnet_model_trt_engine_url=os.path.join(resnet_dir,file_name+'.onnx')
        self.resnet_model_onnx_url =os.path.join(resnet_dir,file_name+'.trt')
        self.opt =opt
    def generate_engine(self):
        if os.path.exists(self.resnet_model_trt_engine_url):
            runtime = trt.Runtime(self.trt_logger)
            with open(self.resnet_model_trt_engine_url, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
        elif os.path.exists(self.resnet_model_onnx_url):
            engine = self.generate_from_onnx()
        elif os.path.exists(self.resnet_model_url):
            engine = self.generate_from_pth(self.opt)
        else:
            raise ValueError('this path {} is not exists'.format(self.resnet_model_url))

        return engine

    def generate_from_onnx(self):
        model = onnx.load(self.resnet_model_onnx_url)
        model_str = model.SerializeToString()
        builder = trt.Builder(self.trt_logger)
        builder.max_batch_size = self.opt.batch_size
        if builder.platform_has_fast_fp16:
            builder.fp16_mode = True
        # if builder.platform_has_fast_int8:
        #     builder.int8_mode = True
        networks = builder.create_network(flags=1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(networks, self.trt_logger)
        if not parser.parse(model_str):
            raise ValueError('parse onnx model fail')
        for layer_idx in range(networks.num_layers):
            layer = networks.get_layer(layer_idx)
            if layer.precision == trt.DataType.FLOAT:
                layer.precision = trt.DataType.HALF
                print('conver {} to HALF'.format(layer.name))
            elif layer.precision == trt.DataType.INT32:
                layer.precision = trt.DataType.INT32
            else:
                layer.precision = layer.precision
        inputs = networks.get_input(0)
        inputs.dtype = trt.DataType.HALF
        # outputs = networks.get_output(0)
        # outputs.dtype = trt.DataType.HALF
        engine = builder.build_cuda_engine(networks)
        with open(self.resnet_model_trt_engine_url, 'wb') as f:
            f.write(engine.serialize())
        return engine

    def generate_from_pth(self,opt):
        model = generate_model(opt)
        model.fc = nn.Linear(model.fc.in_features, opt.n_classes)
        pretrain = torch.load(self.resnet_model_url, map_location=lambda storage, loc: storage.cuda(0))
        model.load_state_dict(pretrain['state_dict'])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        model.eval()
        a = torch.autograd.Variable(
            torch.randn((opt.batch_size, self.opt.n_input_channels, self.opt.sample_duration, opt.sample_size[0], opt.sample_size[1])))
        a = a.cuda()
        model(a)
        torch.onnx.export(model, a, self.resnet_model_onnx_url)
        del model, a
        torch.cuda.empty_cache()
        engine = self.generate_from_onnx()
        return engine






#无跟踪模型trt
class ResNetModelTensorRT:

    def __init__(self):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        opt = Ops()
        opt.n_classes=2
        opt.batch_size=20
        opt.sample_size = (112, 112)
        opt.n_input_channels = 3
        opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
        self.spatial_transform = get_spatial_transform(opt)
        engine = ModelTensorRT(self.trt_logger, resnet_model_url, opt).generate_engine()
        self.engine = Engine(engine)
        self.opt = opt
        self.inputs_shape = self.engine.inputs[0].shape
        print('engine input shape', self.inputs_shape)

    def less_predict(self, inputs):
        print('inputs batch less than engine inputs')
        inp_batch = inputs.shape[0]
        inputs = np.vstack([inputs, np.zeros((self.inputs_shape[0] - inp_batch, *self.inputs_shape[1:]),
                                             dtype=np.float16)])
        outputs = self.engine.run([inputs])[0]
        outputs = outputs[:inp_batch, :]
        return outputs

    def forward(self, clips, batch_flag=True):
        try:
            if batch_flag:
                clip = batch_preprocessing(clips, self.spatial_transform)
            else:
                clip = preprocessing(clips, self.spatial_transform)
            inputs = clip.cpu().numpy()
            inputs = np.array(inputs, copy=True, dtype=np.float16)
            inp_batch = inputs.shape[0]
            if inp_batch < self.inputs_shape[0]:
                outputs = self.less_predict(inputs)
            elif inp_batch == self.inputs_shape[0]:
                print('batch size equal ')
                outputs = self.engine.run([inputs])[0]
            else:
                print('inputs batch greater than engine inputs')
                outputs = []
                for i in range(0, inp_batch, self.inputs_shape[0]):
                    if i != 0:
                        inp = inputs[li:i, :]
                        if inp.shape[0] == self.inputs_shape[0]:
                            outs = self.engine.run([inp])[0]
                        else:
                            outs = self.less_predict(inp)
                        t = outs.copy()
                        outputs.append(t)
                    li = i
                outputs = np.vstack(outputs)
            pickle.dump(outputs,open('./outpus.pkl','wb'))
            outputs = torch.tensor(outputs)
            outputs = F.softmax(outputs, dim=1).cpu()
            score, class_prediction = torch.max(outputs, 1)
            classes = ['basketball_shot', 'negtive']
            return score, [classes[i] for i in class_prediction]
        except Exception as e:
            raise e



#有检测跟踪模型
class Res3DModelTensorRT:

    def __init__(self):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        resnet_model_url = '/DATA/disk1/libing/online/3D-ResNets-PyTorch/data/results_size_112_56/save_30.pth'

        opt = Ops()
        opt.n_classes = 13
        opt.sample_size = (112, 56)
        opt.batch_size=1
        opt.inference_crop = 'resize'
        opt.n_input_channels = 3
        opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
        self.spatial_transform = get_spatial_transform(opt)
        engine = ModelTensorRT(self.trt_logger, resnet_model_url,opt).generate_engine()
        self.engine = Engine(engine)
        self.opt = opt
        self.inputs_shape = self.engine.inputs[0].shape
        print('engine input shape', self.inputs_shape)

    def less_predict(self, inputs):
        print('inputs batch less than engine inputs')
        inp_batch = inputs.shape[0]
        inputs = np.vstack([inputs, np.zeros((self.inputs_shape[0] - inp_batch, *self.inputs_shape[1:]),
                                             dtype=np.float16)])
        outputs = self.engine.run([inputs])[0]
        outputs = outputs[:inp_batch, :]
        return outputs

    def forward(self, clips, batch_flag=True):
        try:
            if batch_flag:
                clip = batch_preprocessing(clips, self.spatial_transform)
            else:
                clip = preprocessing(clips, self.spatial_transform)
            inputs = clip.cpu().numpy()
            inputs = np.array(inputs, copy=True, dtype=np.float16)
            inp_batch = inputs.shape[0]
            if inp_batch < self.inputs_shape[0]:
                outputs = self.less_predict(inputs)
            elif inp_batch == self.inputs_shape[0]:
                print('batch size equal ')
                outputs = self.engine.run([inputs])[0]
            else:
                print('inputs batch greater than engine inputs')
                outputs = []
                for i in range(0, inp_batch, self.inputs_shape[0]):
                    if i != 0:
                        inp = inputs[li:i, :]
                        if inp.shape[0] == self.inputs_shape[0]:
                            outs = self.engine.run([inp])[0]
                        else:
                            outs = self.less_predict(inp)
                        t = outs.copy()
                        outputs.append(t)
                    li = i
                outputs = np.vstack(outputs)
            # pickle.dump(outputs,open('./outpus.pkl','wb'))
            outputs = torch.tensor(outputs)
            outputs = F.softmax(outputs, dim=1).cpu()
            score, class_prediction = torch.max(outputs, 1)
            classes = ["ballet_foot_stretch", "negtive", "ballet_openfly", "ballet_waist", "basketball_bounce",
                       "basketball_dribble", "basketball_shoot", "soccer_carry", "soccer_highfive", "soccer_shoot",
                       "backhand_hit", "forehand_hit", "serve"]

            return score.item(), classes[class_prediction]
        except Exception as e:
            raise e

        # try:
        #     if batch_flag:
        #         clip = batch_preprocessing(clips, self.spatial_transform)
        #     else:
        #         clip = preprocessing(clips, self.spatial_transform)
        #     # don't calculate grads
        #     with torch.no_grad():
        #         # apply model to input
        #         # outputs = self.model(clip.cuda())
        #         outputs = self.engine.run([clip])[0]
        #         # apply softmax and move from gpu to cpu
        #         outputs = F.softmax(outputs, dim=1).cpu()
        #         # get best class
        #         score, class_prediction = torch.max(outputs, 1)
        #         classes = ["ballet_foot_stretch", "negtive", "ballet_openfly", "ballet_waist", "basketball_bounce",
        #                    "basketball_dribble", "basketball_shoot", "soccer_carry", "soccer_highfive", "soccer_shoot",
        #                    "backhand_hit", "forehand_hit", "serve"]
        #         return score.item(), classes[class_prediction]
        # except Exception as e:
        #     print(e)
        #     return 0, ''