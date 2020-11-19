import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from opts import parse_opts
from torch import nn
from mean import get_mean_std

def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)

def get_spatial_transform(opt):
        normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
        spatial_transform = [Resize(opt.sample_size)]
        if opt.inference_crop == 'center':
            spatial_transform.append(CenterCrop(opt.sample_size))
        spatial_transform.append(ToTensor())
        spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
        spatial_transform = Compose(spatial_transform)
        return spatial_transform

def preprocessing(clip, spatial_transform):
    # Applying spatial transformations
    if spatial_transform is not None:
        spatial_transform.randomize_parameters()
        # Before applying spatial transformation you need to convert your frame into PIL Image format (its not the best way, but works)
        clip = [spatial_transform(Image.fromarray(np.uint8(img)).convert('RGB')) for img in clip]
    # Rearange shapes to fit model input
    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
    clip = torch.stack((clip,), 0)
    return clip

def predict(clip, model, spatial_transform, classes):
    # Set mode eval mode
    model.eval()
    # do some preprocessing steps
    clip = preprocessing(clip, spatial_transform)
    # don't calculate grads
    with torch.no_grad():
        # apply model to input
        outputs = model(clip)
        # apply softmax and move from gpu to cpu
        outputs = F.softmax(outputs, dim=1).cpu()
        # get best class
        score, class_prediction = torch.max(outputs, 1)
    print( ">>>>", score, class_prediction )
    # As model outputs a index class, if you have the real class list you can get the output class name
    # something like this: classes = ['jump', 'talk', 'walk', ...]
    #if classes != None:
    #    return score[0], classes[class_prediction[0]]
    return score, class_prediction


opt = parse_opts()
opt.n_input_channels = 3
opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
model = generate_model(opt)
model.fc = nn.Linear(model.fc.in_features, 2)
pretrain_path = '/DATA/disk1/machinelp/3D-ResNets-PyTorch/qpark_action_2/results/save_150.pth'
# pretrain_path = '/DATA/disk1/machinelp/3D-ResNets-PyTorch/qpark_action_2/r3d50_KM_200ep.pth'
pretrain = torch.load(pretrain_path, map_location='cpu')
model.load_state_dict(pretrain['state_dict'])

spatial_transform = get_spatial_transform(opt)
# predict( clip, model, spatial_transform, classes=2 )



import cv2
# we create the video capture object cap
# cap = cv2.VideoCapture(0)
video_path = '/DATA/disk1/libing/online/vlog-ai-server/src/data/input/ch3_20201102191500_20201102192000.mp4'
video_path = './basketball_test.mp4'
# video_path = '/DATA/disk1/phzhao/dataset/qpark_action/negtive/seq145_tennis_swing_neg.mov'
#video_path = '/DATA/disk1/phzhao/dataset/qpark_action/soccer_shot/seq100_pos.mp4'
cap = cv2.VideoCapture(video_path)

#width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
w = int(cap.get(3))
h = int(cap.get(4))

fps = cap.get(cv2.CAP_PROP_FPS)
fps= round(fps)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out_vid = cv2.VideoWriter("out.avi", fourcc, fps, (w, h))

# Frame's list for HAR
full_clip = []

# ... your code ...
i = 0
while True:
        ret, frame = cap.read()
        
        # Save frame's list
        #if i % 1 == 0:
        full_clip.append(frame)
        if len(full_clip)>16:
            probs, class_names = predict( full_clip, model, spatial_transform, classes=2 )    
            print (i/fps, ">>>>>", probs, class_names)
            cv2.putText(frame, str(class_names), (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 255), 2)
            cv2.putText(frame, str(probs), (200, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 255), 2)
            for m in range(1): 
                del full_clip[0] 
        i += 1
        if j>600:
            break
        out_vid.write(frame)

        
        # ... your code ...

        # show us frame with detection
        # cv2.imshow("Web cam input", frame)
        #if cv2.waitKey(25) & 0xFF == ord("q"):
        #    cv2.destroyAllWindows()
        #    break
out_vid.release()
cap.release()
cv2.destroyAllWindows()




