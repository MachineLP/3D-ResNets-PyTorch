'''
Author: your name
Date: 2020-11-18 14:38:56
LastEditTime: 2020-11-18 17:08:57
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /machinelp/3D-ResNets-PyTorch/util_scripts/generate_csv_pp.py
'''
import subprocess
import argparse
from pathlib import Path
import os


def get_n_frames(video_path):
    return len([
        x for x in video_path.iterdir()
        if 'image' in x.name and x.name[0] != '.'
    ])

video_path = "/DATA/disk1/machinelp/3D-ResNets-PyTorch/qpark_action_2/train"

all_labels_path = os.listdir(video_path)

all_img_path_list = []
label_list = []
for per_label_path in all_labels_path:
    per_all_img_path = os.path.join(video_path, per_label_path)
    print (">>>>", per_all_img_path)
    all_video_path = os.listdir(per_all_img_path)
    print (">>>>", all_video_path)
    for per_video_path in all_video_path:
    
        # per_img_path = os.path.join(per_all_img_path, per_video_path)
        per_img_path = os.path.join('/DATA/disk1/machinelp/3D-ResNets-PyTorch/qpark_action_2/kinetics_videos/jpg', per_label_path, per_video_path)
        per_img_path_num = os.listdir(per_img_path)
        if len (per_img_path_num) < 17:
            print ("per_img_path_num<17")
            continue
        per_img_path_num = sorted(per_img_path_num)
        # print ("per_img_path_num",per_img_path_num)
        for i in range(len( per_img_path_num )):
            path_0 = os.path.join( per_img_path, per_img_path_num[i] ) 
            if int(i/10) ==0:
                path_1 = os.path.join( per_img_path, "image_0000"+str(i)+'.jpg' ) 
                os.system( "mv {} {}".format( path_0,  path_1) )
            if int(i/10) >0 and int(i/10) <10  :
                path_1 = os.path.join( per_img_path, "image_000"+str(i)+'.jpg' ) 
                os.system( "mv {} {}".format( path_0,  path_1) )
            if int(i/10) >=10 and int(i/10) <100  :
                path_1 = os.path.join( per_img_path, "image_00"+str(i)+'.jpg' ) 
                os.system( "mv {} {}".format( path_0,  path_1) )
            
        print (">>>>", per_img_path)
        all_img_path_list.append( per_img_path )
        label_list.append( per_label_path )

from pandas.core.frame import DataFrame
res = DataFrame()
res['youtube_id'] = list( all_img_path_list )
res['label'] = list( label_list )
res[ ['youtube_id', 'label'] ].to_csv('train.csv', index=False) 



video_path = "/DATA/disk1/machinelp/3D-ResNets-PyTorch/qpark_action_2/test"

all_labels_path = os.listdir(video_path)

all_img_path_list = []
label_list = []
for per_label_path in all_labels_path:
    per_all_img_path = os.path.join(video_path, per_label_path)
    print (">>>>", per_all_img_path)
    all_video_path = os.listdir(per_all_img_path)
    print (">>>>", all_video_path)
    for per_video_path in all_video_path:
    
        # per_img_path = os.path.join(per_all_img_path, per_video_path)
        per_img_path = os.path.join('/DATA/disk1/machinelp/3D-ResNets-PyTorch/qpark_action_2/kinetics_videos/jpg', per_label_path, per_video_path)
        per_img_path_num = os.listdir(per_img_path)
        if len (per_img_path_num) < 17:
            continue
        per_img_path_num = sorted(per_img_path_num)
        for i in range(len( per_img_path_num )):
            path_0 = os.path.join( per_img_path, per_img_path_num[i] ) 
            if int(i/10) ==0:
                path_1 = os.path.join( per_img_path, "image_0000"+str(i)+'.jpg' ) 
                os.system( "mv {} {}".format( path_0,  path_1) )
            if int(i/10) >0 and int(i/10) <10  :
                path_1 = os.path.join( per_img_path, "image_000"+str(i)+'.jpg' ) 
                os.system( "mv {} {}".format( path_0,  path_1) )
            if int(i/10) >=10 and int(i/10) <100  :
                path_1 = os.path.join( per_img_path, "image_00"+str(i)+'.jpg' ) 
                os.system( "mv {} {}".format( path_0,  path_1) )
            
        print (">>>>", per_img_path)
        all_img_path_list.append( per_img_path )
        label_list.append( per_label_path )

from pandas.core.frame import DataFrame
res = DataFrame()
res['youtube_id'] = list( all_img_path_list )
res['label'] = list( label_list )
res[ ['youtube_id', 'label'] ].to_csv('test.csv', index=False) 


video_path = "/DATA/disk1/machinelp/3D-ResNets-PyTorch/qpark_action_2/val"

all_labels_path = os.listdir(video_path)

all_img_path_list = []
label_list = []
for per_label_path in all_labels_path:
    per_all_img_path = os.path.join(video_path, per_label_path)
    print (">>>>", per_all_img_path)
    all_video_path = os.listdir(per_all_img_path)
    print (">>>>", all_video_path)
    for per_video_path in all_video_path:
    
        # per_img_path = os.path.join(per_all_img_path, per_video_path)
        per_img_path = os.path.join('/DATA/disk1/machinelp/3D-ResNets-PyTorch/qpark_action_2/kinetics_videos/jpg', per_label_path, per_video_path)
        per_img_path_num = os.listdir(per_img_path)
        if len (per_img_path_num) < 17:
            continue
        per_img_path_num = sorted(per_img_path_num)
        for i in range(len( per_img_path_num )):
            path_0 = os.path.join( per_img_path, per_img_path_num[i] ) 
            if int(i/10) ==0:
                path_1 = os.path.join( per_img_path, "image_0000"+str(i)+'.jpg' ) 
                os.system( "mv {} {}".format( path_0,  path_1) )
            if int(i/10) >0 and int(i/10) <10  :
                path_1 = os.path.join( per_img_path, "image_000"+str(i)+'.jpg' ) 
                os.system( "mv {} {}".format( path_0,  path_1) )
            if int(i/10) >=10 and int(i/10) <100  :
                path_1 = os.path.join( per_img_path, "image_00"+str(i)+'.jpg' ) 
                os.system( "mv {} {}".format( path_0,  path_1) )
            
        print (">>>>", per_img_path)
        all_img_path_list.append( per_img_path )
        label_list.append( per_label_path )

from pandas.core.frame import DataFrame
res = DataFrame()
res['youtube_id'] = list( all_img_path_list )
res['label'] = list( label_list )
res[ ['youtube_id', 'label'] ].to_csv('val.csv', index=False) 


