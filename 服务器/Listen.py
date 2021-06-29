"""
Copyright 2021, Yin Aihua, Tang Weiquan and Wang Junhao. 
"""
from flask import request,Flask,jsonify,Response
from plyfile import PlyData,PlyElement
from collections import OrderedDict
import matplotlib.pyplot as plt
from threading import Thread
from shutil import copyfile
import os.path as osp
import numpy as np
import requests
import argparse
import logging
import pickle
import base64
import json
import time
import math
import cv2
import sys
import os

sys.path.insert(0, osp.dirname(__file__) + '/..')

import torch.nn as nn
import torch

from utils import *
from fastmvsnet.model import FastMVSNet

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/PowerfulScanner/'

# 记录当前进度
progress = -1


# target文件夹下存放cams文件夹、images文件夹
def pack_data(index, num_view = 7, target_path=os.path.join('come'), num_virtual_plane=192, interval_scale=1.06):#num_virtual_plane=96, interval_scale=2.13):
	imgs = []
	cams = []
	height = 1152
	width = 1600

    # make_paths 把 第index个图像的路径放到最前面
	cam_paths, img_paths = make_paths(target_path, index)
	print("cam_paths : ", cam_paths)
	print("img_paths : ", img_paths)

	for view in range(num_view):
		img = cv2.imread(img_paths[view])
		cam = load_cam_dtu(cam_paths[view],
			num_depth=num_virtual_plane,
			interval_scale=interval_scale)
		imgs.append(img)
		cams.append(cam)

	h_scale = float(height) / imgs[0].shape[0]
	w_scale = float(width) / imgs[0].shape[1]
	resize_scale = h_scale
	if w_scale > h_scale:
		resize_scale = w_scale

	scaled_input_images, scaled_input_cams = scale_input(imgs, cams, scale=resize_scale)

    # crop to fit network
	croped_images, croped_cams = crop_input(scaled_input_images, scaled_input_cams,
                                                           height=height, width=width)
	ref_image = croped_images[0].copy()
	ref_cam = croped_cams[0].copy()
	for i, image in enumerate(croped_images):
		croped_images[i] = norm_image(image)

	img_list = np.stack(croped_images, axis=0)
	cam_params_list = np.stack(croped_cams, axis=0)
	img_list = torch.tensor(img_list).permute(0, 3, 1, 2).float()
	cam_params_list = torch.tensor(cam_params_list).float()

	if index == 0:
		# 第一个返回值用于三维重建，后面的返回值用于点云融合
		return {"img_list": torch.unsqueeze(img_list,dim=0), "cam_params_list": torch.unsqueeze(cam_params_list,dim=0),
            "ref_img": ref_image,}, ref_image, ref_cam
	else:
		return {"img_list": torch.unsqueeze(img_list,dim=0), "cam_params_list": torch.unsqueeze(cam_params_list,dim=0),
            "ref_img": ref_image,}


# 保存深度图
def save_depth(data2construct, preds, index, output_dir = 'go'):
	init_depth_map = preds["coarse_depth_map"].detach().numpy()[0, 0]
	init_prob_map = preds["coarse_prob_map"].detach().numpy()[0, 0]
	ref_image = data2construct["ref_img"]

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	write_pfm(os.path.join(output_dir,"{}.pfm".format(index)), init_depth_map)
	plt.imshow(init_depth_map, 'rainbow')
	plt.axis('off')
	plt.savefig(os.path.join(output_dir,"{}.jpg".format(index)),bbox_inches='tight')


def depth_image_to_point_cloud(image, depth, intrinsic, extrinsic):
	image = cv2.resize(image, (depth.shape[1], depth.shape[0]))
	u = range(0, image.shape[1])
	v = range(0, image.shape[0])

	u, v = np.meshgrid(u, v)
	u = u.astype(float)
	v = v.astype(float)

	Z = depth.astype(float) 
	X = (u - intrinsic[0, 2]) * Z / intrinsic[0, 0]
	Y = (v - intrinsic[1, 2]) * Z / intrinsic[1, 1]

	X = np.ravel(X)
	Y = np.ravel(Y)
	Z = np.ravel(Z)

	valid = Z > 0

	X = X[valid]
	Y = Y[valid]
	Z = Z[valid]

	position = np.vstack((X, Y, Z, np.ones(len(X))))
	position = np.dot(extrinsic, position)

	R = np.ravel(image[:, :, 0])[valid]
	G = np.ravel(image[:, :, 1])[valid]
	B = np.ravel(image[:, :, 2])[valid]

	points = np.transpose(np.vstack((position[0:3, :], R, G, B))).tolist()

	return points


# 将点云保存为.ply文件
def save_points_as_ply(save_path, points):
    points = np.array(points)
    vertexs = points[:,:3]
    vertex_colors = points[:,3:]
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(save_path)
    print("saving the final model to", save_path)


def save_points_as_json(save_path, points):
    vertexs = []
    vertex_colors = []
    for p in points:
    	vertexs.append(p[:3])
    	vertex_colors.append(p[3:])
    points_dic = {"loc":vertexs,"color":vertex_colors}
    string = json.dumps(points_dic)
    with open(save_path,'w') as f:
    	f.write(string)
    print("saving the final model to", save_path)



def gogogo(num):
	global progress
	appId = "wxc0c4b01053871808"
	appSecret = "3d48cbc0a04837e774db344bb9571de9"
	url_at = "https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid="+appId+"&secret="+appSecret

	# .json()将相应解析成json格式
	res1 = requests.get(url_at).json()
	access_token = res1['access_token']

	url = "https://api.weixin.qq.com/tcb/batchdownloadfile?access_token="+access_token
	for i in range(int(num)):
		file_path = '"cloud://aihuayin0125.6169-aihuayin0125-1301103558/MVS/go/'+str(i)+'.jpg"'
		data = '{"env": "aihuayin0125","file_list": [{"fileid":'+file_path+',"max_age":7200}]}'
		download_url = requests.post(url=url,data=str(data)).json()["file_list"][0]["download_url"]
		# print(res2)
		open(os.path.join('come/images/',str(i)+'.jpg'),'wb').write(requests.get(download_url).content)
		print('Successfully downloaded ' + url)


	print("开始相机标定>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
	os.system("colmap automatic_reconstructor --workspace_path ./come --image_path ./come/images")
	print("相机标定完成!")
	os.system("python colmap2mvsnet.py --dense_folder ./come/dense/0 --max_d 192 --interval_scale 1.06")
	os.system("xcopy .\\come\\dense\\0\\cams\\* .\\come\\cams\\ /E/H/C/I")

	model = FastMVSNet()
	isCPU = True
	if not isCPU:
		model = model.cuda()
	if not isCPU:
		data2construct = {k: v.cuda(non_blocking=True) for k, v in data2construct.items() if isinstance(v, torch.Tensor)}

	state_dict = torch.load('fmvs_models/model_008.pth', map_location=torch.device("cpu"))["model"]
	if isCPU:
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			name = k[7:] # remove `module.`
			new_state_dict[name] = v
		# load params
		state_dict = new_state_dict

	model.load_state_dict(state_dict, strict=False)

	url = "https://api.weixin.qq.com/tcb/uploadfile?access_token="+access_token
	for i in range(int(num)):
		print("开始重建第"+str(i)+"个视角>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
		# num_view是每次重建时扔到模型里面的图片数目，view_num是全部图片
		if i == 0:
			data2construct, ref_img, ref_cam =  pack_data(index = i, num_view = min(7,int(num)))
			print("相机矩阵: ", len(ref_cam))
			print("相机矩阵: ", ref_cam[1][0:3,0:3])
		else:
			data2construct =  pack_data(index = i, num_view = min(7,int(num)))
		# model = nn.DataParallel(model)
		starttime = time.time()
		preds = model(data2construct, (0.25, 0.5), (0.75, 0.15), isGN=True, isTest=True)
		endtime = time.time()
		print("reconstruction time for view_", i, " ： ", endtime - starttime, " sec")
		save_depth(data2construct, preds, i)
		if i == 0:
			points = depth_image_to_point_cloud(ref_img, preds['coarse_depth_map'][0][0].detach().numpy(), ref_cam[1][0:3,0:3], ref_cam[0])
			# print("点云的形状 ： ", np.array(points).shape)
			ply_path = os.path.join('go','points.json')
			save_points_as_json(ply_path, points)
			del points
			file_path = "MVS/back/points.json"
			data = '{"env": "aihuayin0125","path": "'+file_path+'"}'

			res = requests.post(url=url,data=str(data)).json()
			with open(ply_path,'rb') as f:
				bfile = f.read()
			upload_url = res["url"]
			upload_body = fields = {
					"key":file_path,
					"Signature":res["authorization"],
					"x-cos-security-token":res["token"],
					"x-cos-meta-fileid":res["cos_file_id"],
					"file":bfile 
			}
			upload_res = requests.post(url=upload_url,files=upload_body).status_code
			if(upload_res == 204):
				print("Successfully uploaded "+file_path)
			# print("点云 ： ", points)
		del data2construct, preds

		read_path = os.path.join('go',str(i)+'.jpg')
		file_path = "MVS/back/"+str(i)+".jpg"
		data = '{"env": "aihuayin0125","path": "'+file_path+'"}'

		res = requests.post(url=url,data=str(data)).json()
		with open(read_path,'rb') as f:
			bfile = f.read()
		upload_url = res["url"]
		upload_body = fields = {
				"key":file_path,
				"Signature":res["authorization"],
				"x-cos-security-token":res["token"],
				"x-cos-meta-fileid":res["cos_file_id"],
				"file":bfile 
		}
		upload_res = requests.post(url=upload_url,files=upload_body).status_code
		if(upload_res == 204):
			print("Successfully uploaded "+upload_url)
		progress = i
	print("重建完成!")
	# 清理垃圾
	garbages = [".\\come\\cams",".\\come\\dense",".\\come\\sparse"]
	for g in garbages:
		os.system("rd /s/q "+g)


# 定义路由
@app.route("/fastmvs/<int:img_num>", methods=['POST','GET'])
def fastmvs(img_num):
	print(img_num)
	t1 = Thread(target=gogogo, args=(img_num,))
	t1.start()
	return "ok"
 

# 定义路由
@app.route("/ask_and_report", methods=['POST','GET'])
def ask_and_report():
	return str(progress)


if __name__ == "__main__":
    app.run()
