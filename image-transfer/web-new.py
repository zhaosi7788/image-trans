# coding:utf-8

from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import colorsys
import random

import string
import tensorflow as tf
import numpy as np
import os
import model
import time
import vlib.plot as plot
import vlib.save_images as save_img
import vlib.load_data as load_data
import vgg_simple as vgg
import scipy.misc as scm

import model
from train import Train



app = Flask(__name__)

slim = tf.contrib.slim

def load_style_img(styleImgPath):
    img = tf.read_file(styleImgPath)
    style_img = tf.image.decode_jpeg(img, 3)

    style_img = tf.image.resize_images(style_img, [256, 256])

    style_img = load_data.img_process(style_img, True)  # True for substract means

    images = tf.expand_dims(style_img, 0)
    style_imgs = tf.concat([images, images, images, images], 0)  # batch is 4
    # style_imgs = tf.image.resize_images(style_imgs, [256, 256])

    return style_imgs

def load_test_img(img_path):
    style_img = tf.read_file(img_path)

    style_img = tf.image.decode_jpeg(style_img, 3)
    shape = tf.shape(style_img)

    style_img = tf.image.resize_images(style_img, [shape[0], shape[1]])
    style_img = load_data.img_process(style_img, True)

    images = tf.expand_dims(style_img, 0)
    return images



class Train(object):
    def __init__(self, sess, args,style):
        self.sess = sess
        self.batch_size = 4
        self.img_size = 256

        self.img_dim = 3
        self.gamma = 0.7
        self.lamda = 0.001
        self.load_model = False
        self.max_step = 500000
        self.save_step = 10000
        self.lr_update_step = 100000
        self.img_save = 500

        self.args = args
        self.style = style

    def build_model(self):
        data_path = self.args.train_data_path

        imgs = load_data.get_loader(data_path, self.batch_size, self.img_size)

        style_imgs = load_style_img(self.args.style_data_path)

        with slim.arg_scope(model.arg_scope()):
            gen_img, variables = model.gen_net(imgs, reuse=False, name='transform')

            with slim.arg_scope(vgg.vgg_arg_scope()):
                gen_img_processed = [load_data.img_process(image, True)
                                     for image in tf.unstack(gen_img, axis=0, num=self.batch_size)]

                f1, f2, f3, f4, exclude = vgg.vgg_16(tf.concat([gen_img_processed, imgs, style_imgs], axis=0))

                gen_f, img_f, _ = tf.split(f3, 3, 0)
                content_loss = tf.nn.l2_loss(gen_f - img_f) / tf.to_float(tf.size(gen_f))

                style_loss = model.styleloss(f1, f2, f3, f4)

                # load vgg model
                vgg_model_path = self.args.vgg_model
                vgg_vars = slim.get_variables_to_restore(include=['vgg_16'], exclude=exclude)
                # vgg_init_var = slim.get_variables_to_restore(include=['vgg_16/fc6'])
                init_fn = slim.assign_from_checkpoint_fn(vgg_model_path, vgg_vars)
                init_fn(self.sess)
                # tf.initialize_variables(var_list=vgg_init_var)
                print('vgg s weights load done')

            self.gen_img = gen_img

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            self.content_loss = content_loss
            self.style_loss = style_loss*self.args.style_w
            self.loss = self.content_loss + self.style_loss
            self.opt = tf.train.AdamOptimizer(0.0001).minimize(self.loss, global_step=self.global_step, var_list=variables)

        all_var = tf.global_variables()
        # init_var = [v for v in all_var if 'beta' in v.name or 'global_step' in v.name or 'Adam' in v.name]
        init_var = [v for v in all_var if 'vgg_16' not in v.name]
        init = tf.variables_initializer(var_list=init_var)
        self.sess.run(init)

        self.save = tf.train.Saver(var_list=variables)



    def test(self,img_path,result_path):
        print ('test model')
        test_img_path = img_path
        test_img = load_test_img(test_img_path)
        # test_img = tf.random_uniform(shape=(1, 500, 800, 3), minval=0, maxval=1.)
        test_img = self.sess.run(test_img)
        with slim.arg_scope(model.arg_scope()):

            gen_img, _ = model.gen_net(test_img, reuse=False, name='transform')

            # load model
            model_path = 'model_saved/'+self.style + '.ckpt'

            vars = slim.get_variables_to_restore(include=['transform'])
            # vgg_init_var = slim.get_variables_to_restore(include=['vgg_16/fc6'])
            init_fn = slim.assign_from_checkpoint_fn(model_path, vars)
            init_fn(self.sess)
            # tf.initialize_variables(var_list=vgg_init_var)
            print('vgg s weights load done')

            gen_img = self.sess.run(gen_img)
            save_img.save_images(gen_img, result_path)


import argparse

parser = argparse.ArgumentParser()    #记得修改文件名  加上data-0000-1111    结果
parser.add_argument('-is_training', help='train or test', type=bool, default=False)
parser.add_argument('-vgg_model', help='the path of pretrained vgg model', type=str,
                    default='/home/liu/Tensorflow-Project/temp/model/vgg_16.ckpt')
# parser.add_argument('-transfer_model', help='the path of transfer net model', type=str,
#                     default='model_saved/star.ckpt') #  模型名称
parser.add_argument('-train_data_path', help='the path of train data', type=str,
                    default='/home/liu/Downloads/train2014')
parser.add_argument('-style_data_path', help='the path of style image', type=str, default=os.getcwd() + '/img/star.jpg')
parser.add_argument('-test_data_path', help='the path of style image', type=str, default='2.jpg')    #测试图片
# parser.add_argument('-new_img_name', help='the path of style image', type=str, default='result.jpg')  #结果图片
parser.add_argument('-style_w', help='the weight of style loss', type=float, default=100)

args = parser.parse_args()

# parser = argparse.ArgumentParser()    #记得修改文件名  加上data-0000-1111   训练
# parser.add_argument('-is_training', help='train or test', type=bool, default=True)
# parser.add_argument('-vgg_model', help='the path of pretrained vgg model', type=str,
#                     default='/home/liu/Tensorflow-Project/temp/model/vgg_16.ckpt')
# parser.add_argument('-transfer_model', help='the path of transfer net model', type=str,
#                     default='model_saved/star.ckpt')
# parser.add_argument('-train_data_path', help='the path of train data', type=str,
#                     default='/home/liu/Downloads/train2014')
# parser.add_argument('-style_data_path', help='the path of style image', type=str, default='img/chouxiang.jpg')
# parser.add_argument('-test_data_path', help='the path of style image', type=str, default='dog.jpg')
# parser.add_argument('-new_img_name', help='the path of style image', type=str, default='transfer.jpg')
# parser.add_argument('-style_w', help='the weight of style loss', type=float, default=100)
#
# args = parser.parse_args()




'''导入模型'''
@app.route('/')
def about():
    return redirect(url_for('upload'))
@app.route('/upload', methods=['POST', 'GET'])
def upload():

    if request.method == 'POST':
        style = request.form['style']
        st = time.time()

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        sess = sess = tf.Session()
        Model = Train(sess, args, style)

        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        ramdonnum = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        upload_path = os.path.join(basepath,'static/uploads',ramdonnum+f.filename)  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        result_path = os.path.join(basepath, 'static/result', ramdonnum + f.filename)
        print(upload_path)
        f.save(upload_path)
        real_path = ramdonnum+f.filename
        print(real_path)
        Model.test(upload_path,result_path)
        print('the test is over')
        if style =='star':
            flag = '已经完成“梵高星空”图像风格迁移'
        elif style=='face':
            flag = '已经完成“人脸素描”图像风格迁移'
        elif style == 'gold':
            flag = '已经完成“金色抽象”图像风格迁移'
        elif style == 'chouxiang':
            flag = '已经完成“毕加索”图像风格迁移'
        elif style == 'girl':
            flag = '已经完成“哥特玻璃”图像风格迁移'
        elif style == 'scream':
            flag = '已经完成“蒙克尖叫”图像风格迁移'

        print("转换时间为：",time.time()-st)
        return render_template('upload.html',imgpath = real_path,flag = flag)
    return render_template('upload.html')



if __name__ == '__main__':

    app.run('0.0.0.0',port=5004,threaded=True)
