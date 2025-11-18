#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
@author: ninad#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

from keras.models import Sequential, Model, model_from_json
from keras.layers import Lambda, Dropout, Flatten, Dense, Activation, Concatenate
from keras.layers import Conv2D, Convolution2D, BatchNormalization, Input, Embedding, Reshape, GaussianNoise
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, Add, LeakyReLU
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras import losses, optimizers
from keras.initializers import Constant
import keras.backend as K
import tensorflow as tf

import os
import const
from config import Config

config = Config.neural_net
config_rn = Config.run_neural

# K.set_floatx('float64')

def to_fx_const(v):
    return K.constant(v, dtype=K.floatx())

def cast_fx(x):
    return K.cast(x, K.floatx())

def to_f32(x):
    return K.cast(x, 'float32')   # -> float32

'''def _check_weights_for_nan(model, tag):
    import numpy as np
    bad = []
    for l in model.layers:
        try:
            ws = l.get_weights()
        except:
            ws = []
        for i, w in enumerate(ws):
            if not np.isfinite(w).all():
                bad.append((l.name, i, np.nanmin(w), np.nanmax(w)))
    if bad:
        print("[WARN] {} has NaN/Inf weights:".format(tag))
        for b in bad:
            print("  layer={}, idx={}, min={}, max={}".format(b[0], b[1], b[2], b[3]))
    else:
        print("[OK] {} weights finite".format(tag))'''

def model_pilotnet():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    input_vel = (1,)
    # input_delta = (3,)
    ######model#######
    img_input = Input(shape=input_shape)
    vel_input = Input(shape=input_vel)
    # delta_input = Input(shape=input_delta)
    
    lamb_str = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    lamb_vel = Lambda(lambda x: x/40)(vel_input)
    # lamb_delta = Lambda(lambda x: x)(delta_input)
    
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu', name='conv2d_1')(lamb_str)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(64, (5, 5), strides=(2,2), activation='relu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(64, (3, 3), padding='same',activation='relu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(64, (3, 3), padding='same',activation='relu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_v = Dense(50, activation='relu', name='fc_v')(lamb_vel)
    # fc_c = Dense(50, activation='relu', name='fc_d')(lamb_delta)
    fc_1 = Dense(100, activation='relu', name='fc_1')(flat)
    conc = Concatenate()([fc_1, fc_v])
    # conc = Concatenate()([fc_1, fc_v, fc_c])
    fc_2 = Dense(50 , activation='relu', name='fc_2')(conc)
    fc_3 = Dense(10 , activation='relu', name='fc_3')(fc_2)
    fc_out = Dense(config['num_outputs'], name='fc_out')(fc_3)
    
    model = Model(inputs=[img_input, vel_input], outputs=[fc_out])
    # model = Model(inputs=[img_input, vel_input, delta_input], outputs=[fc_out])
    return model


def model_pilotnet_origin():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    # input_delta = (3,)
    ######model#######
    img_input = Input(shape=input_shape)
    # vel_input = Input(shape=input_vel)
    # delta_input = Input(shape=input_delta)
    
    lamb_str = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    # lamb_delta = Lambda(lambda x: x)(delta_input)
    
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu', name='conv2d_1')(lamb_str)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(64, (5, 5), strides=(2,2), activation='relu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(64, (3, 3), padding='same',activation='relu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(64, (3, 3), padding='same',activation='relu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    # fc_c = Dense(50, activation='relu', name='fc_d')(lamb_delta)
    fc_1 = Dense(100, activation='relu', name='fc_1')(flat)
    # conc = Concatenate()([fc_1, fc_v, fc_c])
    fc_2 = Dense(50 , activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(10 , activation='relu', name='fc_3')(fc_2)
    fc_out = Dense(1, name='fc_out')(fc_3)
    
    model = Model(inputs=[img_input], outputs=[fc_out])
    # model = Model(inputs=[img_input, vel_input, delta_input], outputs=[fc_out])
    return model

def pretrained_pilot(base_model_path):
    base_weightsfile = base_model_path+'.h5'
    base_modelfile   = base_model_path+'.json'
    
    base_json_file = open(base_modelfile, 'r')
    base_loaded_model_json = base_json_file.read()
    base_json_file.close()
    base_model = model_from_json(base_loaded_model_json)
    base_model.load_weights(base_weightsfile)
    # if config['style_train'] is True:
    base_model.trainable = False
    
    # print(base_model.get_layer('conv2d_3').output)
    return base_model

###############################
# Conditional Gan
###############################

#---------------------------------
# Generator
#---------------------------------
def generator_cgan(base_model_path):

    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    input_vel = (1,)
    input_gvel = (1,)
    input_style = (1,)
    out_dim = (config['num_outputs']-1) if config.get('only_thr_brk', False) else config['num_outputs']
    emb_dim = config.get('style_embed_dim', 16)   # 임베딩 차원
    num_styles = config.get('num_styles', 2)      # 주행 스타일 클래스

    img_input = Input(shape=input_shape)
    vel_input = Input(shape=input_vel)
    gvel_input = Input(shape=input_gvel)
    style_input = Input(shape=input_style, dtype='int32')   # 일단 정수로 라벨링해서 수행 스타일 구분

    base_model = pretrained_pilot(base_model_path)
    pretrained_model_last = Model(base_model.input, base_model.get_layer('fc_out').output, name='base_model_output')
    pretrained_model_conv3 = Model(base_model.input, base_model.get_layer('conv2d_3').output, name='base_model_conv2d_3')
    pretrained_model_conv5 = Model(base_model.input, base_model.get_layer('conv2d_last').output, name='base_model_conv2d_last')
    
    # freeze
    for m in [pretrained_model_last, pretrained_model_conv3, pretrained_model_conv5]:
        m.trainable = False

    # forward
    base_model_last_output = pretrained_model_last([img_input, vel_input])
    base_model_conv3_output = pretrained_model_conv3([img_input, vel_input])
    base_model_conv5_output = pretrained_model_conv5([img_input, vel_input])

    # Conv Feature concate
    add_base_layer = Add()([base_model_conv3_output, base_model_conv5_output])

    # 임베딩
    fc_vel = Dense(100, activation='relu', name='fc_vel')(vel_input)
    fc_gvel = Dense(100, activation='relu', name='fc_gvel')(gvel_input)
    fc_base_out = Dense(100, activation='relu', name='fc_base_out')(base_model_last_output)
    style_emb = Embedding(num_styles, emb_dim, name='style_emb')(style_input)     # (B, 1, emb_dim)
    style_emb = Reshape((emb_dim,), name='style_flat')(style_emb)                 # (B, emb_dim)

    # conv -> Flatten -> FC
    flat = Flatten()(add_base_layer)
    fc_1 = Dense(500, activation='relu', name='fc_1')(flat)
    conc = Concatenate()([fc_base_out, fc_1, fc_vel, fc_gvel, style_emb])      # 정수 라벨링 추가
    
    # Noise 추가
    conc = GaussianNoise(0.02, name='gn_noise')(conc)
    fc_2 = Dense(200, activation='relu', name='fc_2')(conc)
    drop = Dropout(rate=0.2)(fc_2)
    fc_3 = Dense(100, activation='relu', name='fc_3')(drop)

    # 출력 차원 유지
    gen_out = Dense(out_dim, name='fc_out')(fc_3)

    Generator_cGAN = Model(inputs=[img_input, vel_input, gvel_input], outputs=gen_out, name='Generator_cGAN')

    return Generator_cGAN

#---------------------------------
# Discriminator
#---------------------------------

def discriminator_cgan(base_model_path):

    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    input_vel = (1,)
    input_gvel = (1,)
    input_style = (1,)
    out_dim = (config['num_outputs']-1) if config.get('only_thr_brk', False) else config['num_outputs']
    emb_dim = config.get('style_embed_dim', 16)   # 임베딩 차원
    num_styles = config.get('num_styles', 2)      # 주행 스타일 클래스

    
    img_input = Input(shape=input_shape, name='d_img_input')
    vel_input = Input(shape=input_vel, name='d_vel_input')
    gvel_input = Input(shape=input_gvel, name='gvel_input')
    y_input = Input(shape=(out_dim), name='d_ctrl_input')  #real/fake vector
    style_input = Input(shape=input_style, name='d_style_label')  # 주행 스타일 라벨링

    # Image + vel 조건 Feature 추출
    base = pretrained_pilot(base_model_path)
    feat_last = Model(base.input, base.get_layer('fc_out').output, name='d_base_fc')
    feat_last.trainable = False
    
    cond_feat = feat_last([img_input, vel_input])
    cond_feat = Dense(128, activation='relu')(cond_feat)

    # vel 임베딩
    v = Dense(64, activation='relu')(vel_input)
    gv = Dense(64, activation='relu')(gvel_input)

    # 제어 vector와 Condition Concate
    y_proj = Dense(128, activation='relu')(y_input)

    # 주행 스타일 라벨 임베딩
    style_emb = Embedding(num_styles, emb_dim, name='d_style_emb')(style_input)
    style_emb = Reshape((emb_dim,), name='d_style_flat')(style_emb)

    d_concat = Concatenate(name='d_concat')([cond_feat, v, gv, y_proj, style_emb])
    d_h = Dense(256, activation='relu')(d_concat)
    d_h = Dropout(0.3)(d_h)
    d_h = Dense(128, activation='relu')(d_h)

    d_out = Dense(1, activation='sigmoid', name='d_out')(d_h)

    Discriminator_cGAN = Model(inputs=[img_input, vel_input, gvel_input, y_input], outputs=d_out, name='Discriminator_cGan')
    return Discriminator_cGAN

#---------------------------------
# Combined
#---------------------------------

def Conditional_GAN(base_model_path, lambda_l1=100.0, lr_d=2e-4, lr_g=2e-4, beta_1=0.5, beta_2=0.999):
    # 개별 모델
    G = generator_cgan(base_model_path)
    D = discriminator_cgan(base_model_path)

    # Discriminator Compile
    D.compile(optimizer=optimizers.Adam(lr=lr_d, beta_1=beta_1, beta_2=beta_2),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    # D 는 고정, G 만 업데이트
    D.trainable = False

    # Input (Generator와 동일)
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    out_dim = (config['num_outputs']-1) if config.get('only_thr_brk', False) else config['num_outputs']

    c_img = Input(shape=input_shape)
    c_vel = Input(shape=(1,))
    c_gvel = Input(shape=(1,))
    c_style = Input(shape=(1,), dtype='int32', name='c_style_label')

    y_real = Input(shape=(out_dim,))  # L1 target(실제 제어벡터)

    y_fake = G([c_img, c_vel, c_gvel, c_style])
    d_fake = D([c_img, c_vel, c_gvel, c_style, y_fake])

    CGAN = Model(inputs=[c_img, c_vel, c_gvel, c_style, y_real],
                 outputs=[d_fake, y_fake],
                 name='CGAN')

    CGAN.compile(
        optimizer=optimizers.Adam(lr_g, beta_1=beta_1, beta_2=beta_2),
        loss=['binary_crossentropy', 'mae'],
        loss_weights=[1.0, lambda_l1]
    )
    return G, D, CGAN


#---------------------------------
# Conditional VAE
#---------------------------------

def _cvae_sampling(args):
    z_mean, z_logvar = args
<<<<<<< HEAD
    z_logvar = K.clip(z_logvar, -10, 1.0)
=======
    z_logvar = K.clip(z_logvar, -10, 1.)
>>>>>>> ab1c142b3f69d4c7abe74dd778de3073058b7c7a
    eps = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(0.5 * z_logvar) * eps

def build_cvae_with_label(
        base_model_path,
<<<<<<< HEAD
        latent_dim = config['latent_dim'],
        recon_weight = config['recon_weight'],
        lr = config['vae_lr'],
        emb_dim = config['style_embed_dim']
=======
        latent_dim = 16,
        recon_weight = 1.0,
        lr = 1e-8,
        emb_dim = None
>>>>>>> ab1c142b3f69d4c7abe74dd778de3073058b7c7a
):
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    input_vel = (1,)
    input_gvel = (1,)
    input_style = (1,)
    out_dim = config['num_outputs']
    emb_dim = config['style_embed_dim']   # 임베딩 차원
    num_styles = config['num_styles']    # 주행 스타일 클래스


    img_input = Input(shape=input_shape)
    vel_input = Input(shape=input_vel)
    gvel_input = Input(shape=input_gvel)
    style_input = Input(shape=input_style, dtype='int32')   # 일단 정수로 라벨링해서 수행 스타일 구분
    y_true = Input(shape=(out_dim,))

<<<<<<< HEAD
    lamb_img = Lambda(lambda x: x/127.5 - 1.0)(img_input) 
    lamb_vel = Lambda(lambda x: x/40)(vel_input) 
    lamb_gvel_input = Lambda(lambda x: (K.clip(x, -20.0, 20.0) + 20.0) / 40.0)(gvel_input)

    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu', name='conv2d_1')(lamb_img)
=======
    lamb_str = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    lamb_gvel_input = Lambda(lambda x: x/40)(gvel_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu', name='conv2d_1')(lamb_str)
>>>>>>> ab1c142b3f69d4c7abe74dd778de3073058b7c7a
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(64, (5, 5), strides=(2,2), activation='relu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(64, (3, 3), padding='same',activation='relu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(64, (3, 3), padding='same',activation='relu', name='conv2d_last')(conv_4)
    flat = Flatten(name='cvae_flat_conv')(conv_5)

    base = pretrained_pilot(base_model_path)
    m_tail = Model(base.input, base.get_layer('fc_out').output, name='cvae_backbone_fc')
<<<<<<< HEAD
    m_tail.trainable = False

    tail = m_tail([lamb_img, lamb_vel]) 
    # print(type(tail))

    f_conv = Dense(128, activation='relu', name='cvae_fc1')(flat)
    f_gvel = Dense(64, activation='relu', name='cvae_fc_gvel')(lamb_gvel_input)
    f_tail = Dense(256, activation='relu', name='cvae_fc_tail')(tail)
=======
    tail = m_tail([img_input, vel_input]) 
    # print(type(tail))
    # tail_rescale = tf.divide(tail, config['steering_angle_scale'])

    f_conv = Dense(500, activation='relu', name='cvae_fc1')(flat)
    f_gvel = Dense(100, activation='relu', name='cvae_fc_gvel')(lamb_gvel_input)
    f_tail = Dense(100, activation='relu', name='cvae_fc_tail')(tail)
    #f_tail = Dense(100, activation='relu', name='cvae_fc_tail')(tail_rescale)
    
    #steering, throttle, brake = tail
    #f_str = Dense(100, activation='relu', name='cvae_fc_str')(steering)
    #f_thr = Dense(100, activation='relu', name='cvae_fc_thr')(throttle)
    #f_brk = Dense(100, activation='relu', name='cvae_fc_brk')(brake)
>>>>>>> ab1c142b3f69d4c7abe74dd778de3073058b7c7a

    s_emb = Embedding(num_styles, emb_dim, name='cvae_style_emb')(style_input)
    s_emb = Reshape((emb_dim,), name='cvae_style_flat')(s_emb)

    cond = Concatenate(name='cvae_cond_concat')([f_tail, f_conv, f_gvel, s_emb])
<<<<<<< HEAD
    #cond = GaussianNoise(0.02, name='cvae_cond_noise')(cond)
=======
    #cond = Concatenate(name='cvae_cond_concat')([f_str, f_thr, f_brk, f_conv, f_gvel, s_emb])
    cond = GaussianNoise(0.05, name='cvae_cond_noise')(cond)
>>>>>>> ab1c142b3f69d4c7abe74dd778de3073058b7c7a
    cond = Dense(256, activation='relu', name='cvae_cond_fc')(cond)

    # Encoder
    enc_in = Concatenate(name='cvae_enc_in')([y_true, cond])
    h = Dense(256, activation='relu', name='cvae_enc_h1')(enc_in)
    h = Dropout(0.1, name='cvae_enc_drop')(h)
    h = Dense(128, activation='relu', name='cvae_enc_h2')(h)
    h = Dense(64, activation='relu', name='cvae_enc_h3')(h)
    h = Dense(32, activation='relu', name='cvae_enc_h4')(h)
    z_mean = Dense(latent_dim, name='cvae_z_mean')(h)
    z_logvar = Dense(latent_dim, name='cvae_z_logvar', bias_initializer=Constant(-2.0))(h)
    z = Lambda(_cvae_sampling, name='cvae_z')([z_mean, z_logvar])

    # Decoder
    dec_cond_in = Input(shape=(K.int_shape(cond)[-1],), name='dec_cond_in')
    dec_z_in = Input(shape=(latent_dim,), name='dec_z_in')
    x = Concatenate(name='dec_concat')([dec_cond_in, dec_z_in])
<<<<<<< HEAD
    x = Dense(256, activation='relu', name='dec_h1')(x)
    x = Dense(128, activation='relu', name='dec_h2')(x)
    x = Dense(64, activation='relu', name='dec_h3')(x)
    x = Dense(32, activation='relu', name='dec_h4')(x)

    #steer = Dense(1, activation='tanh', name='steer')(x)
    #thr = Dense(1, activation='sigmoid', name='throttle')(x)
    #brk = Dense(1, activation='sigmoid', name='brake')(x)

    #y_out = Concatenate(name='dec_out')([steer, thr, brk])
    y_out = Dense(out_dim, name='dec_out')(x)
=======
    x = Dense(200, activation='relu', name='dec_h1')(x)
    x = Dropout(0.2, name='dec_drop')(x)
    x = Dense(100, activation='relu', name='dec_h2')(x)

    steer = Dense(1, activation='tanh', name='steer')(x)
    thr = Dense(1, activation='sigmoid', name='throttle')(x)
    brk = Dense(1, activation='sigmoid', name='brake')(x)

    y_out = Concatenate(name='dec_out')([steer, thr, brk])
    # y_out = Dense(out_dim, name='dec_out')(x)
>>>>>>> ab1c142b3f69d4c7abe74dd778de3073058b7c7a
    Decoder = Model(inputs=[dec_cond_in, dec_z_in], outputs=y_out, name='CVAE_Decoder')

    y_pred = Decoder([cond, z])

    # 학습용 VAE : 입력=[img, vel, gvel, style, y_true] -> 출력=y_pred
    VAE = Model(inputs=[img_input, vel_input, gvel_input, style_input, y_true],
                outputs=y_pred,
                name='CVAE_train')
    
    # KL Loss
    def stable_kl(z_mean, z_logvar):
<<<<<<< HEAD
        z_logvar = K.clip(z_logvar, -10, 1.0)   # 클리핑을 통해 KL 안정화
        return -0.5 * K.mean(K.sum(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=-1))
=======
        logvar = K.clip(z_logvar, -10.0, 1.0)   # 클리핑을 통해 KL 안정화
        return -0.5 * K.mean(K.sum(1 + logvar - K.square(z_mean) - K.exp(logvar), axis=-1))

>>>>>>> ab1c142b3f69d4c7abe74dd778de3073058b7c7a
    kl = stable_kl(z_mean, z_logvar)

    kl_w = K.variable(0.0, name='kl_w')  # 0.0에서 시작해서 점진적으로 1.0까지 올릴 예정
    VAE.add_loss(kl_w * kl)

    VAE.kl_w = kl_w
<<<<<<< HEAD
   # VAE.add_loss(kl)

    # Compile
    VAE.compile(optimizer=optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5, amsgrad=True),
=======
    #VAE.add_loss(kl)

    # Compile
    VAE.compile(optimizer=optimizers.Adam(lr=lr, decay=config['decay'], clipnorm=1.0, beta_1=0.9, beta_2=0.999),
>>>>>>> ab1c142b3f69d4c7abe74dd778de3073058b7c7a
                loss=losses.mean_squared_error,
                loss_weights=[recon_weight],
                metrics=['mse'])
    
    # Inference
    # CondEncoder : (img, vel, gvel, style) -> cond
    CondEnc = Model(inputs=[img_input, vel_input, gvel_input, style_input],
                    outputs=cond,
                    name='CVAE_CondEncoder')

    # mean z=0 사용 Predictor : (img, vel, gvel, style) -> y_hat
    def _zeros_like_latent(t):
        b = K.shape(t)[0]
        return K.zeros((b, latent_dim))

    def _rand_latent(t):
        b = K.shape(t)[0]
        return K.random_normal((b, latent_dim))

    z0 = Lambda(_zeros_like_latent, name='cvae_z_zero')(style_input)
    y_mean = Decoder([CondEnc([img_input, vel_input, gvel_input, style_input]), z0])
    Predictor = Model(inputs=[img_input, vel_input, gvel_input, style_input],
                      outputs=y_mean,
                      name='CVAE_Predictor_Mean')
    
    zrand = Lambda(_rand_latent, name='cvae_z_rand')(style_input)
    y_rand = Decoder([CondEnc([img_input, vel_input, gvel_input, style_input]), zrand])
    Sampler = Model(inputs=[img_input, vel_input, gvel_input, style_input],
                    outputs=y_rand,
                    name='CVAE_Predictor_Sample')

    return VAE, CondEnc, Decoder, Predictor, Sampler

def model_style1(base_model_path):

    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    input_vel = (1,)
    input_gvel = (1,)
    ######model#######
    img_input = Input(shape=input_shape)
    vel_input = Input(shape=input_vel)
    gvel_input = Input(shape=input_gvel)

    
    base_model1 = pretrained_pilot(base_model_path)
    base_model2 = pretrained_pilot(base_model_path)
    base_model3 = pretrained_pilot(base_model_path)
    pretrained_model_last = Model(base_model1.input, base_model1.get_layer('fc_out').output, name='base_model_output')
    pretrained_model_conv3 = Model(base_model2.input, base_model2.get_layer('conv2d_3').output, name='base_model_conv2d_3')
    pretrained_model_conv5 = Model(base_model3.input, base_model3.get_layer('conv2d_last').output, name='base_model_conv2d_last')
    # if config['style_train'] is True:
    pretrained_model_conv3.trainable = False
    pretrained_model_conv5.trainable = False
    pretrained_model_last.trainable = False
        
    base_model_last_output = pretrained_model_last([img_input, vel_input])
    base_model_conv3_output = pretrained_model_conv3([img_input, vel_input])
    base_model_conv5_output = pretrained_model_conv5([img_input, vel_input])
    
    add_base_layer = Add()([base_model_conv3_output, base_model_conv5_output])
    fc_vel = Dense(100, activation='relu', name='fc_vel')(vel_input)
    fc_gvel = Dense(100, activation='relu', name='fc_gvel')(gvel_input)
    fc_base_out = Dense(100, activation='relu', name='fc_base_out')(base_model_last_output)
    flat = Flatten()(add_base_layer)
    fc_1 = Dense(500, activation='relu', name='fc_1')(flat)
    conc = Concatenate()([fc_base_out, fc_1, fc_vel, fc_gvel])
    fc_2 = Dense(200, activation='relu', name='fc_2')(conc)
    drop = Dropout(rate=0.2)(fc_2)
    fc_3 = Dense(100, activation='relu', name='fc_3')(drop)
    
    # print(base_model_last_output[0].shape)
    if config['only_thr_brk'] is True: 
        fc_out = Dense(config['num_outputs']-1, name='fc_out')(fc_3)
    else:
        fc_out = Dense(config['num_outputs'], name='fc_out')(fc_3)
    # fc_str = Dense(1, name='fc_str')(base_str)
    # fc_thr = Dense(1, name='fc_thr')(fc_3)
    # fc_brk = Dense(1, name='fc_brk')(fc_3)
    
    model = Model(inputs=[img_input, vel_input, gvel_input], outputs=[fc_out])
    # model = Model(inputs=[img_input, vel_input], outputs=[fc_str, fc_thr, fc_brk])
    return model

def model_style2(base_model_path):

    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    input_vel = (1,)
    input_gvel = (1,)
    ######model#######
    img_input = Input(shape=input_shape)
    vel_input = Input(shape=input_vel)
    gvel_input = Input(shape=input_gvel)
    
    base_model1 = pretrained_pilot(base_model_path)
    base_model2 = pretrained_pilot(base_model_path)
    base_model3 = pretrained_pilot(base_model_path)
    pretrained_model_last = Model(base_model1.input, base_model1.get_layer('fc_out').output, name='base_model_output')
    pretrained_model_conv3 = Model(base_model2.input, base_model2.get_layer('conv2d_3').output, name='base_model_conv2d_3')
    pretrained_model_conv5 = Model(base_model3.input, base_model3.get_layer('conv2d_last').output, name='base_model_conv2d_last')
    # if config['style_train'] is True:
    pretrained_model_conv3.trainable = False
    pretrained_model_conv5.trainable = False
    pretrained_model_last.trainable = False
        
    base_model_last_output = pretrained_model_last([img_input, vel_input])
    base_model_conv3_output = pretrained_model_conv3([img_input, vel_input])
    base_model_conv5_output = pretrained_model_conv5([img_input, vel_input])
    
    add_base_layer = Add()([base_model_conv3_output, base_model_conv5_output])
    fc_vel = Dense(100, activation='relu', name='fc_vel')(vel_input)
    fc_gvel = Dense(100, activation='relu', name='fc_gvel')(gvel_input)
    fc_base_out = Dense(100, activation='relu', name='fc_base_out')(base_model_last_output)
    flat = Flatten()(add_base_layer)
    fc_1 = Dense(500, activation='relu', name='fc_1')(flat)
    conc = Concatenate()([fc_base_out, fc_1, fc_vel, fc_gvel])
    fc_2 = Dense(200, activation='relu', name='fc_2')(conc)
    drop = Dropout(rate=0.2)(fc_2)
    fc_3 = Dense(100, activation='relu', name='fc_3')(drop)
    
    # print(base_model_last_output[0].shape)
    if config['only_thr_brk'] is True: 
        fc_out = Dense(config['num_outputs']-1, name='fc_out')(fc_3)
    else:
        fc_out = Dense(config['num_outputs'], name='fc_out')(fc_3)
    # fc_str = Dense(1, name='fc_str')(base_str)
    # fc_thr = Dense(1, name='fc_thr')(fc_3)
    # fc_brk = Dense(1, name='fc_brk')(fc_3)
    
    model = Model(inputs=[img_input, vel_input, gvel_input], outputs=[fc_out])
    # model = Model(inputs=[img_input, vel_input], outputs=[fc_str, fc_thr, fc_brk])
    return model

def model_nonlstm():
    input_shape = (config['lstm_timestep'], config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    input_vel = (config['lstm_timestep'], 1,)
    ######model#######
    img_input = Input(shape=input_shape)
    vel_input = Input(shape=input_vel)
    lamb_str = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    lamb_vel = Lambda(lambda x: x/40)(vel_input)
    
    conv_1 = TimeDistributed(Conv2D(24, (5, 5), strides=(2,2), activation='relu'), name='conv2d')(lamb_str)
    conv_2 = TimeDistributed(Conv2D(36, (5, 5), strides=(2,2), activation='relu'), name='conv2d_2')(conv_1)
    conv_3 = TimeDistributed(Conv2D(64, (5, 5), strides=(2,2), activation='relu'), name='conv2d_3')(conv_2)
    conv_4 = TimeDistributed(Conv2D(64, (3, 3), padding='same',activation='relu'), name='conv2d_4')(conv_3)
    conv_5 = TimeDistributed(Conv2D(64, (3, 3), padding='same',activation='relu'), name='conv2d_last')(conv_4)
    flat   = TimeDistributed(Flatten())(conv_5)
    fc_v   = TimeDistributed(Dense( 50, activation='relu'), name='fc_v')(lamb_vel)
    fc_1   = TimeDistributed(Dense(100, activation='relu'), name='fc_1')(flat)
    conc   = Concatenate()([fc_1, fc_v])
    fc_2   = TimeDistributed(Dense( 50, activation='relu'), name='fc_2')(conc)
    fc_3   = TimeDistributed(Dense( 10, activation='relu'), name='fc_3')(fc_2)
    # lstm   = LSTM(1, return_sequences=False, name='lstm_c')(fc_3)
    # fc_out = Dense(config['num_outputs'], name='fc_out')(fc_3)
    flat_2 = Flatten()(fc_3)
    fc_str = Dense(1, name='fc_str')(flat_2)
    fc_thr = Dense(1, name='fc_thr')(flat_2)
    model = Model(inputs=[img_input, vel_input], outputs=[fc_str, fc_thr])
    return model
    

class NetModel:
    def __init__(self, model_path, base_model_path=None):
        self.model = None
        self.base_model = None
        self.vae = None
        self.condenc = None
        self.decoder = None
        self.predictor = None
        self.sampler = None
        # self.condtap = None #디버깅용
        self.gen = None
        self.disc = None
        self.cgan = None
        model_name = model_path[model_path.rfind('/'):] # get folder name
        self.name = model_name.strip('/')

        self.model_path = model_path
        self.base_model_path = base_model_path
        #self.config = Config()

        # to address the error:
        #   Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
        os.environ["CUDA_VISIBLE_DEVICES"]=str(config['gpus'])
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.tensorflow_backend.set_session(sess)
        # with tf.device('/cpu:0'):
        
        self._model(base_model_path=base_model_path)

    ###########################################################################
    #
    def _model(self, base_model_path = None):
        if config['network_type'] == const.NET_TYPE_PILOT:
            self.model = model_pilotnet()
        elif config['network_type'] == const.NET_TYPE_CVAE:
            (self.vae, self.condenc, self.decoder, self.predictor, self.sampler) = build_cvae_with_label(
                base_model_path,
<<<<<<< HEAD
                latent_dim = config['latent_dim'],
                recon_weight = config['recon_weight'],
                lr = config['vae_lr'],
                emb_dim = config['style_embed_dim']
=======
                latent_dim = config.get('latent_dim', 16),
                recon_weight = config.get('recon_weight', 1.0),
                lr = config.get('vae_lr', 1e-6),
                emb_dim = config.get('style_embed_dim', None)
>>>>>>> ab1c142b3f69d4c7abe74dd778de3073058b7c7a
            )
            if config['style_train'] is True:
                self.model = self.vae
            else:
                self.model = self.predictor
                self.base_model = model_pilotnet()

        elif config['network_type'] == const.NET_TYPE_CGAN:
            self.gen, self.disc, self.cgan = Conditional_GAN(base_model_path)
            self.model = self.gen
        elif config['network_type'] == const.NET_TYPE_STYLE1:
            self.model = model_style1(base_model_path)
            self.base_model = model_pilotnet()
        elif config['network_type'] == const.NET_TYPE_STYLE2:
            self.model = model_style2(base_model_path)
            self.base_model = model_pilotnet()
        elif config['network_type'] == const.NET_TYPE_STYLE3:
            self.model, _, _,  = build_cvae_with_label(base_model_path)
            self.base_model = model_pilotnet()
        elif config['network_type'] == const.NET_TYPE_STYLE4:
            self.model = model_style2(base_model_path)
            self.base_model = model_pilotnet()
        elif config['network_type'] == const.NET_TYPE_NONLSTM:
            self.model = model_nonlstm()
        else:
            exit('ERROR: Invalid neural network type.')
        self.summary()
        self._compile()



    # ###########################################################################
    # #
    # def _mean_squared_error(self, y_true, y_pred):
    #     diff = K.abs(y_true - y_pred)
    #     if (diff < config['steering_angle_tolerance']) is True:
    #         diff = 0
    #     return K.mean(K.square(diff))

    ###########################################################################
    #
    def _compile(self):
        if config['network_type'] == const.NET_TYPE_CVAE:
            return
        
        if config['lstm'] is True:
            learning_rate = config['lstm_lr']
        else:
            learning_rate = config['cnn_lr']
        decay = config['decay']

        self.model.compile(loss=losses.mean_squared_error,
                    optimizer=optimizers.Adam(lr=learning_rate, decay=decay, clipvalue=1), 
                    metrics=['accuracy'])
        
        if config['style_run'] is True:
            self.base_model.compile(loss=losses.mean_squared_error,
                        optimizer=optimizers.Adam(lr=learning_rate, decay=decay, clipvalue=1), 
                        metrics=['accuracy'])


    ###########################################################################
    #
    # save model
    def save(self, model_name):
        if config['network_type'] == const.NET_TYPE_CGAN:
            self.gen.save_weights(model_name + '_G.h5', overwrite=True)
            # self.disc.save_weights(model_name + '_D.h5', overwrite=True)    # Discriminator 필요 시 주석해제 
            return
        
        if config['network_type'] == const.NET_TYPE_CVAE:
            self.vae.save_weights(model_name + '_VAE.h5', overwrite=True)
            return

        json_string = self.model.to_json()
        #weight_filename = self.model_path + '_' + Config.config_yaml_name \
        #    + '_N' + str(config['network_type'])
        open(model_name+'.json', 'w').write(json_string)
        self.model.save_weights(model_name+'.h5', overwrite=True)


    ###########################################################################
    # model_path = '../data/2007-09-22-12-12-12.
    def weight_load(self, load_model_name):
    
        from keras.models import model_from_json

        # json_string = self.model.to_json()
        # open(load_model_name+'.json', 'w').write(json_string)
        # self.model = model_from_json(open(load_model_name+'.json').read())
        self.model.load_weights(load_model_name)
        self._compile()
    
    
    def load(self):
        # Conditional GAN
        if config['network_type'] == const.NET_TYPE_CGAN:
            self.gen.load_weights(self.model_path + '_G.h5')
            # self.disc.load_weights(self.model_path + '_D.h5')
            return
        
        # Conditional VAE
        if config['network_type'] == const.NET_TYPE_CVAE:
            self.vae.load_weights(self.model_path + '_VAE.h5')
            # _check_weights_for_nan(self.vae, "CVAE (after loading)")
            return

        from keras.models import model_from_json
        # self.model = model_from_json(open(self.model_path+'.json').read())
        self.model.load_weights(self.model_path+'.h5')
        
        if config['style_run'] is True:
            self.base_model.load_weights(self.base_model_path+'.h5')
        self._compile()

    ###########################################################################
    #
    # show summary
    def summary(self):
        if config['network_type'] == const.NET_TYPE_CGAN:
            print('=== CGAN: Generator ==='); self.gen.summary()
            print('=== CGAN: Discriminator ==='); self.disc.summary()
            print('=== CGAN: Combined ==='); self.cgan.summary()
            return
        
        if config['network_type'] == const.NET_TYPE_CVAE:
            if config['style_train'] is True:
                print('=== CVAE: VAE (train) ==='); self.vae.summary()
            else:
                print('=== CVAE: Predictor (inference) ==='); self.model.summary()
            return
        
        self.model.summary()
        if config['style_run'] is True:
            self.base_model.summary()

