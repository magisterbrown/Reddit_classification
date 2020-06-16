#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from glob import glob
from torch import nn
from torch import optim
from torch.utils import data
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch_lr_finder import LRFinder
from fastai.layers import AdaptiveConcatPool2d,Flatten
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


from torchvision import  transforms
from torchvision.transforms import ToTensor
from essential_generators import *
import re
import random
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import imageio
import imgaug.augmenters as iaa
from multiprocessing import Pool
import seaborn as sns


class Dataset_ram(data.Dataset):

    def __init__(
        self,
        cats,
        samples,
        normalizer,
        ):

        allprob = 0
        self.samples = samples
        self.normalizer = normalizer
        self.preloader = None
        self.repeat_preloader = False
        self.inpiter = []
        self.probs = []
        self.generators = []
        self.laind = []
        for val in cats:
            allprob += val[0]
            self.probs.append(val[0])
            self.generators.append(val[1])
            self.laind.append(-1)

    def __len__(self):
        '''Denotes the total number of samples'''

        return self.samples

    def __getitem__(self, index):
        '''Generates one sample of data'''

        try:
            return next(self.preloader)
        except:

            if self.repeat_preloader and self.inpiter:
                self.preloader = iter(self.inpiter)
                return next(self.preloader)
            return self.singgen()

    def singgen(self):
        gener = np.random.choice(self.generators, 1, p=self.probs)[0]
        gege = gener.generate_image()
        X = self.normalizer(gege[0])
        y = gege[1]
        return (X, y)

    def pregen(self, size):
        self.inpiter = []
        pd.Series(np.arange(size)).apply(lambda x: self.preappend(x))
        self.preloader = iter(self.inpiter)

    def preappend(self, x):
        if x % 500 == 0:
            print("Generated: "+str(x))
        self.inpiter.append(self.singgen())

    def loop_pregen(self):
        self.repeat_preloader = True

    def unloop_pregen(self):
        self.repeat_preloader = False

    def unplug_pregen(self):

        self.preloader = None
        self.inpiter = []


#!/usr/bin/python
# -*- coding: utf-8 -*-


class ImageGenerator:

    def __init__(
        self,
        path_to_images,
        transformer,
        fonts,
        words_num,
        fon_mul=6e-6,
        split_rn=(0.5, 1),
        from_folder=True,
        diffimages=False,
        cat=0,
        lim=-1,
        ):

        self.diffimages = diffimages
        self.cat = cat

        self.itms = []
        if from_folder:
            il = glob(path_to_images + '*.jpg')
        else:
            il = [path_to_images]

        if lim > 0:
            il = il[:lim]

        self.cut = 0
        with Pool(16) as p:
            self.itms = p.map(self.loadim, il)

    # pd.Series(il).apply(lambda x:self.loadim(x,self.itms))

        self.tfms = transformer
        self.alnum = alnum = list('abcdefghijklmnopqrstuvwxyz1234567890'
                                  )
        self.text_generator = DocumentGenerator()

        self.fons_sz = (30, 40)
        self.fon_mul = fon_mul
        self.spdep = self.lin_dep(self.fons_sz[0] // 2, self.fons_sz[1]
                                  * 2, split_rn[0], split_rn[1])
        self.wddep = self.lin_dep(self.fons_sz[0] // 2, self.fons_sz[1]
                                  * 2, words_num[0], words_num[1])
        self.fonts = []

        for val in fonts:
            self.fonts.append([ImageFont.truetype(val, i) for i in
                              range(self.fons_sz[0] // 2,
                              self.fons_sz[1] * 2)])

    def generate_image(
        self,
        num_words=4,
        one_style=False,
        fonts=False,
        fonts_tp=False,
        filcol=False,
        ):

        y = random.randint(0, len(self.itms) - 1)
        rand = self.itms[y]
        arr = self.tfms(images=[rand])[0]
        pilmg = Image.fromarray(arr)
        pil_draw = ImageDraw.Draw(pilmg)

        if not fonts:
            fonts = int(np.random.randint(self.fons_sz[0],
                        self.fons_sz[1]) - self.fons_sz[0] // 2)
        if not fonts_tp:
            font_tp = np.random.randint(4)
        if not filcol:
            filcol = (np.random.randint(255), np.random.randint(255),
                      np.random.randint(255))

        im_size = pilmg.size
        im_s = im_size[0] * im_size[1]
        holders = []
        for val in range(num_words):

            num_fonts = len(self.fonts[0])
            if not one_style:
                fonts = int(np.random.randint(self.fons_sz[0],
                            self.fons_sz[1]) - self.fons_sz[0] // 2)
                font_tp = np.random.randint(4)
                filcol = (np.random.randint(255),
                          np.random.randint(255),
                          np.random.randint(255))

            idfont = np.clip(int(fonts * im_s * self.fon_mul), 0,
                             num_fonts - 1)
            szfont = idfont + self.fons_sz[0]
            wnum = int(self.wddep[0] * szfont + self.wddep[1])
            spprob = self.spdep[0] * szfont + self.spdep[1]

            gened = self.gen_text(wnum, spprob)
            font = self.fonts[font_tp][idfont]

            text_size = pil_draw.multiline_textsize(gened, font=font)
            (xp1, yp1) = self.rand_placer(im_size, text_size, holders)
            if xp1 == 'burgh':
                break
            align = random.choice(['left', 'center', 'right'])
            pil_draw.text((xp1, yp1), gened, fill=filcol, font=font,
                          align=align)
            holders.append((xp1, text_size[0] + 20, yp1, text_size[1]
                           + 20))

        if self.diffimages:
            return (pilmg, y + self.cat)
        else:
            return (pilmg, self.cat)

    def rand_placer(
        self,
        im_size,
        text_size,
        holders,
        ):
        x_var = np.arange(0, im_size[0] - text_size[0])
        while True:
            if x_var.size == 0:
                return ('burgh', 'burgh')
            x_pos = np.random.choice(x_var)
            space = np.arange(0, im_size[1] - text_size[1])

            for val in holders:
                fillzone_beg = val[0] - text_size[0]
                fillzone_end = val[0] + val[1]
                if fillzone_beg <= x_pos <= fillzone_end:

                    sprange = np.arange(val[2] - text_size[1], val[2]
                            + val[3])
                    space = space[~np.isin(space, sprange)]
                    if space.size == 0:
                        x_var = x_var[~np.isin(x_var,
                                np.arange(fillzone_beg, fillzone_end))]
                        break
            if space.size > 0:
                y_pos = random.choice(space)
                return (x_pos, y_pos)

    def liner(self, line, rate):
        line = list(line.split())
        bdpr = 0
        for (ind, val) in enumerate(line[:-1]):
            if np.tanh(bdpr) > 0.7 + np.random.uniform(0, 0.25):
                line[ind] = line[ind] + '\n'
                bdpr = 0
            else:
                line[ind] = line[ind] + ' '
            bdpr += rate * len(val)
        return ''.join(line)

    def rep_chars(
        self,
        inpute,
        replacement,
        alnum,
        ):
        newChars = map(lambda x: (x if x in alnum else replacement),
                       inpute)
        return re.sub(' +', ' ', ''.join(newChars))

    def gen_text(self, wordnum, spprob):
        prc = np.ceil(wordnum * 0.2)
        sent = self.text_generator.gen_sentence(min_words=wordnum
                - prc, max_words=wordnum + prc)
        gened = self.liner(self.rep_chars(sent.lower(), ' ',
                           self.alnum), spprob)
        return gened

    def lin_dep(
        self,
        x1,
        x2,
        y1,
        y2,
        ):
        a = (y1 - y2) / (x1 - x2)
        b = y1 - x1 * a
        return (a, b)

    def loadim(self, x):
        return imageio.imread(x)


def freeze(model, freeze):
    if 1 in freeze:
        for val in (model[0])[:6].parameters():
            val.requires_grad = False
    if 2 in freeze:
        for val in (model[0])[6:].parameters():
            val.requires_grad = False
    if 3 in freeze:
        for val in model[1:]:
            for par in val.parameters():
                par.requires_grad = False


def unfreeze(model):
    for val in model[0].parameters():
        val.requires_grad = True
    for val in model[1:]:
        for par in val.parameters():
            par.requires_grad = True

def train_cat(
    model,
    pram_groups,
    max_lr,
    steps,
    loader,
    ):

    optimizer = optim.Adam(pram_groups, lr=0.0000001)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
            steps_per_epoch=steps + 1, epochs=1)
    criterion = nn.CrossEntropyLoss()

    dataiter = iter(loader)
    model.cuda()
    model.train()
    con = 0
    running_loss = 0.0
    for val in range(steps):
        (inputs, labls) = dataiter.next()
        con += 1

        inputs = inputs.cuda()
        labls = labls.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labls)
        loss.backward()

        optimizer.step()
        scheduler.step()
        if con % 5 == 0:
            print("Loss: "+str(loss.item()))

def lr_find(
    model,
    pram_groups,
    loader,
    optimizer,
    ):

    criterion = nn.CrossEntropyLoss()

    lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
    lr_finder.range_test(loader, end_lr=1000, num_iter=100)
    lr_finder.plot()
    lr_finder.reset()

def to_probs(gens, prob):
    res = []
    for val in gens:
        res.append((prob, val))
    return res

def addoutput(model, add):
    outputs = model[9].out_features + add
    old_out = model[9].out_features

    bias = True

    newl = nn.Linear(in_features=model[9].in_features,
                     out_features=outputs, bias=bias)
    lay = model[9]
    newl.weight[:old_out] = lay.weight
    newl.bias[:old_out] = lay.bias
    newl.weight = nn.Parameter(newl.weight)
    newl.bias = nn.Parameter(newl.bias)

    model[-1] = newl
    model.cuda()

def image_load_here(image_path):
    try:
      im = Image.open('data/images/' + image_path)
      return normalizer(im)
    except:
      im = Image.open("happy_animal.jpg")
      return normalizer(im)

def predict(model, df, butch_num):

    buthes = np.array_split(df, butch_num)
    model.eval()
    for (key, val) in enumerate(buthes):
        if key % 10 == 0:
            print("Bunches: "+str(key))
        with Pool(16) as p:
            ffef = torch.stack(p.map(image_load_here,
                               buthes[key]['fname']))
        predictions = model(ffef.cuda())
        buthes[key]['type'] = torch.argmax(predictions,
                dim=1).detach().cpu().numpy()
        soft = F.softmax(predictions).detach().cpu().numpy()
        sofes = np.array([soft[key, val] for (key, val) in
                         enumerate(buthes[key]['type'])])
        buthes[key]['prob'] = sofes
        ffef.cpu()
        predictions.cpu()
        del ffef
        del predictions
    return pd.concat(buthes)

def plots_cat(
    val,
    perclass,
    size,
    df,
    tresh,
    ):
    (fig, axbo) = plt.subplots(perclass + 1, 1, figsize=(size, perclass
                               * size))
    tt = df[np.logical_and(df['type'] == val, df['prob']
            > tresh)].sort_values(by=['prob']).head(perclass)
    axbo[0].axis('off')
    axbo[0].text(x=0.5, y=0, s=tt['prob'][0],
                 horizontalalignment='center')
    ks = 1
    for (key, ser) in tt.iterrows():
        img = Image.open('data/images/' + ser['fname'])
        axbo[ks].imshow(img)
        ks += 1
    plt.show()

def dists(classes, size, df):
    gs_kw = dict(height_ratios=[1, 1.5])
    (fig, axbo) = plt.subplots(2, classes[1] - classes[0],
                               figsize=((classes[1] - classes[0])
                               * size, 2.5 * size), gridspec_kw=gs_kw)
    for val in range(classes[0], classes[1]):
        tt = df[df['type'] == val]
        fnmae = tt.sort_values(by=['prob']).iloc[-1, :]['fname']
        img = Image.open('data/images/' + fnmae)
        axbo[0][val - classes[0]].imshow(img)
        axbo[0][val - classes[0]].axis('off')

        sns.violinplot(x='prob', data=tt, ax=axbo[1][val - classes[0]],
                       orient='v')
        axbo[1][val - classes[0]].set_ylim([0, 1])
        axbo[1][val - classes[0]].spines['top'].set_visible(False)
        axbo[1][val - classes[0]].spines['right'].set_visible(False)
        axbo[1][val - classes[0]].spines['bottom'].set_visible(False)
        axbo[1][val - classes[0]].title.set_text(str(val) + ': '
                + str(tt.shape[0]))
    plt.show()

def set_parameters(model):
    pram_groups = [{'params': (model[0])[:6].parameters()},
                   {'params': (model[0])[6:].parameters()},
                   {'params': model[1:].parameters()}]
    return pram_groups

def memes_by_date(memese):
  memese["secs"]=memese["DateTime"].astype('datetime64[s]').astype('int')
  memese.sort_values(by=["secs"],inplace=True)
  chunks = np.array_split(memese, 7)
  time = [chunks[0].iloc[0]["DateTime"][:10]]
  for val in chunks:
    time.append(val.iloc[-1]["DateTime"][:10])

  (fig, ax) = plt.subplots(figsize=(16, 4))
  sns.violinplot(x='secs', data=memese, ax=ax)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.set_xticklabels(['a']+time)
  ax.set_xlabel('')
  ax.set_title('Memes poseted by date', size=20)
  ax.set_xlim([memese["secs"].min(), memese["secs"].max()])
  plt.show()

def top_memes(num,memes):
  fig, axbo = plt.subplots(1, num, figsize=(5*num, 5))
  toplot = memese.sort_values(by=["ups"],ascending=False).head(num)
  ind = 0
  for key,val in toplot.iterrows():
    img=Image.open("data/images/"+val["fname"])
    axbo[ind].imshow(img)
    axbo[ind].axis('off')
    axbo[ind].set_title("Ups: "+str(int(val["ups"])), size=20)
    ind+=1

def real_val(x,probs):
  if(x["prob"]>probs[int(x["type"])]):
    return x["type"]
  else:
    return 0

def fezt_ext(memese, butches, filterim=None):
    if filterim:
        fnames = pd.DataFrame(memese[memese['rtype'] == 0]['fname'],
                              columns=['fname'] + list(range(1024)))
    else:
        fnames = pd.DataFrame(memese['fname'], columns=['fname']
                              + list(range(1024)))
    buthes = np.array_split(fnames, butches)
    nemmod.eval()
    for (key, val) in enumerate(buthes):
        if key % 10 == 0:
            print('Bunches: ' + str(key))
        with Pool(16) as p:
            ffef = torch.stack(p.map(image_load_here,
                               buthes[key]['fname']))
        predictions = nemmod(ffef.cuda())
        buthes[key].iloc[:, 1:] = \
            predictions.detach().cpu().data.numpy()

        ffef.cpu()
        del ffef
        del predictions
    return pd.concat(buthes)
    
def pca(mndim, pcaed):
    pca = PCA(n_components=2)
    x = mndim.iloc[:, 1:]
    principalComponents = pca.fit_transform(x)
    pcaed[['comp1', 'comp2']] = principalComponents
    return pcaed

def plot_cats(
    pcaed,
    eps,
    min_samples,
    fig,
    more,
    ):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    pcaed['cluster'] = dbscan.fit_predict(pcaed[['comp1', 'comp2']])
    pcaedn = pcaed[pcaed['cluster'] > more]

    (fig, ax) = plt.subplots(1, 2, figsize=fig)
    ax[0].scatter(pcaed['comp1'], pcaed['comp2'], c=pcaed['cluster'],
                  cmap='hot')
    ax[0].set_title('Original', fontsize=15)
    ax[1].scatter(pcaedn['comp1'], pcaedn['comp2'], c=pcaedn['cluster'
                  ], cmap='hot')
    ax[1].set_title('No outliers', fontsize=15)
    plt.show()
    return pcaedn


def plot_top_clusters(
    cats,
    top,
    size,
    clusters,
    ):
    ind = clusters.groupby('cluster').count().sort_values(by=['fname'
            ])['fname']

    (fig, ax) = plt.subplots(top, cats, figsize=(size[0], size[1]))
    for cat in range(cats):
        inds = clusters[clusters['cluster'] == ind.index[(cat + 1)
                        * -1]].index[:top]
        ax[0][cat].set_title('Images: ' + str(ind.iloc[(cat + 1) * -1]))
        for plc in range(top):
            ax[plc][cat].axis('off')
            im = Image.open('data/images/'
                            + clusters.loc[inds[plc]]['fname'])
            ax[plc][cat].imshow(im)


def plt_meme(memes, fig, data):
    (fig, axbo) = plt.subplots(1, len(memes), figsize=(fig[0], fig[1]))
    ind = 0
    for (key, val) in memes.iteritems():
        img = Image.open('data/images/' + data[key])
        axbo[ind].imshow(img)
        axbo[ind].axis('off')
        axbo[ind].set_title('ID: ' + str(int(key)), size=20)
        ind += 1


def bar_ch(
    topvals,
    name,
    xlb,
    ylb,
    size,
    ):

    (fig, ax) = plt.subplots(1, 1, figsize=size)
    barvl = sns.barplot(x=topvals.index, y=topvals, color='#8C96DB',
                        order=topvals.index, ax=ax)
    barvl.spines['top'].set_visible(False)
    barvl.spines['right'].set_visible(False)
    barvl.set_xlabel(xlb)
    barvl.set_ylabel(ylb)
    barvl.title.set_text(name)

    ypos = 0
    for rect in topvals:
        disp = rect
        plt.text(ypos, rect, s=disp, ha='center', va='bottom')
        ypos += 1


def memes_prc(
    memese,
    num,
    roll,
    popular_memes,
    ):
    beg = memese['secs'][0]
    end = memese['secs'][-1]
    step = (end - beg) // num

    dfee = []
    for val in range(beg + step, end, step):
        batch = memese[np.logical_and(memese['secs'] > val - step,
                       memese['secs'] < val)]
        ser = batch.groupby(by=['rtype'])['prob'].count() \
            / batch.shape[0]
        dfee.append(ser)

    toline = pd.DataFrame(dfee)[popular_memes.index].fillna(0)
    index = pd.Series(pd.date_range(memese['DateTime'][0],
                      memese['DateTime'][-1], periods=num
                      - 1)).astype('str').apply(lambda x: x[:10])
    toline.set_index(index.values, inplace=True)

    (fig, ax) = plt.subplots(1, 1, figsize=(18, 8))
    res = toline.rolling(roll).mean()
    ax.plot(res)
    plt.xticks(rotation=30)
    ax.legend(res.columns)
    ax.set_xlabel('Date')
    ax.set_ylabel('prc')
    ax.title.set_text('Percentage of memes by date')
    plt.show()


def denormalize(tensor, mean, std):
    std = torch.tensor([[std]]).permute(2, 1, 0)
    mean = torch.tensor([[mean]]).permute(2, 1, 0)

    return tensor.mul(std).add(mean)
    
def heat_map(
    model,
    image,
    ax,
    interp,
    ):
    normim = normalizer(Image.open(image))
    pred = model(normim[None].cuda())

    ax.imshow(denormalize(normim, mean, std).permute(1, 2, 0))
    ax.imshow(torch.mean(pred, 1)[0].detach().cpu(), alpha=0.6,
              extent=(0, 224, 224, 0), interpolation=interp,
              cmap='magma')

def plt_heat_meme(
    memes,
    fig,
    data,
    model,
    interp,
    ):
    (fig, axbo) = plt.subplots(1, len(memes), figsize=(fig[0], fig[1]))
    ind = 0
    for (key, val) in memes.iteritems():
        heat_map(model, 'data/images/' + data[key], axbo[ind], interp)
        axbo[ind].axis('off')
        axbo[ind].set_title('ID: ' + str(int(key)), size=20)
        ind += 1


def cust_pred(
    model,
    image,
    best_by_cat,
    interp,
    ):
    normim = normalizer(Image.open(image))
    mask_mod = cmod[0]
    pred = model(normim[None].cuda())
    key = torch.argmax(pred).cpu().detach().numpy()
    softs = F.softmax(pred).cpu().detach().numpy()[0]
    img = Image.open('data/images/' + best_by_cat[key])
    (_, ax) = plt.subplots(2, 1, figsize=(8, 16))
    heat_map(mask_mod, image, ax[0], interp)
    ax[0].axis('off')
    ax[0].set_title('Your image:', size=25)

    ax[1].imshow(img)
    ax[1].axis('off')
    ax[1].set_title('Predicted class: ' + str(key) + ' With prob: '
                    + str(round(softs[key], 2)), size=20)


def plt_meme(memes, shape):
    (fig, ax) = plt.subplots(shape[0], shape[1], figsize=(40, 10))
    for col in range(shape[0]):
        for row in range(shape[1]):
            name = memes['fname'].sample(1).values[0]
            img = Image.open('data/images/' + name)
            ax[col, row].imshow(img)
            ax[col, row].axis('off')
    return fig
