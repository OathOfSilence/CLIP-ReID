# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    def add_attribute_loss(loss, gender_score=None, age_score=None, gender=None, age=None):
        loss_details = {
            'gender_loss': None,
            'age_loss': None,
            'weighted_gender_loss': None,
            'weighted_age_loss': None,
        }
        if gender_score is not None and gender is not None and cfg.MODEL.GENDER_LOSS_WEIGHT > 0 and (gender >= 0).any():
            gender_loss = F.cross_entropy(gender_score, gender, ignore_index=-1)
            weighted_gender_loss = cfg.MODEL.GENDER_LOSS_WEIGHT * gender_loss
            loss = loss + weighted_gender_loss
            loss_details['gender_loss'] = gender_loss
            loss_details['weighted_gender_loss'] = weighted_gender_loss
        if age_score is not None and age is not None and cfg.MODEL.AGE_LOSS_WEIGHT > 0 and (age >= 0).any():
            age_loss = F.cross_entropy(age_score, age, ignore_index=-1)
            weighted_age_loss = cfg.MODEL.AGE_LOSS_WEIGHT * age_loss
            loss = loss + weighted_age_loss
            loss_details['age_loss'] = age_loss
            loss_details['weighted_age_loss'] = weighted_age_loss
        return loss, loss_details

    def format_loss_return(loss, loss_details, return_details):
        if return_details:
            return loss, loss_details
        return loss

    if sampler == 'softmax':
        def loss_func(score, feat, target, gender_score=None, age_score=None, gender=None, age=None, return_details=False):
            loss = F.cross_entropy(score, target)
            loss, loss_details = add_attribute_loss(loss, gender_score, age_score, gender, age)
            return format_loss_return(loss, loss_details, return_details)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, i2tscore=None, gender_score=None, age_score=None, gender=None, age=None, return_details=False):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) 
                    else:   
                        TRI_LOSS = triplet(feat, target)[0]
                    
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    if i2tscore != None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                    loss, loss_details = add_attribute_loss(loss, gender_score, age_score, gender, age)

                    return format_loss_return(loss, loss_details, return_details)
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                    loss, loss_details = add_attribute_loss(loss, gender_score, age_score, gender, age)

                    return format_loss_return(loss, loss_details, return_details)
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


