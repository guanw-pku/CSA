"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn

from lib.word_vectors import obj_edge_vectors
# from lib.transformer import transformer
from lib.transformer_wk import transformer_wk
from lib.transformer_img import transformer_img
from lib.fpn.box_utils import center_size
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from lib.extract_bbox_features import extract_feature_given_bbox_base_feat_torch
import pdb

class ObjectClassifier(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, mode='sgdet', obj_classes=None, is_wks=True):
        super(ObjectClassifier, self).__init__()
        self.classes = obj_classes
        self.mode = mode
        self.is_wks = is_wks

        #----------add nms when sgdet
        self.nms_filter_duplicates = True
        self.max_per_img =64
        self.thresh = 0.01

        #roi align
        self.RCNN_roi_align = ROIAlign((7, 7), 1.0/16.0, 0)

        embed_vecs = obj_edge_vectors(obj_classes[1:], wv_type='glove.6B', wv_dir='data', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes)-1, 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=0.01 / 10.0),
                                       nn.Linear(4, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))
        self.obj_dim = 192 # TODO our 192, baseline 2048
        self.decoder_lin = nn.Sequential(nn.Linear(self.obj_dim + 200 + 128, 1024),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, len(self.classes)))

    def clean_class(self, entry, b, class_idx):
        final_boxes = []
        final_dists = []
        final_feats = []
        final_labels = []
        for i in range(b):
            scores = entry['distribution'][entry['boxes'][:, 0] == i]
            pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i]
            feats = entry['features'][entry['boxes'][:, 0] == i]
            pred_labels = entry['pred_labels'][entry['boxes'][:, 0] == i]

            new_box = pred_boxes[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_feats = feats[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores = scores[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores[:, class_idx-1] = 0
            if new_scores.shape[0] > 0:
                new_labels = torch.argmax(new_scores, dim=1) + 1
            else:
                new_labels = torch.tensor([], dtype=torch.long).cuda(0)

            final_dists.append(scores)
            final_dists.append(new_scores)
            final_boxes.append(pred_boxes)
            final_boxes.append(new_box)
            final_feats.append(feats)
            final_feats.append(new_feats)
            final_labels.append(pred_labels)
            final_labels.append(new_labels)

        entry['boxes'] = torch.cat(final_boxes, dim=0)
        entry['distribution'] = torch.cat(final_dists, dim=0)
        entry['features'] = torch.cat(final_feats, dim=0)
        entry['pred_labels'] = torch.cat(final_labels, dim=0)
        return entry

    def forward(self, entry):
        if self.mode  == 'predcls':
            entry['pred_labels'] = entry['labels']
            return entry
        elif self.mode == 'sgcls':
            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            if self.training:
                entry['distribution'] = self.decoder_lin(obj_features)
                entry['pred_labels'] = entry['labels']
            else:
                entry['distribution'] = self.decoder_lin(obj_features)

                box_idx = entry['boxes'][:,0].long()
                b = int(box_idx[-1] + 1)

                entry['distribution'] = torch.softmax(entry['distribution'][:, 1:], dim=1)
                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(obj_features.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    local_human_idx = torch.argmax(entry['distribution'][box_idx == i, 0]) # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                # drop repeat overlap TODO!!!!!!!!!!!!
                for i in range(b):
                    duplicate_class = torch.mode(entry['pred_labels'][entry['boxes'][:, 0] == i])[0]
                    present = entry['boxes'][:, 0] == i
                    if torch.sum(entry['pred_labels'][entry['boxes'][:, 0] == i] ==duplicate_class) > 0:
                        duplicate_position = entry['pred_labels'][present] == duplicate_class

                        ppp = torch.argsort(entry['distribution'][present][duplicate_position][:,duplicate_class - 1])[:-1]
                        for j in ppp:

                            changed_idx = global_idx[present][duplicate_position][j]
                            entry['distribution'][changed_idx, duplicate_class-1] = 0
                            entry['pred_labels'][changed_idx] = torch.argmax(entry['distribution'][changed_idx])+1
                            entry['pred_scores'][changed_idx] = torch.max(entry['distribution'][changed_idx])


                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx==j][entry['pred_labels'][box_idx==j] != 1]: # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(obj_features.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(obj_features.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx

                # entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat((im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                                        torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                # union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                union_feat = []
                for frame_id in range(len(entry['frame_names'])):
                    union_boxes_in_frame_i = union_boxes[union_boxes[:,0] == frame_id]
                    if len(union_boxes_in_frame_i) > 0:
                        union_feat.append(extract_feature_given_bbox_base_feat_torch(entry['faset_rcnn_model'], entry['transforms'], entry['cv2_imgs'][frame_id], union_boxes_in_frame_i[:, 1:], entry['fmaps'][frame_id], False))
                    else:
                        pass
                union_feat = torch.cat(union_feat)
                # entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(obj_features.device)
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                entry['spatial_masks'] = spatial_masks
            return entry
        else:
            # wks时候train与test都用这个
            if self.is_wks or self.training:
                obj_embed = entry['distribution'] @ self.obj_embed.weight
                pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
                obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)

                box_idx = entry['boxes'][:, 0][entry['pair_idx'].unique()]
                l = torch.sum(box_idx == torch.mode(box_idx)[0])
                b = int(box_idx[-1] + 1)  # !!!

                entry['distribution'] = self.decoder_lin(obj_features)
                entry['pred_labels'] = entry['labels']
                entry['pred_scores'] = entry['scores']
            else:
                pdb.set_trace()
                # 只在非wks且test mode下跑这部分
                obj_embed = entry['distribution'] @ self.obj_embed.weight
                pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
                obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1) #use the result from FasterRCNN directly

                box_idx = entry['boxes'][:, 0].long()
                b = int(box_idx[-1] + 1)

                # book->paper/notebook
                # chair->sofa/coach
                # food->?
                # 对数据集中bbox特别接近的情况处理
                entry = self.clean_class(entry, b, 5)
                entry = self.clean_class(entry, b, 8)
                entry = self.clean_class(entry, b, 17)

                # # NMS
                final_boxes = []
                final_dists = []
                final_feats = []
                for i in range(b):
                    # images in the batch
                    scores = entry['distribution'][entry['boxes'][:, 0] == i]
                    pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i, 1:]
                    feats = entry['features'][entry['boxes'][:, 0] == i]

                    # print(scores)
                    if scores.shape[0] != 0 and scores.shape[1] != 0:
                        for j in range(len(self.classes) - 1):
                            # NMS according to obj categories
                            inds = torch.nonzero(torch.argmax(scores, dim=1) == j).view(-1)
                            # if there is det
                            if inds.numel() > 0:
                                cls_dists = scores[inds]
                                cls_feats = feats[inds]
                                cls_scores = cls_dists[:, j]
                                _, order = torch.sort(cls_scores, 0, True)
                                cls_boxes = pred_boxes[inds]
                                cls_dists = cls_dists[order]
                                cls_feats = cls_feats[order]
                                keep = nms(cls_boxes[order, :], cls_scores[order], 0.6)  # hyperparameter

                                final_dists.append(cls_dists[keep.view(-1).long()])
                                final_boxes.append(torch.cat((torch.tensor([[i]], dtype=torch.float).repeat(keep.shape[0],
                                                                                                            1).cuda(0),
                                                            cls_boxes[order, :][keep.view(-1).long()]), 1))
                                final_feats.append(cls_feats[keep.view(-1).long()])

                entry['boxes'] = torch.cat(final_boxes, dim=0)
                box_idx = entry['boxes'][:, 0].long()
                entry['distribution'] = torch.cat(final_dists, dim=0)
                entry['features'] = torch.cat(final_feats, dim=0)

                # 似乎与object_detector的这部分重复了
                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(box_idx.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    # print(entry['distribution'][box_idx == i, 0], entry['distribution'][box_idx == i, 0].shape)
                    if entry['distribution'][box_idx == i, 0].shape[0] != 0:
                        local_human_idx = torch.argmax(entry['distribution'][
                                                       box_idx == i, 0])  # the local bbox index with highest human score in this frame
                        HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx == j][
                        entry['pred_labels'][box_idx == j] != 1]:  # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(box_idx.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(box_idx.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx
                entry['human_idx'] = HUMAN_IDX
                # entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat(
                    (im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                     torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                # entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                entry['spatial_masks'] = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(box_idx.device)

            return entry


class STTran(nn.Module):

    def __init__(self, mode='sgdet',
                 attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None, rel_classes=None,
                 enc_layer_num=None, dec_layer_num=None, transformer_mode=None, is_wks=True, feat_dim=2048, motifs_path=None):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(STTran, self).__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        self.transformer_mode = transformer_mode
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode
        self.is_wks = is_wks
        self.motifs_path = motifs_path

        self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes, is_wks=self.is_wks)

        ###################################
        # new decetor is 2048d
        self.union_func1 = nn.Conv2d(feat_dim, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        obj_dim = 192 # TODO our 192, baseline 2048
        self.subj_fc = nn.Linear(obj_dim, 512)
        self.obj_fc = nn.Linear(obj_dim, 512)
        self.vr_fc = nn.Linear(256*7*7, 512)

        # embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='/home/csq/STTran/Dokumente/neural-motifs-master/data', wv_dim=200)
        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()
        
        if self.transformer_mode == 'org':
            self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1936, nhead=8,
                                              dim_feedforward=2048, dropout=0.1, mode='latter')
        elif self.transformer_mode == 'wk':
            self.glocal_transformer = transformer_wk(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1936, nhead=8,
                                              dim_feedforward=2048, dropout=0.1, mode='latter')
        elif self.transformer_mode == 'img':
            self.glocal_transformer = transformer_img(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1936, nhead=8,
                                              dim_feedforward=2048, dropout=0.1, mode='latter')
                                
        '''
        elif self.transformer_mode == 'new':
            self.glocal_transformer = transformer_new(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1936, nhead=8,
                                              dim_feedforward=2048, dropout=0.1, mode='latter', autoregressive=False)
        elif self.transformer_mode == 'seq2seq':
            self.glocal_transformer = transformer_seq2seq(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1936, nhead=8,
                                              dim_feedforward=2048, dropout=0.1)
        '''

        self.a_rel_compress = nn.Linear(1936, self.attention_class_num)
        self.s_rel_compress = nn.Linear(1936, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(1936, self.contact_class_num)

        '''
        trans_matrix = np.load('/home/csq/project/AG/datasets/ActionGenome/dataset/ag/annotations/same_relation_martix.npz')

        self.a_trans_matrix = torch.nn.Parameter(torch.tensor(trans_matrix['att'], dtype=torch.float32, requires_grad=True))
        self.s_trans_matrix = torch.nn.Parameter(torch.tensor(trans_matrix['spa'], dtype=torch.float32, requires_grad=True))
        self.c_trans_matrix = torch.nn.Parameter(torch.tensor(trans_matrix['con'], dtype=torch.float32, requires_grad=True))
        '''

        if self.motifs_path is not None:
            pred_freq = np.load(self.motifs_path)

            self.a_pred_freq = torch.tensor(pred_freq['att_freq'], dtype=torch.float32, requires_grad=False).cuda(0)
            self.s_pred_freq = torch.tensor(pred_freq['spa_freq'], dtype=torch.float32, requires_grad=False).cuda(0)
            self.c_pred_freq = torch.tensor(pred_freq['con_freq'], dtype=torch.float32, requires_grad=False).cuda(0)

            self.a_pred_freq = torch.log_(self.a_pred_freq)
            self.s_pred_freq = torch.log_(self.s_pred_freq)
            self.c_pred_freq = torch.log_(self.c_pred_freq)


    def forward(self, entry):
        # pdb.set_trace()
        entry = self.object_classifier(entry)

        # visual part
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)
        # entry['union_feat']) 1024*7*7-->256*7*7 entry['spatial_masks'] 2*27*27 vr 256*7*7
        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1, 256*7*7))
        # subj_rep 512, obj_rep 512, vr 512
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)

        # semantic part
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        # subj_emb 200, obj_emb 200
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        # 1936
        rel_features = torch.cat((x_visual, x_semantic), dim=1)
        # Spatial-Temporal Transformer
        global_output, global_attention_weights, local_attention_weights = self.glocal_transformer(features=rel_features, im_idx=entry['im_idx'])

        if self.motifs_path is None:
            entry["attention_distribution"] = self.a_rel_compress(global_output)
            entry["spatial_distribution"] = self.s_rel_compress(global_output)
            entry["contacting_distribution"] = self.c_rel_compress(global_output)
        else:
            entry["attention_distribution"] = self.a_rel_compress(global_output) + self.a_pred_freq[entry['obj_labels']]
            entry["spatial_distribution"] = self.s_rel_compress(global_output) + self.s_pred_freq[entry['obj_labels']]
            entry["contacting_distribution"] = self.c_rel_compress(global_output) + self.c_pred_freq[entry['obj_labels']]

        '''
        attention_distribution_softmax = nn.functional.softmax(entry['attention_distribution'], dim=1)
        spatial_distribution_sigmoid = torch.sigmoid(entry["spatial_distribution"])
        contacting_distribution_sigmoid = torch.sigmoid(entry["contacting_distribution"])

        entry['att_pred'] = torch.matmul(attention_distribution_softmax, self.a_trans_matrix)
        entry['spa_pred'] = torch.matmul(spatial_distribution_sigmoid, self.s_trans_matrix)
        entry['con_pred'] = torch.matmul(contacting_distribution_sigmoid, self.c_trans_matrix)
        '''

        entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
        entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])

        return entry

