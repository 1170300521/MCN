from tensorflow import keras
import tensorflow as tf
from model.mcn_model import yolo_eval_v2
import numpy as np
from utils.utils import get_random_data
from utils.tensorboard_logging import *
import cv2
import tensorflow.keras.backend as K
from matplotlib.pyplot import cm
import spacy
import  progressbar


class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        data,
        anchors,
        config,
        tensorboard=None,
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator        : The generator that represents the dataset to evaluate.
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.val_data       = data
        self.tensorboard     = tensorboard
        self.verbose         = verbose
        self.vis_id=[i for i in np.random.randint(0, len(data), 200)]
        self.batch_size = max(config['batch_size']//2,1)
        self.colors = np.array(cm.hsv(np.linspace(0, 1, 10)).tolist()) * 255
        self.input_shape = (config['input_size'], config['input_size'])  # multiple of 32, hw
        self.config=config
        self.word_embed=spacy.load(config['word_embed'])
        self.word_len = config['word_len']
        self.anchors=anchors
        # mAP setting
        self.det_acc_thresh = config['det_acc_thresh']
        if self.tensorboard is not  None:
            self.log_images=config['log_images']
        else:
            self.log_images=0
        self.input_image_shape = K.placeholder(shape=(2,))
#        self.sess = tf.compat.v1.keras.backend.get_session()
        self.sess = tf.compat.v1.Session()
        self.eval_save_images_id = [i for i in np.random.randint(0, len(self.val_data), 200)]
        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs={}

        # run evaluation
        self.det_acc = self.evaluate(is_save_images=self.log_images)
        if self.tensorboard is not None:
            with self.tensorboard.as_default():
                tf.summary.scalar("det_acc", self.det_acc, step=epoch)
                self.tensorboard.flush()
        logs['det_acc'] = self.det_acc

        if self.verbose == 1:
            print('det_acc: {:.4f}'.format(self.det_acc))

    def evaluate(self, tag='image', is_save_images=False):
        # print(self.model.output_shape)
        self.boxes, self.scores, self.eval_inputs = yolo_eval_v2(self.model.output_shape[0],self.anchors, self.input_image_shape,
                                                                               score_threshold=0., iou_threshold=0.)
        # Add the class predict temp dict
        # pred_tmp = []
        groud_truth = []  # wait
        seg_prec_all = dict()
        id =0
        seg_iou_all =0.
        detect_prec_all = 0.
        fd_ts_count=0.
        td_fs_count=0.
        fd_fs_count=0.
        # Predict!!!
        test_batch_size =self.batch_size
        for start in progressbar.progressbar(range(0, len(self.val_data), test_batch_size), prefix='evaluation: '):
            end = start +test_batch_size
            batch_data = self.val_data[start:end]
            images = []
            images_org = []
            files_id = []
            word_vecs = []
            sentences = []
            gt_boxes = []
            att_data = []
#            gt_segs = []

            for data in batch_data:
                image_data, box, word_vec, image, sentence, att_map = get_random_data(data, self.input_shape,
                                                                                      self.word_embed, self.config,
                                                                                      train_mode=False)  # box is [1,5]
                sentences.extend(sentence)
                word_vecs.extend(word_vec)
                # evaluate each sentence corresponding to the same image
                for ___ in range(len(sentence)):
                    # groud_truth.append(box[0, 0:4])
                    gt_boxes.append(box[0, 0:4])
                    images.append(image_data)
                    images_org.append(image)
                    files_id.append(id)
                    att_data.append(att_map)
 #                   gt_segs.append(seg_map)
                    id += 1

            images = np.array(images)
            word_vecs = np.array(word_vecs)
            att_data = np.array(att_data)
            out_bboxes_1, _ = self.model.predict_on_batch([images, word_vecs])
#            pred_segs = self.sigmoid_(pred_segs)  # logit to sigmoid
            for i, out in enumerate(out_bboxes_1):
                # Predict
                out_boxes, out_scores = self.sess.run(  # out_boxes is [1,4]  out_scores is [1,1]
                    [self.boxes, self.scores],
                    feed_dict={
                        # self.eval_inputs: out
                        self.eval_inputs[0]: np.expand_dims(out, 0),
                        self.input_image_shape: np.array(self.input_shape),
                        K.learning_phase(): 0
                    })
#
#                ih = gt_segs[i].shape[0]
#                iw = gt_segs[i].shape[1]
#                w, h = self.input_shape
#                scale = min(w / iw, h / ih)
#                nw = int(iw * scale)
#                nh = int(ih * scale)
#                dx = (w - nw) // 2
#                dy = (h - nh) // 2

                # up sample
#                pred_seg = cv2.resize(pred_segs[i], self.input_shape)
#                #nls
#                if self.use_nls:
#                    pred_seg = self.nls(pred_seg, self.box_value_fix(out_boxes[0],self.input_shape), out_scores[0])
#                #scale to the size of ground-truth
#                pred_seg = pred_seg[dy:nh + dy, dx:nw + dx, ...]
#                pred_seg = cv2.resize(pred_seg, (gt_segs[i].shape[1], gt_segs[i].shape[0]))
#                pred_seg = np.reshape(pred_seg, [pred_seg.shape[0], pred_seg.shape[1], 1])
#                # segmentation eval
#                seg_iou, seg_prec = self.cal_seg_iou(gt_segs[i], pred_seg, self.seg_min_overlap)
#                seg_iou_all += seg_iou
#                for item in seg_prec:
#                    if seg_prec_all.get(item):
#                        seg_prec_all[item] += seg_prec[item]
#                    else:
#                        seg_prec_all[item] = seg_prec[item]
                # detection eval
                pred_box = self.box_value_fix(out_boxes[0], self.input_shape)
                score = out_scores[0]
                detect_prec = self.cal_detect_iou(pred_box, gt_boxes[i], self.det_acc_thresh)
                detect_prec_all += detect_prec

                # caulate IE metric
#                if detect_prec - seg_prec[0.5] != 0.:
#                    if detect_prec > seg_prec[0.5]:
#                        td_fs_count += 1.
#                    else:
#                        fd_ts_count += 1.
#                elif detect_prec + seg_prec[0.5] == 0.:
#                    fd_fs_count += 1.
#
                #visualization
                if is_save_images and (files_id[i] in self.eval_save_images_id):
                    left, top, right, bottom = pred_box
                    # Draw image
                    gt_left, gt_top, gt_right, gt_bottom = (gt_boxes[i]).astype('int32')
                    image = np.array(images[i] * 255.).astype(np.uint8)
                    # segement image for saving
#                    seg_image = np.array(
#                        cv2.resize(np.array(pred_segs[i] > self.seg_min_overlap).astype(np.float32),
#                                   self.input_shape)).astype(
#                        np.uint8) * 255
                    label = '{:%.2f}' % score
                    color = self.colors[0]
                    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(image, (gt_left, gt_top), (gt_right, gt_bottom), self.colors[1], 2)

                    font_size = 0.8

                    cv2.putText(image,
                                label,
                                (left, max(top - 3, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_size, color, 2)
                    cv2.putText(image,
                                'ground_truth',
                                (gt_left, max(gt_top - 3, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_size, self.colors[1], 2)
                    cv2.putText(image,
                                str(sentences[i]),
                                (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                .9, self.colors[2], 2)
                    cv2.imwrite('./images/'+str(files_id[i])+'.jpg',image)
                    log_images(self.tensorboard, tag + '/' + str(files_id[i]), [image], 0)
#                    log_images(self.tensorboard, tag + '/' + str(files_id[i]) + '_seg', [seg_image], 0)


        # miou_seg = seg_iou_all / id
        miou_detect = detect_prec_all / id
        # ie_score=(td_fs_count+fd_ts_count) / id
#        for item in seg_prec_all:
#            seg_prec_all[item] /= id
        return miou_detect

    def cal_detect_iou(self,box1,box2,thresh=0.5):
        smooth=1e-7
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max((yi2 - yi1),0.)* max((xi2 - xi1),0.)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        iou = (inter_area+smooth) / (union_area+smooth)
        return  float(iou>thresh)
    def cal_seg_iou(self,gt,pred,thresh=0.5):
        t=np.array(pred>thresh)
        p=gt>0.
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)

        prec=dict()
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            prec[thresh]= float(iou > thresh)
        return iou,prec
    def sigmoid_(self,x):
        return 1. / (1. + np.exp(-x))
    def box_value_fix(self,box,shape):
        '''
        fix box to avoid numeric overflow
        :param box:
        :param shape:
        :return:
        '''
        top, left, bottom, right = box
        new_w, new_h = shape
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(new_h, np.floor(bottom + 0.5).astype('int32'))
        right = min(new_w, np.floor(right + 0.5).astype('int32'))
        box=np.array([left, top, right, bottom]).astype('int32')
        return box
