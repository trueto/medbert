import re
import os
import time
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from glob import glob

import torch
from transformers import BertTokenizer

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from ignite.contrib.handlers import ProgressBar

from model import NER_Model
from utils import cmed_ner_metric

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
import random
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf``
    (if installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class cMedNER:

    def __init__(self, dataset, max_split_len=120, max_seq_len=128,
                 model_name_or_path="bert-base-chinese",per_gpu_batch_size=8,
                 embed_size=300, no_cuda=False,
                 dense_layer_type="linear", dropout=0.5, embed_type="random",
                 vector_file="", bert_lr=1e-5, crf_lr=1e-3, patience=3,
                 output_dir="results/cmt_ner", n_saved=3, max_epochs=100):

        set_seed(42)
        self.train_path = os.path.join(dataset, "train.txt")
        self.dev_path = os.path.join(dataset, "dev.txt")
        self.test_path = os.path.join(dataset, "test.txt")

        self.max_split_len = max_split_len
        self.max_seq_len = max_seq_len

        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

        self.en_list = ["bod", "dis", "sym", "pro", "dru", "ite", "mic", "equ", "dep"]

        self.label_list = ['<pad>', '<start>', '<end>', "O"]
        for en in self.en_list:
            for pre in ["B-", "I-", "E-", "S-"]:
                self.label_list.append(pre + en)

        self.per_gpu_batch_size = per_gpu_batch_size
        self.embed_size = embed_size
        self.no_cuda = no_cuda

        self.dense_layer_type = dense_layer_type
        self.dropout = dropout
        self.embed_type = embed_type
        self.model_name_or_path = model_name_or_path
        self.vector_file = vector_file
        self.bert_lr = bert_lr
        self.crf_lr = crf_lr
        self.patience = patience
        self.output_dir = output_dir
        self.n_saved = n_saved
        self.max_epochs = max_epochs

        device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = max(torch.cuda.device_count() if not self.no_cuda else 1, 1)
        self.device = device

        if 'bert' not in self.embed_type:
            model_name = "{}_{}_crf".format(self.embed_type, self.dense_layer_type)
        else:
            embed_type = os.path.split(self.model_name_or_path)[-1]
            model_name = "{}_{}_crf".format(embed_type, self.dense_layer_type)

        self.model_name = model_name
        self.output_dir = "{}/{}".format(output_dir, model_name)

    def evaluation(self, gold_file, start_time, train_time):
        predix = os.path.split(gold_file)[-1].replace(".txt", "")
        pre_file = os.path.join(self.output_dir, '{}_{}.txt'.format(predix, self.model_name))
        score_file = os.path.join(self.output_dir, 'score_{}_{}.txt'.format(predix, self.model_name))
        with open(score_file, 'w', encoding="utf-8") as w:
            res = cmed_ner_metric(pre_file, gold_file, self.en_list)
            w.write("overall_s:\t{}".format(res['overall_s']))
            w.write("\n")
            w.write("{}".format(res['detial_s']))
            w.write("\n")
            w.write("message:\n{}".format(res['message']))
            w.write("\n")
            w.write("train time cost:\t {:.2f} s".format(train_time))
            w.write("\n")
            w.write("time cost:\t {:.2f} s".format(time.time() - start_time - train_time))
            w.write("\n")
            w.write("args:\n{}".format('\n'.join(['%s:%s' % item for item in self.__dict__.items()])))

    def export_results(self, unlabel_path):
        X, cut_his, originalTexts = self.get_X(unlabel_path)
        y_pred = self.predict(X)

        entity_data = []
        predix = os.path.split(unlabel_path)[-1].replace(".txt", "")

        X_align, y_align = originalTexts, self.alignment_X_y(originalTexts, cut_his, y_pred)

        for i, (text, y) in enumerate(tqdm(zip(X_align, y_align), desc="Decoding")):
            entities = []
            for k, label in enumerate(y):
                if "-" in label:
                    tag_1 = label.split("-")[0]
                    tag_2 = label.split("-")[1]
                    ## Single
                    if tag_1 == "S":
                        start_pos = k
                        end_pos = k + 1
                        entity = text[start_pos: end_pos]
                        en_line = "{}    {}    {}".format(start_pos, end_pos-1, tag_2)
                        entities.append(en_line)
                        entity_data.append((i + 1, entity, tag_2, start_pos, end_pos))

                    if tag_1 == "B":
                        start_pos = k
                        end_pos = k + 1
                        for j in range(start_pos + 1, len(y)):
                            if y[j] == "I-" + tag_2:
                                end_pos += 1
                            elif y[j] == 'E-' + tag_2:
                                end_pos += 1
                                break
                            else:
                                break
                        entity = text[start_pos: end_pos]
                        en_line = "{}    {}    {}".format(start_pos, end_pos - 1, tag_2)
                        entities.append(en_line)
                        entity_data.append((i + 1, entity, tag_2, start_pos, end_pos))

            with open(os.path.join(self.output_dir, '{}_{}.txt'.format(predix, self.model_name)), 'a', encoding="utf-8") as f:
                entity_text = "|||".join(entities)
                s = "{}|||{}|||".format(text, entity_text)
                f.write(s)
                f.write("\n")

        tempDF = pd.DataFrame(data=entity_data, columns=['text_id', 'entity', 'label_type', 'start_pos', 'end_pos'])
        tempDF.to_csv(os.path.join(self.output_dir, "tmp_entities_{}_{}.csv".format(predix, self.model_name)), index=False)

    def alignment_X_y(self, originalTexts, cut_his, y_pred):
        y_align = []
        for i, X in enumerate(originalTexts):
            cut_index = cut_his[i]
            if isinstance(cut_index, int):
                y_ = y_pred[cut_index]
            else:
                y_ =[]
                for index in cut_index:
                    y_.extend(y_pred[index])

            assert len(X) == len(y_), 'i:{};text_len:{};while label_len:{}'.format(i, len(X), len(y_))
            y_align.append(y_)

        assert len(originalTexts) == len(y_align)
        return y_align

    def train(self):
        ## train data
        train_X, train_y, _ = self.get_X_y(self.train_path)
        train_input_ids, train_input_mask_ids, train_label_ids, train_label_mask_ids = self.get_X_y_ids(train_X, train_y)

        ## dev data
        dev_X, dev_y, _ = self.get_X_y(self.dev_path)
        dev_input_ids, dev_input_mask_ids, dev_label_ids, dev_label_mask_ids = self.get_X_y_ids(dev_X, dev_y)

        train_ds = TensorDataset(train_input_ids, train_input_mask_ids, train_label_ids, train_label_mask_ids)
        dev_ds = TensorDataset(dev_input_ids, dev_input_mask_ids, dev_label_ids, dev_label_mask_ids)

        batch_size = self.n_gpu * self.per_gpu_batch_size
        train_iter = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        dev_iter = DataLoader(dev_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        model = NER_Model(vocab_size=self.bert_tokenizer.vocab_size, embed_size=self.embed_size,
                          num_tags=len(self.label_list), max_len=self.max_seq_len, device=self.device,
                          dense_layer_type=self.dense_layer_type, dropout=self.dropout, embed_type=self.embed_type,
                          model_name_or_path=self.model_name_or_path, vector_file=self.vector_file)

        model.to(self.device)

        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info("model.named_parameters()")
        for n, p in model.named_parameters():
            logger.info(n)

        parameters = [{
            "params": [p for n, p in model.named_parameters() if "bert" in n],
            "lr": self.bert_lr
        }, {
            "params": [p for n, p in model.named_parameters() if "bert" not in n],
            "lr": self.crf_lr
        }]

        optimizer = torch.optim.AdamW(parameters, lr=self.crf_lr)

        tb_writer = SummaryWriter()

        def train_fn(engine, batch):
            model.train()
            optimizer.zero_grad()
            batch = tuple(t.to(self.device) for t in batch)
            labels = batch[2]

            inputs = {
                "token_ids": batch[0],
                "input_masks": batch[1],
                "label_ids": labels,
            }

            loss, sequence_tags = model(**inputs)

            score = (sequence_tags == labels).float().detach().cpu().numpy()

            condition_1 = (labels != self.label_list.index("O")).detach().cpu().numpy()
            condition_2 = (labels != self.label_list.index("<pad>")).detach().cpu().numpy()
            patten = np.logical_and(condition_1, condition_2)
            score = score[patten].mean()

            if self.n_gpu > 1:
                loss = loss.mean()

            ## tensorboard
            global_step = global_step_from_engine(engine)(engine, engine.last_event_name)
            # tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('train_loss', loss.item(), global_step)
            tb_writer.add_scalar('train_score', score.item(), global_step)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            return loss.item(), score

        trainer = Engine(train_fn)
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'score')

        def dev_fn(engine, batch):
            model.eval()
            optimizer.zero_grad()
            with torch.no_grad():
                batch = tuple(t.to(self.device) for t in batch)
                labels = batch[2]

                inputs = {
                    "token_ids": batch[0],
                    "input_masks": batch[1],
                    "label_ids": labels,
                }

                loss, sequence_tags = model(**inputs)

            score = (sequence_tags == labels).float().detach().cpu().numpy()

            condition_1 = (labels != self.label_list.index("O")).detach().cpu().numpy()
            condition_2 = (labels != self.label_list.index("<pad>")).detach().cpu().numpy()
            patten = np.logical_and(condition_1, condition_2)
            score = score[patten].mean()

            if self.n_gpu > 1:
                loss = loss.mean()

            ## tensorboard
            global_step = global_step_from_engine(engine)(engine, engine.last_event_name)
            # tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('dev_loss', loss.item(), global_step)
            tb_writer.add_scalar('dev_score', score.item(), global_step)

            return loss.item(), score

        dev_evaluator = Engine(dev_fn)
        RunningAverage(output_transform=lambda x: x[0]).attach(dev_evaluator, 'loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(dev_evaluator, 'score')

        pbar = ProgressBar(persist=True, bar_format="")
        pbar.attach(trainer, ['loss', 'score'])
        pbar.attach(dev_evaluator, ['loss', 'score'])

        def score_fn(engine):
            loss = engine.state.metrics['loss']
            score = engine.state.metrics['score']
            '''
            if score < 0.5:
                logger.info("Too low to learn!")
                trainer.terminate()
            '''

            return score / (loss + 1e-12)

        handler = EarlyStopping(patience=self.patience, score_function=score_fn, trainer=trainer)
        dev_evaluator.add_event_handler(Events.COMPLETED, handler)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_dev_results(engine):
            dev_evaluator.run(dev_iter)
            dev_metrics = dev_evaluator.state.metrics
            avg_score = dev_metrics['score']
            avg_loss = dev_metrics['loss']
            logger.info(
                "Validation Results - Epoch: {}  Avg score: {:.2f} Avg loss: {:.2f}"
                    .format(engine.state.epoch, avg_score, avg_loss))

        def model_score(engine):
            score = engine.state.metrics['score']
            return score


        checkpointer = ModelCheckpoint(self.output_dir, "cmed_ner", n_saved=self.n_saved,
                                       create_dir=True, score_name="model_score",
                                       score_function=model_score,
                                       global_step_transform=global_step_from_engine(trainer),
                                       require_empty=False)

        dev_evaluator.add_event_handler(Events.COMPLETED, checkpointer,
                                        {self.model_name: model.module if hasattr(model, 'module') else model})

        # Clear cuda cache between training/testing
        def empty_cuda_cache(engine):
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
        dev_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

        trainer.run(train_iter, max_epochs=self.max_epochs)

    def predict(self, X):
        all_input_ids, all_input_mask_ids, all_label_ids, all_label_mask_ids = self.get_X_y_ids(X)
        dataset = TensorDataset(all_input_ids, all_input_mask_ids, all_label_ids)

        batch_size = self.n_gpu * self.per_gpu_batch_size
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        model = NER_Model(vocab_size=self.bert_tokenizer.vocab_size, embed_size=self.embed_size,
                  num_tags=len(self.label_list), max_len=self.max_seq_len, device=self.device,
                  dense_layer_type=self.dense_layer_type, dropout=self.dropout, embed_type=self.embed_type,
                  model_name_or_path=self.model_name_or_path, vector_file=self.vector_file)

        model.to(self.device)

        y_preds = []

        for model_state_path in glob(os.path.join(self.output_dir, '*{}*.pt*'.format(self.model_name))):
            model.load_state_dict(torch.load(model_state_path))
            y_pred = self.single_predict(model, dataloader)
            y_preds.append(y_pred)

        y_preds = torch.tensor(y_preds)
        y_pred = torch.mode(y_preds, dim=0).values
        y_pred = y_pred.numpy()

        preds_list = [[] for _ in range(all_label_mask_ids.shape[0])]

        for i in range(all_label_mask_ids.shape[0]):
            for j in range(all_label_mask_ids.shape[1]):
                if all_label_mask_ids[i, j] != -100:
                    preds_list[i].append(self.label_list[y_pred[i][j]])
        return preds_list

    def single_predict(self, model, dataloader):
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        model.eval()
        preds = None
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    "token_ids": batch[0],
                    "input_masks": batch[1],
                }
                _, sequence_tags = model(**inputs)

                sequence_tags = sequence_tags.detach().cpu().numpy()
                if preds is None:
                    preds = sequence_tags
                else:
                    preds = np.append(preds, sequence_tags, axis=0)
        return preds

    def get_X_y_ids(self, X, y=None):
        all_input_ids = []
        all_label_ids = []
        all_input_mask_ids = []
        all_label_mask_ids = []
        for i, X_ in enumerate(tqdm(X, desc="Tokens to ids")):
            text = list(map(str.lower, X_))
            input_ids = self.bert_tokenizer.encode(text=text)
            input_mask_ids = [1] * len(input_ids)
            padding_len = self.max_seq_len - len(input_ids)
            input_ids += [self.bert_tokenizer.pad_token_id] * padding_len
            input_mask_ids += [0] * padding_len

            try:
                y_ = ['<start>'] + y[i] + ['<end>']
                y_ += ['<pad>'] * padding_len
                label_mask_id = [-100] + [100] * len(y[i]) + [-100]
                label_mask_id += [-100] * padding_len
            except:
                y_ = ['<start>', '<end>'] + ['<pad>'] * (self.max_seq_len - 2)
                label_mask_id = [-100 if idx in [
                    self.bert_tokenizer.pad_token_id,
                    self.bert_tokenizer.cls_token_id,
                    self.bert_tokenizer.sep_token_id] else 100 for idx in input_ids]

            label_ids = list(map(self.label_list.index, y_))

            assert len(input_ids) == len(input_mask_ids) == len(label_ids) == len(label_mask_id) == self.max_seq_len
            all_input_ids.append(input_ids)
            all_input_mask_ids.append(input_mask_ids)
            all_label_ids.append(label_ids)
            all_label_mask_ids.append(label_mask_id)

            if i == 0:
                logger.info("tokens:\n{}".format(text))
                logger.info("token_ids: \n{}".format(input_ids))
                logger.info("labels:\n{}".format(y_))
                logger.info("label_ids: \n{}".format(label_ids))

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask_ids = torch.tensor(all_input_mask_ids, dtype=torch.long)
        all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)
        all_label_mask_ids = torch.tensor(all_label_mask_ids, dtype=torch.long)
        return all_input_ids, all_input_mask_ids, all_label_ids, all_label_mask_ids

    def get_X_y(self, file_path):
        X = []
        y = []
        flag = 0
        entity_data = []
        with open(file_path, 'r', encoding="utf-8") as reader:
            for i, line in enumerate(tqdm(reader.readlines(), desc="Read {}".format(file_path))):
                line_list = line.split("|||")
                line_list.pop()
                originalText = line_list.pop(0)
                text = self.clean_text(originalText)
                entities = line_list
                if len(text) < self.max_split_len:
                    X_ = list(text)
                    y_ = ['O'] * len(X_)
                    for entity in entities:
                        en_list = entity.split("    ")
                        start_pos = int(en_list[0])
                        end_pos = int(en_list[1]) + 1
                        tag = en_list[2]

                        if end_pos - start_pos > 1:
                            y_[start_pos] = 'B-' + tag
                            for i in range(start_pos+1, end_pos-1):
                                y_[i] = 'I-' + tag
                            y_[end_pos - 1] = 'E-' + tag
                        else:
                            y_[start_pos] = 'S-' + tag

                        entity_data.append((text[start_pos: end_pos], tag))
                    X.append(X_)
                    y.append(y_)

                else:
                    # split text
                    dot_index_list = self.get_dot_index(text)

                    X_list, y_list, entity_data_ = self.get_short_text_label(text, dot_index_list, entities)

                    assert len(text) == sum(map(len, X_list))

                    if flag < 3:
                        logger.info("full text:\n{}".format(text))
                        X_list_str = list(map("".join, X_list))
                        logger.info("short texts:\n{}".format("\n".join(X_list_str)))
                        flag += 1

                    X.extend(X_list)
                    y.extend(y_list)
                    entity_data.extend(entity_data_)

        vocab_df = pd.DataFrame(data=entity_data, columns=['entity', 'label_type'])
        vocab_df.drop_duplicates(inplace=True, ignore_index=True)
        assert len(X) == len(y)
        return X, y, vocab_df

    def get_X(self, unlabeled_file):
        X = []
        cut_his = {}
        originalTexts = []
        print_flag = 0
        with open(unlabeled_file, 'r', encoding='utf-8') as f:
            for text_id, line in enumerate(tqdm(f.readlines(), desc="Reading {}".format(unlabeled_file))):
                line_list = line.split("|||")
                originalText = line_list.pop(0)
                originalTexts.append(originalText)

                text = self.clean_text(originalText)
                if len(text) < self.max_split_len:
                    X.append(list(text))
                    cut_his[text_id] = len(X) - 1
                else:
                    # split text
                    dot_index_list = self.get_dot_index(text)
                    flag = 0
                    text_id_list = []
                    if print_flag < 3:
                        logger.info("full text:\n{}".format(text))

                    for i, do_index in enumerate(dot_index_list):
                        short_text = text[flag: do_index + 1]
                        if print_flag < 3:
                            logger.info("short texts:\n{}".format(short_text))
                        # print("Short text:{}".format(short_text))
                        X_ = list(short_text)
                        X.append(X_)
                        text_id_list.append(len(X) - 1)
                        flag = do_index + 1

                    print_flag += 1

                    cut_his[text_id] = text_id_list
        return X, cut_his, originalTexts

    def get_short_text_label(self, text, dot_index_list, entities):
        X = []
        y = []
        flag = 0
        entity_data = []
        for i, dot_index in enumerate(dot_index_list):
            short_text = text[flag : dot_index+1]
            X_ = list(short_text)
            y_ = ["O"] * len(X_)

            for entity in entities:
                en_list = entity.split("    ")
                start_pos = int(en_list[0])
                end_pos = int(en_list[1]) + 1
                tag = en_list[2]

                k = start_pos - flag
                en_list = []
                if end_pos - start_pos > 1:
                    if k >= 0 and k < len(y_):
                        y_[k] = 'B-' + tag
                        en_list.append(X_[k])

                    for j in range(start_pos + 1, end_pos - 1):
                        j = j - flag
                        if j >= 0 and j < len(y_):
                            y_[j] = 'I-' + tag
                            en_list.append(X_[j])
                    e = end_pos - 1 - flag
                    if e >= 0 and e < len(y_):
                        y_[e] = 'E-' + tag
                        en_list.append(X_[e])
                else:
                    if k >= 0 and k < len(y_):
                        y_[k] = 'S-' + tag
                        en_list.append(X_[k])

                if len(en_list) > 0:
                    entity_data.append(("".join(en_list), tag))

            flag = dot_index + 1

            X.append(X_)
            y.append(y_)
        return X, y, entity_data

    def get_dot_index(self, text):
        flag = 0
        text_ = text
        dot_index_list = []

        while (len(text_) > self.max_split_len):
            text_ = text_[:self.max_split_len]
            index_list = []
            for match in re.finditer("[,|，|;|；|。|、]", text_):
                index_list.append(match.span()[0])

            index_list.sort()
            if len(index_list) > 1:
                last_dot = index_list.pop()
            else:
                last_dot = len(text_)
            dot_index_list.append(last_dot + flag)
            text_ = text[(last_dot + flag) :]
            flag += last_dot

        dot_index_list.append(len(text))
        return dot_index_list

    def clean_text(self, text):
        def special2n(string):
            string = string.replace(r"\n", "")
            return re.sub("[ |\t|\r|\n|\\\|\u0004]", "_", string)

        def strQ2B(ustr):
            "全角转半角"
            rstr = ""
            for uchar in ustr:
                inside_code = ord(uchar)
                # 全角空格直接转换
                if inside_code == 12288:
                    inside_code = 32
                # 全角字符（除空格）根据关系转化
                elif (inside_code >= 65281 and inside_code <= 65374):
                    inside_code -= 65248

                rstr += chr(inside_code)
            return rstr

        return strQ2B(special2n(text)).lower()

    def explore_dataset(self, file_path):
        text_list = []
        entity_list = []
        with open(file_path, 'r', encoding="utf8") as reader:
            for i, line in enumerate(tqdm(reader.readlines(), desc="Read {}".format(file_path))):
                line_list = line.split("|||")
                line_list.pop()
                text = line_list.pop(0)
                text_list.append(text)
                for en_l in line_list:
                    en_l = en_l.split("    ")
                    try:
                        start_pos = int(en_l[0])
                    except ValueError:
                        print(i+1, line)
                    end_pos = int(en_l[1]) + 1
                    label_type = en_l[2]
                    entity_list.append((label_type, text[start_pos: end_pos]))

        text_df = pd.DataFrame(data=text_list, columns=['text'])
        entity_df = pd.DataFrame(data=entity_list, columns=['label_type', 'entity'])
        desc_path = file_path.replace("txt", "desc")

        label_type_desc = entity_df['label_type'].value_counts()
        text_lens = text_df['text'].str.len().describe()
        with open(desc_path, 'w', encoding="utf-8") as writer:
            writer.write("label_type distribution:\n{}".format(label_type_desc))
            writer.write("\n\n")
            writer.write("text len distribution:\n{}".format(text_lens))
        logger.info("label_type distribution:\n{}".format(label_type_desc))
        logger.info("text len distribution:\n{}".format(text_lens))

def clean_cache():
    torch.cuda.empty_cache()
    import gc
    gc.collect()

if __name__ == '__main__':
    import time
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    start_time = time.time()
    cmed_ner = cMedNER(dataset="datasets/CMTNER", embed_type="bert",
                       dense_layer_type="gru",
                       model_name_or_path="bert-base-chinese",
                       bert_lr=5e-5, crf_lr=1e-3, output_dir="results/cmed_ner",
                       per_gpu_batch_size=32, patience=5, max_epochs=200)

    cmed_ner.train()
    train_time = time.time() - start_time
    cmed_ner.export_results(cmed_ner.dev_path)
    cmed_ner.evaluation(cmed_ner.dev_path, start_time, train_time)

    train_time = time.time() - start_time
    cmed_ner.export_results(cmed_ner.test_path)
    cmed_ner.evaluation(cmed_ner.test_path, start_time, train_time)

    clean_cache()



