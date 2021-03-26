#   Copyright 2020 trueto

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# coding: utf-8

import json
# In[1]:

import collections


# In[2]:

Result = collections.namedtuple("Result", ["score", "detail_s", "message"])

def ner_metric(sub_file, result_file, en_list):
    with open(sub_file, 'r', encoding="utf-8") as f1, open(result_file, 'r', encoding="utf-8") as f2:
        sub_data = f1.readlines()
        res_data = f2.readlines()

    dict_sub = {}
    dict_res = {}
    row = 0
    row_line = 0

    for sub_line in sub_data:
        row += 1
        if len(sub_line.strip()) > 0:
            row_line += 1
            sub_line_l = sub_line.split("|||")
            sub_line_l.pop()
            text = sub_line_l.pop(0)
            entities = []
            for en_line in sub_line_l:
                en_list_ = en_line.split("    ")
                if len(en_list_) < 3:
                    continue
                start_pos = int(en_list_[0])
                end_pos = int(en_list_[1]) + 1
                tag = en_list_[2]
                entities.append({
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "label_type": tag
                })
            dict_sub[text] = entities

    for res_line in res_data:
        if len(res_line.strip()) > 0:
            res_line_l = res_line.split("|||")
            res_line_l.pop()
            text = res_line_l.pop(0)
            entities = []
            for en_line in res_line_l:
                en_list_ = en_line.split("    ")
                if len(en_list_) < 3:
                    continue
                start_pos = int(en_list_[0])
                end_pos = int(en_list_[1]) + 1
                tag = en_list_[2]
                entities.append({
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "label_type": tag
                })

            dict_res[text] = entities

    if len(dict_sub) != len(dict_res):
        return Result(-1, 'out of data')

    en_dict = { en: {} for en in en_list}
    en_g = { en: 0 for en in en_list}
    overall_g = 0

    for row_id in dict_res:
        if row_id not in dict_sub:
            return Result(-1, 'Incorrect ID in line: ' + str(row_id))

        t_lst = dict_res[row_id]
        for item in t_lst:
            overall_g += 1

            label_type = item["label_type"]
            en_g[label_type] += 1

            if row_id not in en_dict[label_type]:
                en_dict[label_type][row_id] = []
                en_dict[label_type][row_id].append(item)
            else:
                en_dict[label_type][row_id].append(item)

    en_s, overall_s = {en: 0 for en in en_list}, 0
    en_r, overall_r = {en: 0 for en in en_list}, 0

    predict, en_body = 0, {en: 0 for en in en_list}

    for row_id in dict_sub:
        if row_id not in dict_res:
            return Result(-1, ("unknown id:" + row_id))
        s_lst = dict_sub[row_id]
        predict += len(s_lst)
        for item in s_lst:
            label_type = item["label_type"]
            en_body[label_type] += 1

            if row_id not in en_dict[label_type]:
                continue

            if item in en_dict[label_type][row_id]:
                en_s[label_type] += 1
                overall_s += 1
                en_r[label_type] += 1
                overall_r += 1
                en_dict[label_type][row_id].remove(item)

            else:
                for gold in en_dict[label_type][row_id]:
                    if max(int(item["start_pos"]), int(gold["start_pos"])) <= min(int(item["end_pos"]),int(gold["end_pos"])):
                        en_dict[label_type][row_id].remove(gold)
                        en_r[label_type] += 1
                        overall_r += 1
                        break

    precision, recall, f1 = {}, {}, {},

    for label_type in en_list:
        if en_body[label_type] == 0:
            precision['{}_s'.format(label_type)] = 0
            precision['{}_r'.format(label_type)] = 0
        else:
            precision['{}_s'.format(label_type)] = en_s[label_type] / en_body[label_type]
            precision['{}_r'.format(label_type)] = en_r[label_type] / en_body[label_type]


    if predict == 0:
        precision['overall_s'] = 0
    else:
        precision['overall_s'] = overall_s / predict

    if predict == 0:
        precision['overall_r'] = 0
    else:
        precision['overall_r'] = overall_r / predict

    for label_type in en_list:
        recall["{}_s".format(label_type)] = en_s[label_type] / en_g[label_type]
        recall["{}_r".format(label_type)] = en_r[label_type] / en_g[label_type]

    recall['overall_s'] = overall_s / overall_g
    recall['overall_r'] = overall_r / overall_g

    for item in precision:
        f1[item] = 2 * precision[item] * recall[item] / (precision[item] + recall[item]) \
            if (precision[item] + recall[item]) != 0 else 0

    s = ""
    for label_type in en_list:
        s += "{}_s:\t{} {}_r:\t{}\n".format(label_type,
                                            [precision['{}_s'.format(label_type)], recall['{}_s'.format(label_type)],
                                             f1['{}_s'.format(label_type)]], label_type,
                                            [precision['{}_r'.format(label_type)],
                                             recall['{}_r'.format(label_type)],
                                             f1['{}_r'.format(label_type)]])
    detail_s = "recall_s: \t{} \t precision_s: \t{} \t f1_s: \t{} \n" \
               "recall_r: \t{} \t precision_r: \t {} \t f1_r: \t{} \n".format(recall['overall_s'], precision['overall_s'],
                                                              f1['overall_s'],
                                                              recall['overall_r'], precision['overall_r'],
                                                              f1['overall_r'])
    return {
        "overall_s": f1['overall_s'],
        "detial_s": detail_s,
        "message": s
    }
