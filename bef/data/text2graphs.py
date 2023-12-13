
import os
import random
import pickle
import time
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
from typing import Dict

from typing import List, Dict, Tuple

import networkx
import argparse

import bef.datautils as datautils

from DeepEventMine.bert.tokenization import BertTokenizer

from bef.datautils import chunk_dict, StandoffEntity, StandoffEvent, load_events_lines

from DeepEventMine.eval.evaluate import predict

from DeepEventMine.nets import deepEM
from DeepEventMine.loader.prepData import prepdata
from DeepEventMine.loader.prepNN import prep4nn
from DeepEventMine.utils import utils
from torch.profiler import profile, record_function, ProfilerActivity


# from memory_profiler import profile

def json2graphs(arguments, parameters, deepee_model,tokenizer):

    out_file = 'graph_' + arguments['json_file']
    result_dir = os.path.join(arguments['result_dir'], arguments['task_name'])
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print('Loading pmid2sentences from Json file')
    with open(arguments['json_file_dir']+arguments['json_file']) as json_file:
        allsentences = json.load(json_file)

    CHUNK_SIZE = 100
    NUM_CHUNKS = len(allsentences)/CHUNK_SIZE
    ##Divide sentces
    ids2ent_tri_eve = {}
    ### Loop batch
    for sentences in tqdm(chunk_dict(allsentences, CHUNK_SIZE), desc="Iteration", leave=False, total=NUM_CHUNKS):
        # print(' Processing data...')
        test_data = prepdata.prep_input_data(
            arguments['test_data'], parameters, sentences=sentences)
        # nntest_data, test_dataloader = read_test_data(test_data, parameters)

        test = prep4nn.data2network(
            test_data, 'predict', parameters, tokenizer=tokenizer)
        if len(test) == 0:
            raise ValueError("Test set empty.")
        # start_time = time.time()
        #Slow code
        nntest_data = prep4nn.torch_data_2_network(
            cdata2network=test, params=parameters, do_get_nn_data=True, tokenizer=tokenizer)
        # print("torch_data_2_network: --- %s seconds ---" % (time.time() - start_time))

        te_data_size = len(nntest_data['nn_data']['ids'])

        test_data_ids = TensorDataset(torch.arange(te_data_size))
        test_sampler = SequentialSampler(test_data_ids)
        test_dataloader = DataLoader(
            test_data_ids, sampler=test_sampler, batch_size=parameters['batchsize'])

        # with profile(activities=[
        #         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,with_stack=True) as prof:
        #     with record_function("model_inference"):

        new_ent_tri_eve = predict(model=deepee_model,
                                  result_dir=result_dir,
                                  eval_dataloader=test_dataloader,
                                  eval_data=nntest_data,
                                  g_entity_ids_=test_data['g_entity_ids_'],
                                  params=parameters,
                                  write_files=False,
                                  disable_tqdm=True,
                                  get_data=True)

        ids2ent_tri_eve.update(new_ent_tri_eve)
        # DEBUG
        # break

    ## Loop
    ##Converting DEM data to graph data and saving to JSON
    all_graphs = []

    for idd, data in ids2ent_tri_eve.items():

        entity_lines = data[0] + data[1]
        event_lines = data[2]
        entities, events = load_events_lines(entity_lines, event_lines)

        # with open(datafile.embeddings) as ff:
        #     embeddings = json.load(ff)

        # Create a graph with all the entities and events
        graph = networkx.DiGraph(
            source_doc=idd, dataset='PUBMED')
        for ent in entities.values():
            graph.add_node(ent.id, type=ent.type, name=ent.name,span=ent.span)
        for event in events.values():
            for argument, role in event.arguments:
                if type(argument) is StandoffEntity:
                    arg_id = argument.id
                else:
                    arg_id = argument.trigger.id
                graph.add_edge(event.trigger.id, arg_id,
                               key=role, event_id=event.id)

        # Find all the "root" events (not nested)
        roots = [node for node in graph.nodes if graph.in_degree(
            node) == 0 and graph.out_degree(node) > 0]
        for root in roots:
            root_event = networkx.induced_subgraph(
                graph, networkx.descendants(graph, root) | set([root])).copy()
            root_event.graph['root'] = root
            all_graphs.append(networkx.node_link_data(root_event))

    print(f'Saving {len(all_graphs)} graphs...')

    try:
        with open(os.path.join(result_dir, out_file), 'w') as ff:
            print('Save graph to', result_dir + out_file)
            json.dump(all_graphs, ff)
    except FileNotFoundError:
        with open(out_file, 'w') as ff:
            print('Save graph to', out_file)
            json.dump(all_graphs, ff)



# @profile
def main():
    # read predict config
    # set config path by command line
    inp_args = utils._parsing()
    config_path = getattr(inp_args, 'yaml')
    # config_path = '/home/julio/repos/event_finder/DeepEventMine_fork/experiments/pubmed100/configs/predict-pubmed-100.yaml'


    # set config path manually
    # config_path = 'configs/debug.yaml'

    with open(config_path, 'r') as stream:
        arguments = utils._ordered_load(stream)

    if inp_args.gpu is not None:
        arguments['gpu'] = int(inp_args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = inp_args.gpu

    if inp_args.json_file is not None:
        arguments['json_file'] = inp_args.json_file

    # Fix seed for reproducibility
    os.environ["PYTHONHASHSEED"] = str(arguments['seed'])
    random.seed(arguments['seed'])
    np.random.seed(arguments['seed'])
    torch.manual_seed(arguments['seed'])
    #https://github.com/pytorch/pytorch/issues/40134
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True

    # Load pre-trained parameters
    with open(arguments['saved_params'], "rb") as f:
        parameters = pickle.load(f)

    parameters['predict'] = True

    # Set predict settings value for params
    parameters['gpu'] = arguments['gpu']
    parameters['batchsize'] = arguments['batchsize']
    print('GPU available:' ,torch.cuda.is_available())
    if parameters['gpu'] >= 0:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        # torch.cuda.set_device(parameters['gpu'])
    else:
        device = torch.device("cpu")
    parameters['device'] = device

    # Set evaluation settings
    parameters['test_data'] = arguments['test_data']
    parameters['bert_model'] = arguments['bert_model']
    parameters['result_dir'] = arguments['result_dir']
    # raw text
    parameters['raw_text'] = arguments['raw_text']
    parameters['ner_predict_all'] = arguments['raw_text']
    parameters['a2_entities'] = arguments['a2_entities']

    
    #####
    print('Loading model')
    deepee_model = deepEM.DeepEM(parameters)

    model_path = arguments['model_path']

    # Load all models
    print('Loading checkpoints mode')
    utils.handle_checkpoints(model=deepee_model,
                             checkpoint_dir=model_path,
                             params={
                                 'device': device
                             },
                             resume=True)
    deepee_model.to(device)

    tokenizer = BertTokenizer.from_pretrained(
        parameters['bert_model'], do_lower_case=False
    )
    parameters['stats'] = False


    ######
    json2graphs(arguments, parameters, deepee_model, tokenizer)
    ##
        
if __name__ == '__main__':
    main()
