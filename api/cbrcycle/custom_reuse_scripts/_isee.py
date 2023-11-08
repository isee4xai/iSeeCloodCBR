# Reuse functions
import requests
import config as cfg
import uuid
import numpy as np
import edist.sed as sed
import copy

'''
functions for transformational adaptation
'''


def transform_adapt(input=None):
    """
    Reuse operation that requires certain attributes to be present in the cases.
    Each case requires an 'UserIntent' string and 'UserQuestion' array.

    @param input should include the query case (dict) that needs a solution (query_case) and
    an ordered list of cases (list of dict) from which a solution is constructed (neighbours).
    "acceptance_threshold" which determines cutoff for average similarity is an optional parameter that is set to 0.8 by default. 
    """
    if input is None:
        return {}

    query_case = input.get("query_case")
    neighbours = input.get("neighbours")

    acceptance_threshold = input.get(
        "acceptance_threshold", 0.8)  # default as 0.8

    if query_case is None or neighbours is None:
        return {}

    query_questions = []
    for index, value in enumerate(query_case['UserQuestion']):
        obj = {'id': str(index), 'k': -1,
               'intent': query_case['UserIntent'], 'question': value}
        query_questions.append(obj)
    pairings, score, neighbours_considered, intent_overlap, case_questions = MATCH(
        query_questions, neighbours, 1, acceptance_threshold)
    # get details of pairings

    res = []
    for k, v in pairings.items():
        pair_obj = {}
        query_side = get_from_id(k, query_questions)
        if v is not None:
            case_side = get_from_id(v, case_questions)
            pair_obj['query'] = query_side
            pair_obj['case'] = case_side
            res.append(pair_obj)
    pairings = res
    adapted_solution = adapt_solution(pairings, neighbours)

    return {
        "pairings": pairings,
        "score": score,
        "neighbours_considered": neighbours_considered,
        "intent_overlap": intent_overlap,
        "adapted_solution": adapted_solution
    }


def getSimilaritySemanticSBERT(text1, text2):
    """
    Calls an external service to get the similarity between a pair of texts.
    """
    url = cfg.sbert_similarity
    res = requests.post(url, json={
                        'text1': text1, 'text2': text2, 'access_key': cfg.vectoriser_access_key})
    res_dictionary = res.json()
    return res_dictionary['similarity']


def get_from_id(id, lst):
    """
    Returns the first list entry with id that matches 'id'.
    """
    for k in lst:
        if k['id'] == id:
            return k
    return None


def get_from_id_key(_id, lst, key):
    for k in [l for l in lst if type(l) is dict]:
        if key in k and int(k[key]['id']) == _id:
            return k
    return None


def get_index_in_list(pref_list, entry_id):
    # find index of entry_id in pref_list
    for index, val in enumerate(pref_list):
        if val['id'] == entry_id:
            return index
    return None


def get_cumulative_case_questions(nearest_neighbours, idx):
    """
    Returns all the list entries up to index 'idx'. Returned items are transformed into a uniform format.
    """
    lst = []
    for i in range(idx):
        case_id = i
        if 'id' in nearest_neighbours[i]:  # use case_id if it's available
            case_id = nearest_neighbours[i]['id']
        for index, value in enumerate(nearest_neighbours[i]['UserQuestion']):
            obj = {'id': str(case_id)+'_'+str(index), 'k': i,
                   'intent': nearest_neighbours[i]['UserIntent'], 'question': value}
            lst.append(obj)
    return lst


def get_intent_overlap(query_list, nn_list, pairings):
    """
    Measures intent overlap
    """
    count = 0
    match = 0
    for k, v in pairings.items():
        count += 1
        query_intent = get_from_id(k, query_list)['intent']
        case_intent = None
        if v is not None:
            case_intent = get_from_id(v, nn_list)['intent']
        if query_intent == case_intent:
            match += 1
    if count == 0:
        return 0.0
    return float(match)/count


def get_nodes(current_node, nodes, node_list):
    node_list.append(current_node)
    # if a composite node with children
    children = current_node['firstChild'] if 'firstChild' in current_node else None
    while children:
        # find node
        child_id = children['Id']
        temp_node = [nodes[nkey]
                     for nkey in list(nodes.keys()) if nkey == child_id][0]
        get_nodes(temp_node, nodes, node_list)
        children = children['Next'] if 'Next' in children else None
    return node_list


def question_match(q, q_list):
    # multiple questions split by ;
    if ";" in q_list:
        return q in q_list.split(";")
    # single question
    else:
        return q == q_list


def empty_solution(c):
    empty_solution = c.copy()
    # leave only one tree in tree list
    empty_solution['trees'] = [t for t in empty_solution['trees']
                               if t['id'] == empty_solution['selectedTree']]
    selected_tree = empty_solution['trees'][0]
    # new tree id
    tree_id = str(uuid.uuid4())
    selected_tree['id'] = tree_id
    empty_solution['selectedTree'] = tree_id
    # new root node id
    root_id = str(uuid.uuid4())
    current_root_id = selected_tree['root']
    root_node = [selected_tree['nodes'][nkey]
                 for nkey in list(selected_tree['nodes'].keys()) if (nkey == current_root_id)][0]
    # remove children of the root Sequence Node
    temp_child = {
        "Id": "",
        "Next": None
    }
    root_node['id'] = root_id
    selected_tree['root'] = root_id
    root_node['firstChild'] = temp_child
    # remove all nodes except the root sequence node
    empty_solution['trees'][0]['nodes'] = {root_id: root_node}
    return empty_solution


def generate_next(template, ids, idx):
    template['Id'] = ids[idx]
    template['Next'] = None if idx + \
        1 >= len(ids) else generate_next(template.copy(), ids, idx+1)
    return template


def generate_solution(c, sub_trees):
    # if no solutions found
    if not sub_trees:
        return c
    # otherwise
    c_nodes = c['trees'][0]['nodes']
    c_root_node = c['trees'][0]['nodes'][c['trees'][0]['root']]
    temp_child = c_root_node['firstChild'].copy()
    roots = []
    # add nodes in each sub-tree to the empty solution
    for sub_tree in sub_trees:
        root_node_id = sub_tree[1]
        nodes = sub_tree[0]
        for node in nodes:
            c_nodes[node['id']] = node
        roots.append(root_node_id)
    c['trees'][0]['nodes'] = c_nodes
    # on the root, recursively create the references to top level children
    idx = 0
    first_child = generate_next(temp_child.copy(), roots, idx)
    c_root_node['firstChild'] = first_child
    c['trees'][0]['nodes'][c['trees'][0]['root']] = c_root_node
    # updated solution tree
    return c


def replace_references_in_dict(_dict, p_id, n_id):
    for _key in _dict:
        if type(_dict[_key]) is dict:
            _dict[_key] = replace_references_in_dict(_dict[_key], p_id, n_id)
        # dont replace if its 'id'
        if _dict[_key] == p_id and _key != 'id':
            _dict[_key] = n_id
    return _dict


def replace_references_in_list(_list, p_id, n_id):
    new_list = []
    for node in _list:
        new_node = replace_references_in_dict(node, p_id, n_id)
        new_list.append(new_node)
    return new_list


def clean_uuid(nodes, root_id):
    new_nodes = []
    for node in nodes:
        previous_id = node['id']
        new_id = str(uuid.uuid4())
        if previous_id == root_id:
            # update root id
            root_id = new_id
        nodes = replace_references_in_list(nodes, previous_id, new_id)
        node['id'] = new_id
        new_nodes.append(node)
    return new_nodes, root_id


def adapt_solution(pairs, neighbours):
    sub_trees = []
    for idx in range(len(pairs)):
        matched_pair = get_from_id_key(idx, pairs, 'query')
        q_q = matched_pair['query']['question']
        c_q = matched_pair['case']['question']
        c_idx = matched_pair['case']['k']
        c_solution = neighbours[c_idx]['Solution']
        # solution tree
        c_sol_tree = [t for t in c_solution['trees']
                      if t['id'] == c_solution['selectedTree']][0]
        # all sequence nodes
        c_sol_subs = [c_sol_tree['nodes'][nkey] for nkey in list(
            c_sol_tree['nodes'].keys()) if c_sol_tree['nodes'][nkey]['Concept'] == 'Sequence']
        # sequence nodes where the first child is a User Question Node and the question text matches the case question
        for a_sub in c_sol_subs:
            first_child = a_sub['firstChild']['Id']
            first_child_type = [c_sol_tree['nodes'][nkey] for nkey in list(c_sol_tree['nodes'].keys()) if (nkey == first_child
                                                                                                           and c_sol_tree['nodes'][nkey]['Concept'] == 'User Question'
                                                                                                           and question_match(c_q, c_sol_tree['nodes'][nkey]['params']['Question']['value']))]
            # first node is a User Question Node and question is c_q
            # also there are only two children in the sub tree: User Question and Explanation Strategy
            if first_child_type and not a_sub['firstChild']['Next']['Next']:
                # get the explainer strategy sibling
                es_sub_id = a_sub['firstChild']['Next']['Id']
                es_sub_node = [c_sol_tree['nodes'][nkey] for nkey in list(
                    c_sol_tree['nodes'].keys()) if nkey == es_sub_id][0]
                # collect all children
                node_list = []
                node_list = get_nodes(
                    es_sub_node, c_sol_tree['nodes'], node_list)
                # create a new sub tree with q_q
                temp_question_node = first_child_type[0].copy()
                temp_question_node['params']['Question']['value'] = q_q+';'
                node_list.append(temp_question_node)
                node_list.append(a_sub)
                node_list, root_node_id = clean_uuid(node_list, a_sub['id'])
                sub_trees.append([node_list, root_node_id])
    adaptedSolution = generate_solution(
        empty_solution(neighbours[0]['Solution']), sub_trees)
    return adaptedSolution


def generate_preference_dict(men, women):
    """
    Generates a preference dictionary based on the semantic similarity of names using the S-Bert model.

    Arguments:
    names -- a list of names

    Returns:
    A dictionary mapping each name to a list of names in order of preference
    """
    pref_dict = {}
    for man in men:
        # Calculate the semantic similarity between the current name and all other names
        distances = [(woman, getSimilaritySemanticSBERT(
            man['question'], woman['question'])) for woman in women]

        # Sort the distances in descending order and convert to a list of items
        preferences = [item for item, distance in sorted(
            distances, key=lambda x: x[1], reverse=True)]

        pref_dict[man['id']] = preferences
    return pref_dict


def stable_marriage(men, women):
    """
    Returns a stable matching between men and women using the Gale-Shapley algorithm,
    where preferences are determined by the semantic similarity of names using S-Bert.

    Arguments:
    men -- a list of men's names
    women -- a list of women's names

    Returns:
    A dictionary mapping men's names to their matched partners' names
    """
    # Generate preference dictionaries based on the similarity of names
    men_prefs = generate_preference_dict(men, women)
    women_prefs = generate_preference_dict(women, men)

    # Initialize all men and women to be free and without partners
    free_men = set([x['id'] for x in men])
    free_women = set([x['id'] for x in women])
    matches = {}

    while free_men:
        # Choose a free man
        man = free_men.pop()

        # Get the man's preference list of women
        preferences = men_prefs[man]

        # Loop through the man's preferences and propose to the highest-ranked woman who is still free
        for woman in preferences:
            if woman['id'] in free_women:
                matches[man] = woman['id']
                free_women.remove(woman['id'])
                break
            else:
                # If the woman is not free, check if she prefers the man over her current partner
                # get the woman's current partner
                current_partner = [
                    k for k, v in matches.items() if v == woman['id']][0]
                if get_index_in_list(women_prefs[woman['id']], man) < get_index_in_list(women_prefs[woman['id']], current_partner):
                    matches[man] = woman['id']
                    matches[current_partner] = None
                    free_men.add(current_partner)
                    break
    return matches


def match(men, women):
    """
    Returns pairings of two lists using a stable marriage algo. and the average similarity of the pairings
    """
    pairings = stable_marriage(men, women)
    score = 0.0
    counter = 0
    total = 0.0
    for k, v in pairings.items():
        if k is not None:
            counter += 1
            if v is not None:
                total += getSimilaritySemanticSBERT(get_from_id(
                    k, men)['question'], get_from_id(v, women)['question'])
    if counter > 0:
        score = total / counter
    return pairings, score


def MATCH(cq, nn, i, alpha=0.8):
    """
    Recursion returns the first pairing between 'cq' and 'nn' whose average similarity score is above 'alpha'.
    Matching starts from the 'i'th entry in 'nn' and increases 'i' in each iteration until 'alpha' is exceeded
    or 'nn' is exhausted. Use 'i = 1' to start matching from the first entry in 'nn'.
    """
    nn_lst = get_cumulative_case_questions(nn, i)
    pairings, score = match(cq, nn_lst)
#     print(i, score, pairings)
    if score > alpha or i == len(nn):
        return pairings, score, i, get_intent_overlap(cq, nn_lst, pairings), nn_lst
    else:
        return MATCH(cq, nn, i+1, alpha)


'''
functions for constructive adaptation 
1. explainer applicability - get the list of explainers applicable for a use case with/without NL explanation
2. substitute explainer - get a list of replacement explainers based on use case applicability and search criteria ordered by similarity
3. substitute subtree - get a list of replacement subtrees based on use case applicability and search criteria ordered by similarity
'''
ANY_URI = 'http://www.w3id.org/iSeeOnto/explainer#Any'

ANY_ACCESS_URI = 'http://www.w3id.org/iSeeOnto/explainer#Any_access'

INTENTS = {
    "DEBUGGING": ["Is this the same outcome for similar instances?", "Is this instance a common occurrence?"],
    "TRANSPARENCY": ["What is the impact of feature X on the outcome?", "How does feature X impact the outcome?", "What are the necessary features that guarantee this outcome?", "Why does the AI system have given outcome A?", "Which feature contributed to the current outcome?", "How does the AI system respond to feature X?", "What is the goal of the AI system?", "What is the scope of the AI system capabilities?", "What features does the AI system consider?", "What are the important features for the AI system?", "What is the impact of feature X on the AI system?", "How much evidence has been considered to build the AI system?", "How much evidence has been considered in the current outcome?", "What are the possible outcomes of the AI system?", "What features are used by the AI system?"],
    "PERFORMANCE": ["How confident is the AI system with the outcome?", "Which instances get a similar outcome?", "Which instances get outcome A?", "What are the results when others use the AI System?", "How accurate is the AI system?", "How reliable is the AI system?", "In what situations does the AI system make errors?", "What are the limitations of the AI system?", "In what situations is the AI system likely to be correct?"],
    "COMPLIANCY": ["How well does the AI system capture the real-world?", "Why are instances A and B given different outcomes?"],
    "COMPREHENSIBILITY": ["How to improve the AI system performance?", "What does term X mean?", "What is the overall logic of the AI system?", "What kind of algorithm is used in the AI system?"],
    "EFFECTIVENESS": ["What would be the outcome if features X is changed to value V?", "What other instances would get the same outcome?", "How does the AI system react if feature X is changed?", "What is the impact of the current outcome?"],
    "ACTIONABILITY": ["What are the alternative scenarios available?", "What type of instances would get a different outcome?", "How can I change feature X to get the same outcome?", "How to get a different outcome?", "How to change the instance to get a different outcome?", "Why does the AI system have given outcome A not B?", "Which features need changed to get a different outcome?"]
}

COMPOSITE_NODES = ['Sequence', 'Priority', 'Supplement',
                   'Replacement', 'Variant', 'Complement']

INSERTION_COST = 1.
DELETION_COST = 1.
LEAVE_CHANGE = 1.
DEFAULT_COST = 100


def get_usecase_context(usecase):
    context = {}
    context["ai_task"] = usecase["settings"]["ai_task"]
    context["ai_method"] = usecase["settings"]["ai_method"]
    context["dataset_type"] = usecase["settings"]["dataset_type"]
    context["implementation"] = usecase["model"]["backend"]
    context["model_mode"] = "http://www.w3id.org/iSeeOnto/explainer#File_access" if usecase["model"]["mode"] == "file" else "http://www.w3id.org/iSeeOnto/explainer#URL_access" if usecase["model"]["mode"] == "api" else "http://www.w3id.org/iSeeOnto/explainer#Any_access"
    context["has_training_data"] = usecase["model"]["dataset_file"] is not None

    return context


def format_attr(attr, code, key, ontology_prop):
    if (code == 0):
        return ontology_prop[key][attr]
    elif (code == 1):
        if isinstance(attr, list):
            if isinstance(attr[-1], list):
                attr = attr[-1]
            if (len(attr) == 1):
                return format_attr(attr[-1], 0, key, ontology_prop)
            i = 0
            msg = ""
            while i < len(attr)-1:
                msg = msg+format_attr(attr[i], 0, key, ontology_prop)+", "
                i = i+1
            msg = msg[:-2]+" and " + \
                format_attr(attr[i], 0, key, ontology_prop)
            return msg
        else:
            return format_attr(attr, 0, key, ontology_prop)
    elif (code == 2):
        if isinstance(attr, list) and isinstance(attr[0], list):
            attr = [a[-1] for a in attr]
            return format_attr(attr, 1, key, ontology_prop)
    elif (code == 3):
        if isinstance(attr, list):
            if isinstance(attr[-1], list):
                attr = attr[-1]
        attr = attr[-1]
        return format_attr(attr, 0, key, ontology_prop)


def explainer_applicability(context, explainer, ontology_props, explain):
    flag, msg = True, ''
    if context["dataset_type"] != explainer["dataset_type"]:
        flag = False
        if explain:
            msg = msg+"\n- Dataset Type Mismatch: The model uses " + \
                format_attr(context["dataset_type"], 0, "DatasetType", ontology_props) + \
                " data but " + explainer["name"] + \
                " only supports " + \
                format_attr(explainer["dataset_type"], 0,
                            "DatasetType", ontology_props) + " data."

    if ANY_URI not in explainer["implementation"] and context["implementation"] not in explainer["implementation"]:
        flag = False
        if explain:
            msg = msg+"\n- Implementation Mismatch: This is a " + \
                format_attr(context["implementation"], 0, "Implementation_Framework", ontology_props) + \
                " model but " + explainer["name"] + " only supports " + \
                format_attr(explainer["implementation"], 1,
                            "Implementation_Framework", ontology_props) + " implementations."

    if ANY_URI not in explainer["ai_methods"] and len(set(_i for i in context["ai_method"] for _i in i) & set(explainer["ai_methods"])) == 0:
        flag = False
        if explain:
            msg = msg+"\n- AI Method Mismatch: The model is a " + \
                format_attr(context["ai_method"], 2, "AIMethod", ontology_props) + \
                " but " + explainer["name"] + " only supports " + \
                format_attr(explainer["ai_methods"], 1,
                            "AIMethod", ontology_props) + "."

    if ANY_URI not in explainer["ai_tasks"] and len(set(context["ai_task"]) & set(explainer["ai_tasks"])) == 0:
        flag = False
        if explain:
            msg = msg+"\n- AI Task Mismatch: " + explainer["name"] + " does not support " + \
                format_attr(context["ai_task"], 3, "AITask",
                            ontology_props) + " tasks."
            
    if ANY_ACCESS_URI != explainer["model_access"] and explainer["model_access"] != context["model_mode"]:
        flag = False
        if explain:
            msg = msg+"\n- Model Access Mismatch: " + explainer["name"] + " does not support " + \
                context["model_mode"]+ " model access."

    if explainer["needs_training_data"] == "true" and not context["has_training_data"]:
        flag = False
        if explain:
            msg = msg+"\n- Explainer requires training data."
    return flag, msg


def explainers_applicability(context, explainer_props, ontology_props, explain):
    result = {}
    for e_props in explainer_props:
        flag, msg = explainer_applicability(
            context, e_props, ontology_props, explain)
        result[e_props["name"]] = {'flag': flag,
                                   'message': msg}
    return result


def applicability(data=None):
    if data is None:
        return {}

    ontology_support = data.get("ontology_props")
    query_case = data.get("query_case")
    explain = data.get("explain") == 'true'

    if ontology_support is None:
        return {}

    explainer_props = ontology_support["explainer_props"]
    ontology_props = ontology_support["ontology_props"]
    usecase_context = get_usecase_context(query_case)
    result = explainers_applicability(
        usecase_context, explainer_props, ontology_props, explain)
    return result


def nlg_explainer_batch(query, others, ontology_props):
    results = {}
    for other in others:
        results[other["name"]] = nlg_explainer(query, other, ontology_props)
    return results


def nlg_explainer(ex1, ex2, ontology_props):
    explanation = ""

    explanation = "Explainers are similar because "
    if ex1['dataset_type'] == ex2['dataset_type']:
        explanation = explanation + "they can be applied to the same dataset type: " + \
            ontology_props["DatasetType"][ex1['dataset_type']] + " data"
    if ex1['concurrentness'] == ex2['concurrentness']:
        explanation = explanation + ', ' + "they have the same concurrentness: " + \
            ontology_props["Concurrentness"][ex1['concurrentness']]
    if ex1['scope'] == ex2['scope']:
        explanation = explanation + ', ' + "they have the same scope: " + \
            ontology_props["Scope"][ex1['scope']]
    if ex1['portability'] == ex2['portability']:
        explanation = explanation + ', ' + "they have the same portability: " + \
            ontology_props["Portability"][ex1['portability']]
    if ex1['target'] == ex2['target']:
        explanation = explanation + ', ' + "they have the same target type: " + \
            ontology_props["Target"][ex1['target']]
    if ex1['computational_complexity'] == ex2['computational_complexity']:
        explanation = explanation + ', ' + "they have the same computational complexity: " + \
            ontology_props["ComputationalComplexity"][ex1['computational_complexity']]
    if ex1['model_access'] == ex2['model_access'] :
        explanation = explanation + ', ' + "they support the same model access type: " + \
            ontology_props["ModelAccess"][ex1['model_access']]
    if ex1['needs_training_data'] == ex2['needs_training_data'] :
        explanation = explanation + ', ' + "they both have the same training data requirements"
    
    technique = nlg_complex(ex1['technique'], ex2['technique'],
                            "they are the same explainability technique type: ", ontology_props['ExplainabilityTechnique']).strip()
    explanation = explanation + (', ' + technique if technique else '')
    explanation_type = nlg_complex(ex1['explanation_type'], ex2['explanation_type'],
                                   "they show the same explanation type: ", ontology_props['Explanation']).strip()
    explanation = explanation + \
        (', ' + explanation_type if explanation_type else '')
    implementation = nlg_complex(ex1['implementation'], ex2['implementation'],
                                 "they use the same backend: ", ontology_props['Implementation_Framework'], True).strip()
    implementation = implementation + (', ' + technique if technique else '')

    presentation = nlg_complex_multi(ex1['presentations'], ex2['presentations'],
                                     "they show the explanation with the same output type: ", ontology_props['InformationContentEntity']).strip()
    explanation = explanation + (', ' + presentation if presentation else '')
    ai_methods = nlg_complex_multi(ex1['ai_methods'], ex2['ai_methods'],
                                   "they are applicable to the same AI method type: ", ontology_props['AIMethod']).strip()
    explanation = explanation + (', ' + ai_methods if ai_methods else '')
    ai_tasks = nlg_complex_multi(ex1['ai_tasks'], ex2['ai_tasks'],
                                 "and they are applicable to the same AI task type: ", ontology_props['AITask']).strip()
    explanation = explanation + (', ' + ai_tasks if ai_tasks else '')
    return explanation


def nlg_complex(v1, v2, pretext, ontology, isBackend=False):
    overlap = [x for x in v1 if x in v2]
    if overlap:
        if isBackend == False:
            return pretext + ontology[overlap[-1]]
        else:
            return pretext + ','.join([ontology[o] for o in overlap])
    return ""


def nlg_complex_multi(v1, v2, pretext, ontology):
    overlaps = set()
    for i in v1:
        for j in v2:
            if len(i) == len(j):
                overlap = [i[-1]] if i[-1] == j[-1] else []
                if overlap:
                    overlaps.add(overlap[-1])
            elif len(i) > len(j):
                overlap = [j[-1]] if j[-1] in i else []
                if overlap:
                    overlaps.add(overlap[-1])
    if overlaps:
        return pretext + ','.join([ontology[o] for o in overlaps])
    return ""


def filter_explainers_by_criteria(explainer_props, criteria):
    filtered = []
    for explainer in explainer_props:
        _match = True
        for c, c_prop in criteria.items():
            _match = _match and match_prop(c_prop, explainer[c])
        if _match:
            filtered.append(explainer)
    return filtered


def match_prop(criteria_prop, explainer_prop):
    if criteria_prop == [ANY_URI]:
        return True
    elif type(explainer_prop) is list:
        overlap = [x for x in criteria_prop if x in explainer_prop]
        if overlap:
            return True
        else:
            return False
    else:
        overlap = [x for x in criteria_prop if x == explainer_prop]
        if overlap:
            return True
        else:
            return False


def replace_explainer(data):
    if data is None:
        return {}

    ontology_support = data.get("ontology_props")
    query_case = data.get("query_case")
    explain = data.get("explain") == 'true'
    query_explainer = data.get("query_explainer")
    criteria = data.get("criteria")

    if ontology_support is None:
        return {}

    explainer_props = ontology_support["explainer_props"]
    explainer_props_extended = ontology_support["explainer_props_extended"]
    similarities = ontology_support["similarities"]
    ontology_props = ontology_support["ontology_props"]

    usecase_context = get_usecase_context(query_case)
    applicabilities = explainers_applicability(
        usecase_context, explainer_props, ontology_props, False)

    similarities = similarities[query_explainer]
    query_explainer_props_extended = [
        e for e in explainer_props_extended if e["name"] == query_explainer][0]
    explainer_props_filtered = [e for e in explainer_props if (
        applicabilities[e["name"]]["flag"] and e["name"] != query_explainer)]
    if criteria:
        explainer_props_filtered = filter_explainers_by_criteria(
            explainer_props_filtered, criteria)
        similarities = {k: s for k, s in similarities.items() if (
            (k in [e["name"] for e in explainer_props_filtered]) and (k != query_explainer))}
    else:
        explainer_props_filtered = explainer_props_filtered
        similarities = {k: s for k, s in similarities.items()
                        if k != query_explainer}

    explainers_props_extended_filtered = [e for e in explainer_props_extended if e["name"] in [
        f["name"] for f in explainer_props_filtered]]
    nlg_result = nlg_explainer_batch(query_explainer_props_extended,
                           explainers_props_extended_filtered, ontology_props) if explain else {}

    result = [{"explainer": e["name"],
               "explanation":nlg_result[e["name"]] if e["name"] in nlg_result else "",
               "similarity":similarities[e["name"]]
               } for e in explainer_props_filtered]

    result_sorted = sorted(result, key=lambda x: x["similarity"], reverse=True)

    return result_sorted


def bt_sequence(tree, node, adj_node, seq):
    seq.append(node)
    if adj_node:
        for child in adj_node:
            bt_sequence(tree, tree["nodes"][child], tree["adj"][child], seq)


def edit_distance(q, c, delta):
    s1 = []
    bt_sequence(q, q["nodes"][0], q["adj"][0], s1)
    s2 = []
    bt_sequence(c, c["nodes"][0], c["adj"][0], s2)
    dist = sed.sed(s1, s2, delta)
    return dist


def semantic_delta_parent(sims):
    def semantic_delta(x, y):
        if x == y:
            _dist = 0.
        elif (x != None and y == None):
            _dist = INSERTION_COST
        elif (x == None and y != None):
            _dist = DELETION_COST
        elif (x == 'r' or y == 'r'):
            _dist = np.inf
        elif (x in COMPOSITE_NODES and y in COMPOSITE_NODES):
            _dist = 0.
        elif (x in COMPOSITE_NODES or y in COMPOSITE_NODES):
            _dist = np.inf
        elif x in sims and y in sims:
            _dist = 1-sims[x][y]
        elif (x in sims and y in sims) or (x not in sims and y in sims):
            _dist = np.inf
        elif typeQuestion(x) != "NO_QUESTION" and typeQuestion(y) != "NO_QUESTION":
            if typeQuestion(x) == typeQuestion(y):
                _dist = 0.75
            else:
                _dist = 0.5
        else:
            return DEFAULT_COST
        return _dist
    return semantic_delta


def typeQuestion(question):
    question_type = [key for key in INTENTS.keys() if question in INTENTS[key]]
    if question_type == []:
        return "NO_QUESTION"
    else:
        return question_type[0]


def print_node_instances(node_id, nodes_dict, node_list, id_list):
    node = nodes_dict[node_id]

    node_instance = node['Instance']
    if node_instance is None:
        return None
    elif node_instance == "User Question":
        node_instance = node["params"]["Question"]["value"]
    node_list.append(node_instance)
    id_list.append(node_id)

    if 'firstChild' in node:
        first_child_id = node['firstChild']['Id']
        print_node_instances(first_child_id, nodes_dict, node_list, id_list)
        next_child = node['firstChild'].get('Next')

        while next_child is not None:
            next_child_id = next_child['Id']
            print_node_instances(next_child_id, nodes_dict, node_list, id_list)
            next_child = next_child.get('Next')

    return node_list, id_list


def get_index(node_id, nodes_dict, id_list):
    node = nodes_dict[node_id]
    node_instance = node.get('Instance')
    node_index = id_list.index(node_id)
    node_index = node_index + 1

    return node_index, node_instance


def find_parent(node_id, node, parent_child_dict, id_list, nodes_dict):
    parent_index, parent_instance = get_index(node_id, nodes_dict, id_list)

    if 'firstChild' in node:
        first_child_id = node['firstChild']['Id']
        child_index, child_instance = get_index(
            first_child_id, nodes_dict, id_list)

        if parent_index not in parent_child_dict:
            parent_child_dict[parent_index] = []
        if child_index not in parent_child_dict[parent_index]:
            parent_child_dict[parent_index].append(child_index)

        next_child = node['firstChild'].get('Next')
        while next_child is not None:
            next_child_id = next_child['Id']
            child_index, child_instance = get_index(
                next_child_id, nodes_dict, id_list)
            if child_index not in parent_child_dict[parent_index]:
                # Add child index to the parent's list
                parent_child_dict[parent_index].append(child_index)
            next_child = next_child.get('Next')

        return parent_instance


def create_parent_child_dict(nodes_dict, node_list, id_list):
    parent_child_dict = {}
    parent_child_dict[0] = [1]

    for i, (instance, node_id) in enumerate(zip(node_list[1:], id_list), start=1):
        node_index = i
        node_id = id_list[node_index-1]
        node = nodes_dict[node_id]
        find_parent(node_id, node, parent_child_dict, id_list, nodes_dict)

    return parent_child_dict


def build_adjacency_list(node_list, parent_child_dict):
    adjacency_list = [[] for _ in range(len(node_list))]

    for node_index, node_instance in enumerate(node_list):
        if node_index in parent_child_dict:
            children = parent_child_dict[node_index]
            adjacency_list[node_index] = children

    return adjacency_list


def convert_to_graph(cases):
    tree_dict, nodes_dict, parent_child_dict = {}, {}, {}
    node_list = ['r']  # Added 'r' as the default root node in the node list
    id_list = []  # List of node id's

    for idx, obj in enumerate(cases, start=1):
        trees = obj['data']['trees']

        for tree in trees:
            nodes = tree.get('nodes', {})
            nodes_dict.update(nodes)
            root_node_id = tree.get('root')

        # Call the recursive function to print node instances
        node_list, id_list = print_node_instances(
            root_node_id, nodes_dict, node_list=['r'], id_list=[])

        # Call the function to create the parent_child dictionary
        parent_child_dict = create_parent_child_dict(
            nodes_dict, node_list, id_list)

        # Build the adjacency list from the behavior tree
        adjacency_list = build_adjacency_list(node_list, parent_child_dict)

        tree_key = f'tree_{idx}'
        tree_dict[tree_key] = {
            'complete_json': obj,
            'tree_json': trees,
            'tree_graph': {
                'nodes': node_list,
                'adj': adjacency_list
            }
        }

    return tree_dict


def check_applicability(bt_graph, applicabilities):
    applicability = True
    my_nodes = bt_graph["nodes"]
    i = 0
    while applicability and i < len(my_nodes):
        node = my_nodes[i]
        if node[0] == '/':
            applicability = applicability and applicabilities[node]["flag"]
        i = i + 1
    return applicability


def filter_trees_by_criteria(matching_explainers, tree):
    tree_match = False
    if 'tree_graph' in tree:
        graph = tree['tree_graph']
        if 'nodes' in graph:
            nodes = graph['nodes']
            common_explainers = list(set(nodes) & set(matching_explainers))
            if common_explainers != []:
                tree_match = True

    return tree_match


def remove_root(_tree):
    _tree_ = copy.deepcopy(_tree)

    for tree in _tree_:
        root_id = tree.get('root')

        if tree['root'] == root_id:
            del tree['nodes'][root_id]
            del tree['root']
            break  # Assuming there is only one tree with the specified root

    most_similar_subtree = _tree_[0]['nodes']
    return most_similar_subtree


def search_and_remove(original_tree, target_id):
    modified_tree = copy.deepcopy(original_tree)
    nodes = modified_tree['trees'][0]['nodes']
    target_node = nodes.get(target_id)
    if target_node["id"] == target_id:
        children_ids = extract_children_ids(target_node)
        del nodes[target_id]
        for child_id in children_ids:
            modified_tree = search_and_remove(modified_tree, child_id)
    return modified_tree


def extract_children_ids(node):
    child_nodes = []
    if "firstChild" in node:
        child_nodes.append(node["firstChild"]['Id'])
        next_child = node['firstChild'].get('Next')
        while next_child is not None:
            child_nodes.append(next_child['Id'])
            next_child = next_child.get('Next')
    return child_nodes


def get_parent_node(node_id, nodes):
    for parent_node_id, node_data in nodes.items():
        if "firstChild" in node_data and node_data["firstChild"]["Id"] == node_id:
            return parent_node_id
        if "Next" in node_data and node_data["Next"]["Id"] == node_id:
            return parent_node_id
    for parent_node_id, node_data in nodes.items():
        if "id" in node_data:
            parent = get_parent_node(node_id, node_data)
            if parent:
                parent = node_data['id']
                return parent
    return None


def substitute_node(node, target_id, new_node):
    if isinstance(node, dict):
        if "id" in node and node.get("id") == target_id:
            return new_node
        if "firstChild" in node:
            if node["firstChild"]["Id"] == target_id:
                node["firstChild"]["Id"] = new_node
            else:
                next_child = node['firstChild'].get('Next')
                while next_child is not None:
                    if next_child["Id"] == target_id:
                        next_child["Id"] = new_node
                    else:
                        next_child = next_child.get('Next')
    return node


def get_modified_case(query_tree, query_subtree, solution_tree):
    selected_node = query_subtree['trees'][0]['root']
    modified_tree = copy.deepcopy(query_tree)
    modified_tree = search_and_remove(modified_tree, selected_node)
    solution_root = solution_tree[0]['root']

    parent = get_parent_node(selected_node,
                             modified_tree['trees'][0]['nodes'])
    if parent:
        parent_node = modified_tree['trees'][0]['nodes'][parent]
        substitute_node(
            parent_node, selected_node, solution_root)
    else:
        modified_tree['trees'][0]['root'] = solution_root

    modified_tree['trees'][0]['nodes'].update(solution_tree[0]['nodes'])
    return modified_tree


def filter_nodes(node, nodes, result):
    result[node["id"]] = node
    if "firstChild" in node:
        children = node["firstChild"]
        result[children["Id"]] = nodes[children["Id"]]
        if "Next" in children:
            filter_nodes(nodes[children["Next"]["Id"]], nodes, result)
        return
    return


def find_subtree(_tree, _node_id):
    parent_tree = copy.deepcopy(_tree)
    for tree in parent_tree["data"]["trees"]:
        nodes = tree.get('nodes', {})

        selected_node = [n for k, n in nodes.items() if k == _node_id]
        if selected_node:
            result = {}
            filter_nodes(selected_node[0], nodes, result)
            tree['nodes'] = result
            tree['root'] = selected_node[0]["id"]
        else:
            continue
    return parent_tree

def nlg_subtree(q_nodes, s_nodes):
    print(q_nodes)
    print(s_nodes)
    return ""

def replace_subtree(data):
    if data is None:
        return {}
    ontology_support = data.get("ontology_props")
    query_case = data.get("query_case")
    explain = data.get("explain") == 'true'
    query_subtree_id = data.get("query_subtree")
    query_tree = data.get("query_tree")
    neighbours = data.get("neighbours")
    criteria = data.get("criteria")

    if ontology_support is None:
        return {}

    explainer_props = ontology_support["explainer_props"]
    explainer_props_extended = ontology_support["explainer_props_extended"]
    similarities = ontology_support["similarities"]
    ontology_props = ontology_support["ontology_props"]

    usecase_context = get_usecase_context(query_case)
    applicabilities = explainers_applicability(
        usecase_context, explainer_props, ontology_props, False)

    tree_dict = convert_to_graph(neighbours)
    tree_dict_filtered = dict()
    for key, tree in tree_dict.items():
        if check_applicability(tree['tree_graph'], applicabilities):
            if criteria:
                if "explainer" in criteria:
                    explainers_filtered = [e for e in explainer_props if e["name"] in criteria["explainer"]]
                else:
                    explainers_filtered = filter_explainers_by_criteria(
                        explainer_props, criteria)
                if filter_trees_by_criteria([e["name"] for e in explainers_filtered], tree):
                    tree_dict_filtered[key] = tree
            else:
                tree_dict_filtered[key] = tree

    query_subtree = [find_subtree(query_tree, query_subtree_id)]
    query_subtree_graph = convert_to_graph(
        query_subtree)['tree_1']['tree_graph']

    solution = {}
    for bt in tree_dict_filtered:
        tree_case = tree_dict_filtered[bt]['tree_graph']
        if query_subtree_graph != tree_case: 
            edit_distance_value = edit_distance(
                query_subtree_graph, tree_case, semantic_delta_parent(similarities))
            if edit_distance_value != 0:
                solution[bt] = edit_distance_value

    sorted_BTs = sorted(solution.items(), key=lambda x: x[1])
    results = []
    k = min(len(sorted_BTs), data.get("k"))

    for key in range(k):
        solution_graph_format = sorted_BTs[key][0]
        solution_json = tree_dict_filtered[solution_graph_format]['tree_json']
        # solution_no_root = remove_root(solution_json)
        modified_tree = get_modified_case(
            query_tree["data"], query_subtree[0]["data"], solution_json)
    
        tree_dict_filtered[solution_graph_format]["complete_json"]["data"] = modified_tree
        tree_dict_filtered[solution_graph_format]["complete_json"]["explanation"] = ""
        results.append(tree_dict_filtered[solution_graph_format]["complete_json"])

    return results


def substitute(data):
    if data is None:
        return {}

    ontology_support = data.get("ontology_props")
    query_case = data.get("query_case")

    if ontology_support is None or query_case is None:
        return {}

    if data.get("query_explainer"):
        return replace_explainer(data)
    elif data.get("query_subtree"):
        return replace_subtree(data)
    return {}
